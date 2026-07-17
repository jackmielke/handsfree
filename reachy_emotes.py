#!/usr/bin/env python3
"""
reachy_emotes.py — Vibey's emotional vocabulary: 5 emotes, motion + sound.

Each emote is a short head/antenna choreography (via the daemon's /move/goto)
plus a matching R2-style chirp. The chirps are synthesized right here (sine
sweeps, stdlib `wave` only) and uploaded to the robot's speaker on first use —
no audio assets to download or commit.

    happy    perk + side-to-side wiggle          rising major arpeggio
    excited  double head-bounce, antennas flared  fast double up-sweep
    curious  head tilt, one antenna up            "hmm?" bend with vibrato
    sad      slow droop, antennas fall            long falling sweep
    smug     look away + up-tilt, half antenna    two low deadpan blips

Use as a module (non-blocking):
    from reachy_emotes import play
    play("happy")                # motion only (safe while Vibey is speaking)
    play("happy", sound=True)    # motion + chirp

Or from the CLI:
    python3 reachy_emotes.py happy
"""

from __future__ import annotations

import io
import json
import math
import os
import struct
import threading
import time
import urllib.request
import uuid
import wave

from reachy_voice import load_env

load_env()

REACHY_URL = os.environ.get("REACHY_URL", "http://192.168.12.240:8000").rstrip("/")

NEUTRAL = {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}

EMOTIONS = ["happy", "excited", "curious", "sad", "smug", "thinking", "victory"]


# --------------------------------------------------------------------------- #
# Robot REST helpers                                                          #
# --------------------------------------------------------------------------- #
def _post(path: str, body: dict | None = None, timeout: float = 8.0):
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(f"{REACHY_URL}{path}", data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"null")
    except Exception as e:  # noqa: BLE001 - robot hiccups shouldn't crash callers
        print(f"[emote] POST {path} failed: {e}", flush=True)
        return None


def _goto(head: dict, antennas: list[float], duration: float):
    _post("/api/move/goto",
          {"head_pose": head, "antennas": antennas, "duration": duration})


def _pose(**kw) -> dict:
    return dict(NEUTRAL, **kw)


# --------------------------------------------------------------------------- #
# Chirp synthesis — tiny sine sweeps with a soft envelope, 16-bit mono WAV.   #
# --------------------------------------------------------------------------- #
SR = 22050


def _sweep(f0: float, f1: float, dur: float, vol: float = 0.5,
           vibrato: float = 0.0) -> list[float]:
    n = int(SR * dur)
    out = []
    for i in range(n):
        t = i / SR
        frac = i / max(1, n - 1)
        f = f0 + (f1 - f0) * frac
        if vibrato:
            f += vibrato * math.sin(2 * math.pi * 18 * t)
        # soft attack/release so chirps don't click
        env = min(1.0, i / (SR * 0.01), (n - i) / (SR * 0.03))
        out.append(vol * env * math.sin(2 * math.pi * f * t))
    return out


def _silence(dur: float) -> list[float]:
    return [0.0] * int(SR * dur)


def _to_wav(samples: list[float]) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(b"".join(
            struct.pack("<h", int(max(-1, min(1, s)) * 32767)) for s in samples))
    return buf.getvalue()


def _chirp(emotion: str) -> bytes:
    if emotion == "happy":       # rising major arpeggio
        s = _sweep(523, 523, .09) + _sweep(659, 659, .09) + _sweep(784, 784, .14)
    elif emotion == "excited":   # fast double up-sweep
        s = _sweep(500, 1400, .16) + _silence(.05) + _sweep(600, 1600, .18, .6)
    elif emotion == "curious":   # questioning bend with vibrato
        s = _sweep(600, 500, .12) + _sweep(500, 950, .28, .45, vibrato=25)
    elif emotion == "sad":       # long falling sweep
        s = _sweep(700, 220, .7, .4)
    elif emotion == "thinking":   # slow "hmm" wobble
        s = (_sweep(440, 480, .18, .35, vibrato=8) + _silence(.07)
             + _sweep(480, 420, .22, .3, vibrato=12) + _silence(.06)
             + _sweep(420, 460, .18, .25, vibrato=10))
    elif emotion == "victory":    # ascending fanfare + triumphant burst
        s = (_sweep(523, 784, .12) + _sweep(784, 1046, .12)
             + _sweep(1046, 1046, .08) + _silence(.04)
             + _sweep(880, 1318, .2, .65) + _silence(.05)
             + _sweep(1046, 1568, .25, .7))
    else:                        # smug — two low deadpan blips
        s = _sweep(330, 320, .1, .45) + _silence(.09) + _sweep(280, 270, .14, .45)
    return _to_wav(s)


_uploaded: set[str] = set()
_upload_lock = threading.Lock()


def _sound_name(emotion: str) -> str:
    return f"wonder_sfx_{emotion}.wav"


def _ensure_sound(emotion: str) -> bool:
    with _upload_lock:
        if emotion in _uploaded:
            return True
        wav = _chirp(emotion)
        boundary = f"----emote{uuid.uuid4().hex}"
        name = _sound_name(emotion)
        payload = ((f"--{boundary}\r\nContent-Disposition: form-data; "
                    f'name="file"; filename="{name}"\r\n'
                    f"Content-Type: audio/wav\r\n\r\n").encode()
                   + wav + f"\r\n--{boundary}--\r\n".encode())
        req = urllib.request.Request(
            f"{REACHY_URL}/api/media/sounds/upload", data=payload, method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                r.read()
            _uploaded.add(emotion)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[emote] sfx upload failed: {e}", flush=True)
            return False


# --------------------------------------------------------------------------- #
# The 5 choreographies                                                        #
# --------------------------------------------------------------------------- #
def _do_happy():
    _goto(_pose(roll=0.15), [0.9, -0.9], 0.25); time.sleep(0.27)
    _goto(_pose(roll=-0.15), [0.7, -0.7], 0.25); time.sleep(0.27)
    _goto(_pose(roll=0.12), [0.9, -0.9], 0.22); time.sleep(0.24)
    _goto(NEUTRAL, [0.2, -0.2], 0.3)


def _do_excited():
    for _ in range(2):
        _goto(_pose(pitch=-0.25, z=0.01), [1.2, -1.2], 0.18); time.sleep(0.2)
        _goto(_pose(pitch=0.1), [0.5, -0.5], 0.18); time.sleep(0.2)
    _goto(NEUTRAL, [0.3, -0.3], 0.25)


def _do_curious():
    _goto(_pose(roll=0.35, pitch=-0.1), [1.0, 0.0], 0.5); time.sleep(0.9)
    _goto(_pose(roll=0.3, pitch=-0.15), [1.0, -0.3], 0.3); time.sleep(0.5)
    _goto(NEUTRAL, [0.0, 0.0], 0.4)


def _do_sad():
    _goto(_pose(pitch=0.35, z=-0.01), [-0.8, 0.8], 1.1); time.sleep(1.4)
    _goto(_pose(pitch=0.3), [-1.0, 1.0], 0.8); time.sleep(1.0)
    _goto(NEUTRAL, [0.0, 0.0], 0.9)


def _do_thinking():
    _goto(_pose(roll=0.38, pitch=-0.08), [0.9, 0.1], 0.6); time.sleep(1.0)
    _goto(_pose(roll=0.42, pitch=-0.1), [1.0, 0.05], 0.4); time.sleep(0.5)
    _goto(_pose(roll=0.35, pitch=-0.06), [0.85, 0.15], 0.35); time.sleep(0.6)
    _goto(NEUTRAL, [0.2, -0.2], 0.5)


def _do_victory():
    for _ in range(3):
        _goto(_pose(pitch=-0.28, z=0.015), [1.3, -1.3], 0.14); time.sleep(0.16)
        _goto(_pose(pitch=0.08), [0.3, -0.3], 0.14); time.sleep(0.16)
    _goto(_pose(roll=0.2, pitch=-0.2), [1.4, -0.2], 0.2); time.sleep(0.28)
    _goto(_pose(roll=-0.2, pitch=-0.2), [0.2, -1.4], 0.2); time.sleep(0.28)
    _goto(_pose(pitch=-0.3), [1.4, -1.4], 0.2); time.sleep(0.35)
    _goto(NEUTRAL, [0.3, -0.3], 0.35)


def _do_smug():
    _goto(_pose(yaw=0.5, pitch=-0.12), [0.6, 0.0], 0.6); time.sleep(1.0)
    _goto(_pose(yaw=0.45, pitch=-0.15), [0.75, 0.0], 0.3); time.sleep(0.6)
    _goto(NEUTRAL, [0.0, 0.0], 0.5)


_MOVES = {"happy": _do_happy, "excited": _do_excited, "curious": _do_curious,
          "sad": _do_sad, "smug": _do_smug, "thinking": _do_thinking,
          "victory": _do_victory}


def play(emotion: str, sound: bool = False) -> bool:
    """Fire an emote (non-blocking). Returns False for unknown emotions."""
    move = _MOVES.get(emotion)
    if not move:
        return False

    def _run():
        if sound and _ensure_sound(emotion):
            _post("/api/media/play_sound", {"file": _sound_name(emotion)})
        try:
            move()
        except Exception as e:  # noqa: BLE001
            print(f"[emote] {emotion} failed: {e}", flush=True)

    threading.Thread(target=_run, daemon=True).start()
    return True


# --------------------------------------------------------------------------- #
# Dance mode — a synthesized beat + a looping full-body groove.               #
# --------------------------------------------------------------------------- #
def _beat_track(seconds: float = 12.0, bpm: int = 118) -> bytes:
    """A tiny synthesized dance beat: sine-kick four-on-the-floor with an
    off-beat blip. No samples, no downloads — pure math, very robot."""
    spb = 60.0 / bpm
    total = int(SR * seconds)
    buf = [0.0] * total
    t = 0.0
    beat_i = 0
    while t < seconds:
        start = int(t * SR)
        # kick: 150→50 Hz thump
        for i, s in enumerate(_sweep(150, 50, 0.11, 0.85)):
            j = start + i
            if j < total:
                buf[j] += s
        # off-beat blip every other beat
        if beat_i % 2 == 1:
            for i, s in enumerate(_sweep(880, 860, 0.04, 0.18)):
                j = start + int(SR * spb / 2) + i
                if j < total:
                    buf[j] += s
        t += spb
        beat_i += 1
    return _to_wav([max(-1, min(1, s)) for s in buf])


def dance(seconds: float = 12.0, bpm: int = 118) -> None:
    """Dance mode: play the beat on the robot and groove until it ends.
    Blocking — call from a thread (play(..) style) if you need async."""
    name = "wonder_sfx_beat.wav"
    with _upload_lock:
        if "beat" not in _uploaded:
            wav = _beat_track(seconds, bpm)
            boundary = f"----emote{uuid.uuid4().hex}"
            payload = ((f"--{boundary}\r\nContent-Disposition: form-data; "
                        f'name="file"; filename="{name}"\r\n'
                        f"Content-Type: audio/wav\r\n\r\n").encode()
                       + wav + f"\r\n--{boundary}--\r\n".encode())
            req = urllib.request.Request(
                f"{REACHY_URL}/api/media/sounds/upload", data=payload,
                method="POST",
                headers={"Content-Type":
                         f"multipart/form-data; boundary={boundary}"})
            urllib.request.urlopen(req, timeout=15).read()
            _uploaded.add("beat")
    _post("/api/media/play_sound", {"file": name})

    spb = 60.0 / bpm
    end = time.time() + seconds
    moves = [
        lambda: _goto(_pose(roll=0.22, pitch=0.1), [0.9, -0.4], spb * 0.9),
        lambda: _goto(_pose(roll=-0.22, pitch=-0.08), [-0.4, 0.9], spb * 0.9),
        lambda: _goto(_pose(pitch=0.18, yaw=0.25), [1.0, -1.0], spb * 0.9),
        lambda: _goto(_pose(pitch=-0.15, yaw=-0.25), [0.5, 0.5], spb * 0.9),
    ]
    i = 0
    while time.time() < end:
        moves[i % len(moves)]()
        time.sleep(spb)
        i += 1
    _goto(NEUTRAL, [0.2, -0.2], 0.6)


def play_dance(seconds: float = 12.0) -> None:
    """Non-blocking dance mode."""
    threading.Thread(target=dance, args=(seconds,), daemon=True).start()


if __name__ == "__main__":
    import sys
    emotion = sys.argv[1] if len(sys.argv) > 1 else "happy"
    if emotion == "dance":
        secs = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
        print(f"[emote] dance mode for {secs:.0f}s 🕺")
        dance(secs)
        sys.exit(0)
    if emotion not in _MOVES:
        print(f"unknown emotion {emotion!r}; pick from {EMOTIONS} or 'dance'")
        sys.exit(1)
    print(f"[emote] playing {emotion} (with sound)")
    play(emotion, sound=True)
    time.sleep(5)  # let the daemon thread finish before the CLI exits
