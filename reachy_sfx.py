#!/usr/bin/env python3
"""
reachy_sfx.py - Vibey's dashboard soundboard.

Short, original, synthesized robot/space-opera effects. No samples, no
downloaded assets, no franchise audio: just little sine/noise spells uploaded
to the robot speaker on first use.
"""

from __future__ import annotations

import io
import json
import math
import os
import random
import struct
import threading
import urllib.request
import uuid
import wave

from reachy_voice import load_env

load_env()

REACHY_URL = os.environ.get("REACHY_URL", "http://192.168.1.120:8000").rstrip("/")
SR = 22050

SFX = [
    {"name": "laser_pew", "label": "Laser Pew", "icon": "Pew", "group": "space"},
    {"name": "laser_burst", "label": "Triple Blaster", "icon": "3x", "group": "space"},
    {"name": "saber_on", "label": "Light Blade On", "icon": "On", "group": "space"},
    {"name": "saber_swing", "label": "Light Blade Swing", "icon": "Swish", "group": "space"},
    {"name": "hyperjump", "label": "Hyperjump", "icon": "Warp", "group": "space"},
    {"name": "shield_up", "label": "Shield Up", "icon": "Shield", "group": "space"},
    {"name": "tractor_beam", "label": "Tractor Beam", "icon": "Beam", "group": "space"},
    {"name": "airlock", "label": "Airlock Door", "icon": "Door", "group": "space"},
    {"name": "droid_yes", "label": "Droid Yes", "icon": "Yes", "group": "droid"},
    {"name": "droid_no", "label": "Droid No", "icon": "No", "group": "droid"},
    {"name": "droid_gossip", "label": "Droid Gossip", "icon": "Talk", "group": "droid"},
    {"name": "scanner", "label": "Scanner Sweep", "icon": "Scan", "group": "droid"},
    {"name": "cantina", "label": "Tiny Cantina", "icon": "Band", "group": "music"},
    {"name": "success", "label": "Quest Complete", "icon": "Win", "group": "mood"},
    {"name": "fail", "label": "Sad Trombone Bot", "icon": "Womp", "group": "mood"},
    {"name": "mischief", "label": "Mischief", "icon": "Hmm", "group": "mood"},
]


def catalog() -> list[dict]:
    return SFX


def _sine(freq: float, dur: float, vol: float = 0.35,
          bend: float = 0.0, vibrato: float = 0.0) -> list[float]:
    n = int(SR * dur)
    phase = 0.0
    out = []
    for i in range(n):
        t = i / SR
        frac = i / max(1, n - 1)
        f = freq + bend * frac
        if vibrato:
            f += vibrato * math.sin(2 * math.pi * 12 * t)
        phase += 2 * math.pi * f / SR
        env = min(1.0, i / (SR * 0.008), (n - i) / (SR * 0.025))
        out.append(vol * env * math.sin(phase))
    return out


def _noise(dur: float, vol: float = 0.18, seed: int = 1) -> list[float]:
    rnd = random.Random(seed)
    n = int(SR * dur)
    out = []
    last = 0.0
    for i in range(n):
        last = last * 0.72 + rnd.uniform(-1, 1) * 0.28
        env = min(1.0, i / (SR * 0.01), (n - i) / (SR * 0.04))
        out.append(last * vol * env)
    return out


def _silence(dur: float) -> list[float]:
    return [0.0] * int(SR * dur)


def _mix(*tracks: list[float]) -> list[float]:
    n = max((len(t) for t in tracks), default=0)
    out = [0.0] * n
    for t in tracks:
        for i, s in enumerate(t):
            out[i] += s
    return [max(-1.0, min(1.0, s)) for s in out]


def _overlay(base: list[float], sound: list[float], at: float) -> list[float]:
    start = int(SR * at)
    need = start + len(sound)
    if len(base) < need:
        base.extend([0.0] * (need - len(base)))
    for i, s in enumerate(sound):
        base[start + i] += s
    return base


def _to_wav(samples: list[float]) -> bytes:
    samples = [max(-1.0, min(1.0, s)) for s in samples]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(b"".join(struct.pack("<h", int(s * 32767)) for s in samples))
    return buf.getvalue()


def _effect(name: str) -> bytes:
    if name == "laser_pew":
        s = _mix(_sine(1550, .16, .45, bend=-1150), _noise(.12, .06, 11))
    elif name == "laser_burst":
        s = []
        for at in (0.0, .14, .28):
            _overlay(s, _mix(_sine(1450, .11, .38, bend=-950), _noise(.09, .05, 12)), at)
    elif name == "saber_on":
        s = _mix(_sine(90, .65, .34, bend=30), _sine(180, .65, .16, bend=50),
                 _noise(.65, .07, 21))
        s = _overlay(s, _sine(740, .22, .22, bend=220), .03)
    elif name == "saber_swing":
        s = _mix(_sine(120, .42, .30, bend=85), _sine(380, .34, .18, bend=-160),
                 _noise(.36, .10, 22))
    elif name == "hyperjump":
        s = _mix(_sine(120, .75, .24, bend=1600), _sine(240, .72, .16, bend=2100),
                 _noise(.7, .07, 31))
        s = _overlay(s, _sine(2100, .16, .32, bend=-1400), .62)
    elif name == "shield_up":
        s = _mix(_sine(260, .55, .25, bend=320), _sine(520, .55, .16, bend=480),
                 _sine(1040, .42, .10, vibrato=35))
    elif name == "tractor_beam":
        s = _mix(_sine(85, .9, .30, bend=-18, vibrato=9),
                 _sine(170, .9, .13, bend=-35, vibrato=14), _noise(.9, .05, 41))
    elif name == "airlock":
        s = _mix(_sine(70, .55, .26, bend=-18), _noise(.7, .12, 51))
        s = _overlay(s, _sine(410, .09, .23, bend=-80), .58)
    elif name == "droid_yes":
        s = _sine(520, .08, .35) + _sine(760, .09, .36) + _sine(1060, .12, .35)
    elif name == "droid_no":
        s = _sine(480, .12, .33) + _silence(.04) + _sine(300, .22, .34, bend=-30)
    elif name == "droid_gossip":
        s = []
        notes = [720, 980, 610, 1210, 840, 560, 1040, 700, 1320]
        for i, f in enumerate(notes):
            _overlay(s, _sine(f, .055, .24, bend=random.Random(i).choice([-90, 80])), i * .07)
    elif name == "scanner":
        s = _mix(_sine(420, .9, .18, bend=760, vibrato=10), _sine(1180, .9, .08, bend=-500))
        for at in (.12, .28, .44, .60, .76):
            _overlay(s, _sine(1800, .035, .18, bend=-250), at)
    elif name == "cantina":
        s = []
        melody = [(440, .12), (554, .12), (659, .12), (554, .12),
                  (440, .12), (370, .12), (440, .22)]
        at = 0.0
        for f, d in melody:
            _overlay(s, _mix(_sine(f, d, .22), _sine(f * 2, d, .06)), at)
            at += d + .025
    elif name == "success":
        s = _sine(523, .09, .28) + _sine(659, .09, .28) + _sine(784, .09, .30) + _sine(1046, .20, .34)
    elif name == "fail":
        s = _sine(330, .18, .30, bend=-20) + _sine(294, .18, .30, bend=-20) + _sine(262, .32, .32, bend=-70)
    elif name == "mischief":
        s = _sine(300, .09, .22) + _silence(.05) + _sine(450, .11, .25, bend=70) + _silence(.04) + _sine(390, .18, .24, vibrato=18)
    else:
        raise KeyError(name)
    return _to_wav(s)


_uploaded: set[str] = set()
_lock = threading.Lock()


def _filename(name: str) -> str:
    return f"wonder_sfx_board_{name}.wav"


def _upload(name: str) -> bool:
    with _lock:
        if name in _uploaded:
            return True
        wav = _effect(name)
        filename = _filename(name)
        boundary = f"----sfx{uuid.uuid4().hex}"
        payload = ((f"--{boundary}\r\nContent-Disposition: form-data; "
                    f'name="file"; filename="{filename}"\r\n'
                    f"Content-Type: audio/wav\r\n\r\n").encode()
                   + wav + f"\r\n--{boundary}--\r\n".encode())
        req = urllib.request.Request(
            f"{REACHY_URL}/api/media/sounds/upload", data=payload, method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=15) as r:
            r.read()
        _uploaded.add(name)
        return True


def play(name: str) -> bool:
    if name not in {s["name"] for s in SFX}:
        return False
    if not _upload(name):
        return False
    req = urllib.request.Request(
        f"{REACHY_URL}/api/media/play_sound",
        data=json.dumps({"file": _filename(name)}).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        r.read()
    return True


if __name__ == "__main__":
    import sys
    choice = sys.argv[1] if len(sys.argv) > 1 else "laser_pew"
    print(f"[sfx] playing {choice}")
    play(choice)
