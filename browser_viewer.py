"""handsfree viewer + gesture → music instrument.

Serves:
  /          HTML page with live MJPEG feed + Web Audio synth
  /stream    multipart MJPEG of the camera with landmarks drawn on
  /events    Server-Sent Events stream: motion + smile + clap triggers

Gesture wiring (v1):
  - clap (two hands close together)  → toggle beat on/off
  - body motion (landmark velocity)  → brightens the drum filter
  - smile (MediaPipe blendshape)     → fades in a warm major chord pad
"""
import http.server
import json
import math
import os
import re
import tempfile
import socketserver
import signal
import subprocess
import threading
import time
import urllib.request
import webbrowser
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

try:
    from Quartz import (
        CGEventCreateKeyboardEvent,
        CGEventCreateMouseEvent,
        CGEventCreateScrollWheelEvent,
        CGEventPost,
        CGEventSetFlags,
        CGWarpMouseCursorPosition,
        CGAssociateMouseAndMouseCursorPosition,
        kCGEventFlagMaskSecondaryFn,
        kCGEventMouseMoved,
        kCGHIDEventTap,
        kCGMouseButtonLeft,
        kCGScrollEventUnitPixel,
    )
    _QUARTZ_OK = True
    # Re-associate OS cursor with the mouse so warp placements aren't
    # de-coupled from input. Safe/idempotent.
    try:
        CGAssociateMouseAndMouseCursorPosition(True)
    except Exception:
        pass
except Exception as _e:  # pragma: no cover
    print(f"[viewer] Quartz import failed: {_e}", flush=True)
    _QUARTZ_OK = False

from avcamera import Camera

# ---- local speech-to-text (lazy load) -----------------------------------
# Plan B for voice commands when Web Speech API doesn't work (Arc,
# offline, etc.). faster-whisper "tiny" runs comfortably on Apple Silicon
# CPU with int8 quant. ~75MB, downloaded to ~/.cache/huggingface on first
# use. Loaded only when /transcribe is first hit, so startup stays fast.
_whisper_model = None
_whisper_lock = threading.Lock()

def _get_whisper():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    with _whisper_lock:
        if _whisper_model is not None:
            return _whisper_model
        print("[viewer] loading whisper (tiny.en, int8)…", flush=True)
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(
            "tiny.en", device="cpu", compute_type="int8",
        )
        print("[viewer] whisper loaded", flush=True)
        return _whisper_model


# ---- voice daemon (direct-mic, bypasses browser) -------------------------
# Background thread that opens the default input device directly (via
# sounddevice, which uses CoreAudio on macOS), runs a simple energy-gated
# VAD to detect speech segments, ships each segment through the Whisper
# model, and dispatches matched commands. Permissions are inherited from
# whatever process launched browser_viewer.py — in our case the Terminal.
_voice_enabled: bool = False
_voice_thread: Optional[threading.Thread] = None
_voice_stop = threading.Event()
_voice_state: str = "off"          # off | listening | transcribing | err
_voice_last_text: str = ""
_voice_last_result: str = ""
_voice_err: str = ""


def _voice_loop() -> None:
    """Continuous push-nothing voice loop: listen → segment → transcribe →
    dispatch. Kept conservative so it can run alongside the camera + models
    without overwhelming the CPU."""
    global _voice_state, _voice_last_text, _voice_last_result, _voice_err
    try:
        import sounddevice as sd
    except Exception as e:
        _voice_err = f"sounddevice import failed: {e}"
        _voice_state = "err"
        print(f"[voice] {_voice_err}", flush=True)
        return

    SR = 16000
    BLOCK_S = 0.1
    BLOCK_N = int(SR * BLOCK_S)
    SPEECH_RMS = 0.012         # tuneable: higher = require louder speech
    SILENCE_HANG_S = 0.8       # end-of-utterance after this much quiet
    PRE_ROLL_S = 0.3           # keep this many seconds of pre-speech audio
    MIN_SEGMENT_S = 0.4
    MAX_SEGMENT_S = 8.0

    try:
        stream = sd.InputStream(
            samplerate=SR, channels=1, dtype="float32",
            blocksize=BLOCK_N,
        )
        stream.start()
    except Exception as e:
        _voice_err = f"mic open failed: {e}"
        _voice_state = "err"
        print(f"[voice] {_voice_err}", flush=True)
        return

    # Warm the model early — first transcribe call is otherwise ~5s.
    try:
        _get_whisper()
    except Exception as e:
        _voice_err = f"whisper load failed: {e}"
        _voice_state = "err"
        print(f"[voice] {_voice_err}", flush=True)
        stream.stop(); stream.close()
        return

    print("[voice] listening…", flush=True)
    _voice_state = "listening"
    _voice_err = ""

    pre_roll: Deque = deque(maxlen=int(PRE_ROLL_S / BLOCK_S))
    buf: List = []
    in_speech = False
    silence_s = 0.0
    segment_s = 0.0

    try:
        while not _voice_stop.is_set():
            try:
                data, _overflow = stream.read(BLOCK_N)
            except Exception as e:
                _voice_err = f"mic read err: {e}"
                print(f"[voice] {_voice_err}", flush=True)
                time.sleep(0.1)
                continue

            # data shape: (BLOCK_N, 1) float32 in [-1, 1]
            rms = float(np.sqrt(np.mean(data ** 2)))
            is_speech = rms > SPEECH_RMS

            if is_speech:
                if not in_speech:
                    in_speech = True
                    buf = list(pre_roll)   # include pre-roll
                    segment_s = len(buf) * BLOCK_S
                buf.append(data.copy())
                segment_s += BLOCK_S
                silence_s = 0.0
            else:
                pre_roll.append(data.copy())
                if in_speech:
                    buf.append(data.copy())
                    segment_s += BLOCK_S
                    silence_s += BLOCK_S

            hit_silence = in_speech and silence_s >= SILENCE_HANG_S
            hit_max     = in_speech and segment_s >= MAX_SEGMENT_S
            if hit_silence or hit_max:
                if segment_s >= MIN_SEGMENT_S and buf:
                    audio = np.concatenate(
                        [b[:, 0] for b in buf]
                    ).astype(np.float32)
                    _voice_state = "transcribing"
                    _voice_handle_segment(audio, SR)
                    _voice_state = "listening"
                in_speech = False
                buf = []
                silence_s = 0.0
                segment_s = 0.0
    finally:
        try:
            stream.stop(); stream.close()
        except Exception:
            pass
        print("[voice] stopped", flush=True)
        _voice_state = "off"


# Whisper bias prompt: feeding the model a sentence that uses our command
# vocabulary makes it vastly more likely to pick these words over common-
# English neighbors. This is how we get a "dictionary" without retraining.
def _scan_installed_apps() -> list:
    """Enumerate /Applications, ~/Applications, and /System/Applications
    for *.app bundles. Returns clean app names (no .app suffix), deduped,
    alphabetized, and trimmed to skip obvious noise. Called once at import."""
    import os as _os
    seen = set()
    out = []
    for root in ("/Applications", _os.path.expanduser("~/Applications"),
                 "/System/Applications", "/System/Applications/Utilities"):
        try:
            entries = _os.listdir(root)
        except OSError:
            continue
        for name in entries:
            if not name.endswith(".app"):
                continue
            bare = name[:-4].strip()
            if not bare or bare.startswith("."):
                continue
            key = bare.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(bare)
    out.sort(key=str.lower)
    return out


INSTALLED_APPS: list = _scan_installed_apps()
print(f"[viewer] found {len(INSTALLED_APPS)} installed apps "
      f"for voice bias", flush=True)


def _build_voice_bias_prompt(apps: list) -> str:
    """Compose the Whisper initial_prompt. Keeps total length reasonable
    (tiny.en model ignores tokens past ~224 anyway) by sampling a core
    set of likely-said apps first, then filling with the rest."""
    # Whisper pays most attention to the START of the prompt; front-load
    # common everyday-use apps, then append the rest.
    priority = [
        "Telegram", "Notion", "Arc", "Slack", "Spotify", "Figma",
        "Safari", "Chrome", "Cursor", "Terminal", "Finder",
        "Messages", "Mail", "Calendar", "ChatGPT", "Claude",
        "Raycast", "Superhuman", "Signal", "WhatsApp", "Discord",
        "Visual Studio Code", "Zoom", "Loom", "Wispr Flow",
    ]
    installed_lc = {a.lower(): a for a in apps}
    ordered = []
    for p in priority:
        if p.lower() in installed_lc:
            ordered.append(installed_lc[p.lower()])
    for a in apps:
        if a not in ordered:
            ordered.append(a)
    # First chunk: "open X" phrasings of the top entries.
    head = ", ".join(f"open {a}" for a in ordered[:18])
    # Then a flat roster of names so saying JUST the app name also biases.
    roster = ", ".join(ordered)
    return (
        f"Voice commands: {head}. "
        f"Apps: {roster}. "
        "Scroll up, scroll down, page up, page down. "
        "Click, tap, volume up, volume down, mute. "
        "Next desktop, previous desktop."
    )


_VOICE_BIAS_PROMPT = _build_voice_bias_prompt(INSTALLED_APPS)

# Rolling buffer of recent utterances so the UI can show what Whisper
# actually heard, even for phrases that didn't match a command.
_voice_transcript: Deque = deque(maxlen=12)


def _voice_handle_segment(audio_np, sr: int) -> None:
    """Transcribe one captured utterance and, if it matches a known
    command, dispatch it."""
    global _voice_last_text, _voice_last_result
    try:
        model = _get_whisper()
        t0 = time.time()
        segments, _info = model.transcribe(
            audio_np, beam_size=1, vad_filter=False,
            language="en",
            initial_prompt=_VOICE_BIAS_PROMPT,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        dt = time.time() - t0
        print(f"[voice] {dt*1000:.0f}ms → {text!r}", flush=True)
        if not text:
            return
        _voice_last_text = text
        cmd = _match_command(text)
        if cmd:
            result = _dispatch_command(cmd)
            _voice_last_result = (
                f"{cmd.get('action')} {cmd.get('target','')}".strip()
            )
            print(f"[voice] fired: {result}", flush=True)
        else:
            _voice_last_result = "(no match)"
        _voice_transcript.append({
            "text": text,
            "result": _voice_last_result,
            "ts": int(time.time() * 1000),
        })
    except Exception as e:
        print(f"[voice] transcribe err: {e}", flush=True)


def _voice_start() -> None:
    global _voice_enabled, _voice_thread, _voice_state
    if _voice_enabled:
        return
    _voice_stop.clear()
    _voice_enabled = True
    _voice_state = "starting"
    _voice_thread = threading.Thread(target=_voice_loop, daemon=True)
    _voice_thread.start()


def _voice_stop_fn() -> None:
    global _voice_enabled, _voice_thread
    if not _voice_enabled:
        return
    _voice_enabled = False
    _voice_stop.set()
    _voice_thread = None


PORT = 8765
BOUNDARY = "handsfree-frame"
HERE = Path(__file__).parent
FACE_MODEL_PATH = HERE / "face_landmarker.task"
HAND_MODEL_PATH = HERE / "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# Hand skeleton: pairs of landmark indices to connect.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)
PALM_IDXS = (0, 5, 9, 13, 17)

FACE_DOT_COLOR_BGR = (183, 231, 110)   # #6ee7b7
HAND_DOT_COLOR_BGR = (236, 180, 112)
HAND_LINE_COLOR_BGR = (200, 140, 80)

MOTION_EMA_ALPHA = 0.25
SMILE_EMA_ALPHA = 0.30

# Head bob: look at nose Y over a short window; fire when the descent peaks.
BOB_WINDOW_FRAMES = 6         # rolling window size (~0.20s @ 30fps)
BOB_DESCENT_THRESHOLD = 0.006 # min drop across 3 frames to count as a bob
BOB_COOLDOWN_S = 0.22

# Blink: both eyelids closed together briefly.
BLINK_ON_THRESHOLD = 0.55
BLINK_OFF_THRESHOLD = 0.25
BLINK_COOLDOWN_S = 0.28

# Hand swipe across the frame — loose enough to fire on a real-feeling swipe.
SWIPE_WINDOW_S = 1.0
SWIPE_MIN_TRAVEL = 0.40
SWIPE_EDGE_START = 0.32
SWIPE_EDGE_END = 0.68
SWIPE_COOLDOWN_S = 1.2

# Prayer hands: hold two palms close → hold Wispr hotkey until released.
PRAYER_CLOSE_THRESHOLD = 0.14          # a bit easier to enter
PRAYER_OPEN_THRESHOLD = 0.22           # clearly apart before releasing
PRAYER_ENTER_HOLD_S = 0.20
PRAYER_MAX_HOLD_S = 30.0               # safety: auto-release after 30s
PRAYER_LOST_HANDS_GRACE_S = 1.2        # tolerate long tracker flicker

# Hands-above-head toggles cursor tracking.
HANDS_UP_Y_MAX = 0.22          # wrist Y (top=0) must be above this
HANDS_UP_HOLD_S = 0.35
HANDS_UP_COOLDOWN_S = 1.2

# Clap detection — thresholds are tunable at runtime via "clap sensitivity"
# preset in Control Center. Keys: close, far, gap_min, gap_max, cooldown, grace.
CLAP_PRESETS = {
    "tight":   (0.09, 0.18, 0.08, 0.55, 1.8, 0.10),
    "normal":  (0.14, 0.22, 0.05, 1.00, 1.5, 0.25),
    "loose":   (0.18, 0.28, 0.04, 1.40, 1.2, 0.40),
    "wide":    (0.22, 0.34, 0.03, 1.80, 1.0, 0.60),
    "ironman": (0.28, 0.42, 0.03, 2.20, 0.8, 0.90),
}
_clap_preset = "off"
(CLAP_CLOSE_THRESHOLD, CLAP_FAR_THRESHOLD, CLAP_GAP_MIN_S,
 CLAP_GAP_MAX_S, CLAP_BOOT_COOLDOWN_S,
 CLAP_HAND_LOST_GRACE_S) = CLAP_PRESETS["normal"]

# Head-pose cursor control. Jaw-open toggles tracking so it never hijacks
# the mouse until you explicitly open your mouth.
CURSOR_SENSITIVITY_X = 55.0   # pixels per degree of yaw
CURSOR_SENSITIVITY_Y = 55.0   # pixels per degree of pitch
CURSOR_SMOOTHING = 0.28
CURSOR_DEAD_ZONE_DEG = 1.8
CURSOR_CALIB_S = 1.2

# Wispr hotkey. Fn (key code 63) doesn't reliably press/release via CGEvent,
# so the default is F19 (key code 80) — an unused function key that any app
# can be bound to. Bind Wispr Flow to F19 in its settings.
WISPR_KEYCODE = int(os.environ.get("HANDSFREE_WISPR_KEYCODE", "80"))
WISPR_USE_FN_FLAG = os.environ.get("HANDSFREE_WISPR_FN_FLAG", "0") == "1"

# Theremin gating: hand must be above this normalized Y (0=top) to sound.
HAND_PLAY_THRESHOLD_Y = 0.55


BOXING_HTML = r"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>handsfree — 🥊 muay thai</title>
<style>
  :root {
    --warm: #f5c24a;
    --hot:  #ff4d4d;
    --cool: #6ee7b7;
    --bg:   #0a0a0f;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0; height: 100%;
    background: #000; color: #fff;
    font-family: 'Bebas Neue', Impact, system-ui, sans-serif;
    overflow: hidden; user-select: none; -webkit-user-select: none;
  }
  #stage {
    position: fixed; inset: 0;
    background: radial-gradient(ellipse at center, #1a0a0a 0%, #000 70%);
    overflow: hidden;
  }
  /* Camera feed full-bleed, slightly desaturated/dimmed so emoji+FX pop */
  #cam {
    position: absolute; inset: 0; width: 100%; height: 100%;
    object-fit: cover;
    filter: brightness(0.55) saturate(0.6) contrast(1.15);
    z-index: 1;
  }
  #vignette {
    position: absolute; inset: 0; pointer-events: none; z-index: 2;
    background:
      radial-gradient(ellipse at center, transparent 40%, rgba(0,0,0,0.7) 100%);
  }
  /* The opponent — McGregor's head emoji 🤡 */
  #target {
    position: absolute; left: 50%; top: 50%;
    transform: translate(-50%,-50%);
    font-size: 280px; line-height: 1;
    z-index: 5; cursor: pointer;
    filter: drop-shadow(0 0 24px rgba(255,77,77,0.5));
    transition: transform 120ms cubic-bezier(.2,1.4,.4,1);
  }
  #target.hit {
    animation: shake 280ms cubic-bezier(.2,1.4,.4,1);
  }
  @keyframes shake {
    0%   { transform: translate(-50%,-50%) scale(1.0) rotate(0); }
    20%  { transform: translate(-58%,-46%) scale(0.85) rotate(-12deg); }
    40%  { transform: translate(-44%,-54%) scale(0.92) rotate(10deg); }
    60%  { transform: translate(-54%,-48%) scale(1.08) rotate(-6deg); }
    100% { transform: translate(-50%,-50%) scale(1.0) rotate(0); }
  }
  /* HUD */
  .hud {
    position: absolute; z-index: 10;
    text-shadow: 0 2px 0 #000, 0 0 12px rgba(255,77,77,0.4);
    letter-spacing: 0.04em;
  }
  #counter { top: 24px; left: 28px; font-size: 88px; line-height: 1; color: var(--warm); }
  #counter small { display:block; font-size: 14px; letter-spacing: 0.32em;
    color: #999; font-family: ui-monospace, Menlo, monospace; }
  #stats { top: 28px; right: 28px; text-align: right; font-size: 16px;
    font-family: ui-monospace, Menlo, monospace; color: #ddd;
    line-height: 1.6; }
  #stats span { color: var(--warm); font-weight: 700; }
  #title { bottom: 24px; left: 50%; transform: translateX(-50%);
    font-size: 28px; letter-spacing: 0.42em; color: #777; }
  /* Toggles */
  .toolbar {
    position: absolute; bottom: 24px; right: 24px; z-index: 12;
    display: flex; gap: 8px;
  }
  .btn {
    background: rgba(255,255,255,0.08); border: 1px solid #333;
    color: #ddd; padding: 8px 14px; border-radius: 999px;
    font-family: ui-monospace, Menlo, monospace; font-size: 12px;
    letter-spacing: 0.18em; text-transform: uppercase; cursor: pointer;
  }
  .btn.on { background: var(--hot); color: #150404; border-color: var(--hot); }
  .btn:hover { border-color: var(--warm); }
  /* Comic-book POW! pop text */
  .pow {
    position: absolute; pointer-events: none; z-index: 8;
    font-size: 92px; font-weight: 900; letter-spacing: 0.04em;
    color: #fff;
    -webkit-text-stroke: 4px #000;
    text-shadow: 6px 6px 0 #ff4d4d, 12px 12px 0 #000;
    transform-origin: center;
    animation: pow 720ms cubic-bezier(.2,1.6,.4,1) forwards;
  }
  .pow.uppercut { color: var(--cool); text-shadow: 6px 6px 0 #1d6c4a, 12px 12px 0 #000; }
  .pow.hook     { color: var(--warm); text-shadow: 6px 6px 0 #8a6a18, 12px 12px 0 #000; }
  .pow.cross, .pow.jab { color: #fff; }
  @keyframes pow {
    0%   { opacity: 0; transform: translate(-50%,-50%) scale(0.2) rotate(-12deg); }
    20%  { opacity: 1; transform: translate(-50%,-50%) scale(1.25) rotate(-6deg); }
    100% { opacity: 0; transform: translate(-50%,-50%) scale(1.0) rotate(8deg)
                                  translateY(-40px); }
  }
  body.shake #stage { animation: stage-shake 220ms ease-out; }
  @keyframes stage-shake {
    0%, 100% { transform: translate(0,0); }
    20% { transform: translate(-6px, 4px); }
    40% { transform: translate(8px, -6px); }
    60% { transform: translate(-4px, -2px); }
    80% { transform: translate(4px, 4px); }
  }
  body.flash::after {
    content: ""; position: fixed; inset: 0; z-index: 100; pointer-events: none;
    background: rgba(255,77,77,0.25);
    animation: flash 180ms ease-out forwards;
  }
  @keyframes flash { 0% { opacity: 1; } 100% { opacity: 0; } }
  /* Health bar over target */
  #hp-wrap {
    position: absolute; left: 50%; top: calc(50% - 200px);
    transform: translateX(-50%); width: 320px; height: 14px; z-index: 6;
    background: #1a0606; border: 2px solid #000;
    border-radius: 8px; overflow: hidden;
  }
  #hp { width: 100%; height: 100%;
    background: linear-gradient(90deg, #ff4d4d, var(--warm));
    transition: width 220ms ease-out;
  }
  /* round timer */
  #round { position: absolute; top: 28px; left: 50%; transform: translateX(-50%);
    z-index: 10; font-size: 48px; color: #fff; letter-spacing: 0.16em;
    text-shadow: 0 2px 0 #000; }
  #round small { display:block; font-size:11px; letter-spacing:0.3em; color:#888;
    font-family: ui-monospace, Menlo, monospace; }

  /* === McGregor fights back === */
  #target {
    /* Smooth roam to new positions */
    transition: left 600ms cubic-bezier(.4,1.3,.5,1),
                top  600ms cubic-bezier(.4,1.3,.5,1),
                transform 120ms cubic-bezier(.2,1.4,.4,1);
  }
  #target.bob { animation: bob 1.6s ease-in-out infinite; }
  @keyframes bob {
    0%,100% { transform: translate(-50%,-50%) scale(1) rotate(-2deg); }
    50%     { transform: translate(-50%,-54%) scale(1.02) rotate(2deg); }
  }
  /* Speech bubble taunts */
  #taunt {
    position: absolute; z-index: 9; pointer-events:none;
    background:#fff; color:#111; padding:10px 16px;
    border-radius: 18px; font-family: 'Bebas Neue', Impact, sans-serif;
    font-size: 28px; letter-spacing:0.04em;
    box-shadow: 0 4px 0 #000, 0 0 0 3px #000;
    opacity: 0; transition: opacity 200ms ease-out;
    max-width: 320px; text-align: center;
    transform: translate(-50%, -50%);
  }
  #taunt.show { opacity: 1; }
  #taunt::after {
    content:""; position:absolute; left:50%; bottom:-14px;
    transform: translateX(-50%);
    border: 8px solid transparent; border-top: 14px solid #000;
  }
  /* User HP bar (bottom) */
  #user-hp-wrap {
    position: absolute; left: 50%; bottom: 70px;
    transform: translateX(-50%);
    width: 320px; height: 14px; z-index: 11;
    background: #060616; border: 2px solid #000; border-radius:8px;
    overflow: hidden;
  }
  #user-hp { width:100%; height:100%;
    background: linear-gradient(90deg, #70b4ec, #6ee7b7);
    transition: width 260ms ease-out;
  }
  #user-hp-label {
    position:absolute; left:50%; bottom: 90px; transform:translateX(-50%);
    z-index:11; font-size:11px; letter-spacing:0.3em; color:#88a;
    font-family: ui-monospace, Menlo, monospace;
  }
  /* Incoming-punch warning arrow */
  .incoming {
    position: absolute; z-index: 7; pointer-events:none;
    font-size: 80px; line-height:1;
    filter: drop-shadow(0 0 18px rgba(255,77,77,0.8));
    transform: translate(-50%,-50%) scale(0.4);
    opacity: 0;
  }
  .incoming.warn { animation: warn 700ms ease-out forwards; }
  .incoming.swoop { animation: swoop 480ms cubic-bezier(.6,.05,.9,.6) forwards; }
  @keyframes warn {
    0%   { opacity: 0; transform: translate(-50%,-50%) scale(0.4); }
    50%  { opacity: 1; transform: translate(-50%,-50%) scale(1.2); }
    100% { opacity: 0.9; transform: translate(-50%,-50%) scale(1.0); }
  }
  @keyframes swoop {
    0%   { opacity: 1; transform: translate(-50%,-50%) scale(1.0); }
    100% { opacity: 0; transform: translate(-50%, calc(-50% + 60vh))
                                   scale(6) rotate(8deg); }
  }
  /* Block / dodge indicator */
  .blocked {
    position: absolute; z-index: 9; pointer-events:none;
    font-size: 96px; font-weight: 900;
    -webkit-text-stroke: 4px #000;
    color: var(--cool);
    text-shadow: 6px 6px 0 #1d6c4a, 12px 12px 0 #000;
    transform: translate(-50%,-50%) scale(0.4);
    opacity: 0;
    animation: blocked 700ms cubic-bezier(.2,1.6,.4,1) forwards;
  }
  @keyframes blocked {
    0%   { opacity: 0; transform: translate(-50%,-50%) scale(0.3) rotate(-10deg); }
    25%  { opacity: 1; transform: translate(-50%,-50%) scale(1.3) rotate(5deg); }
    100% { opacity: 0; transform: translate(-50%,-50%) scale(1.1) rotate(0)
                                   translateY(-30px); }
  }
  /* KO cinematic overlay */
  body.ko::before {
    content: "K.O.";
    position: fixed; inset: 0; z-index: 200;
    background: rgba(255,77,77,0.35);
    display: flex; align-items: center; justify-content: center;
    color: #fff; font-size: 240px; font-weight: 900; letter-spacing: 0.08em;
    -webkit-text-stroke: 8px #000;
    text-shadow: 14px 14px 0 #000;
    animation: ko 1.6s cubic-bezier(.2,1.4,.4,1) forwards;
    pointer-events: none;
  }
  @keyframes ko {
    0%   { opacity: 0; transform: scale(0.4) rotate(-8deg); }
    25%  { opacity: 1; transform: scale(1.2) rotate(2deg); }
    80%  { opacity: 1; transform: scale(1.0) rotate(0); }
    100% { opacity: 0; transform: scale(1.05); }
  }
  /* Combo callouts */
  #combo {
    position: absolute; left: 50%; top: 130px;
    transform: translateX(-50%); z-index: 10;
    font-size: 36px; color: var(--warm); letter-spacing: 0.18em;
    text-shadow: 0 2px 0 #000, 0 0 18px rgba(245,194,74,0.6);
    opacity: 0; transition: opacity 260ms ease-out;
    pointer-events:none; font-weight: 900;
  }
  #combo.show { opacity: 1; }
</style>
</head>
<body>
  <div id="stage">
    <img id="cam" src="/stream" alt="cam">
    <div id="vignette"></div>
    <div id="hp-wrap"><div id="hp"></div></div>
    <div id="target">🤡</div>
    <div id="counter">0<small>strikes</small></div>
    <div id="stats">
      <div>jab    <span id="s-jab">0</span></div>
      <div>cross  <span id="s-cross">0</span></div>
      <div>hook   <span id="s-hook">0</span></div>
      <div>upper  <span id="s-uppercut">0</span></div>
      <div>peak   <span id="s-peak">0.0</span></div>
    </div>
    <div id="round">3:00<small>round 1</small></div>
    <div id="combo"></div>
    <div id="taunt"></div>
    <div id="user-hp-label">YOUR HP</div>
    <div id="user-hp-wrap"><div id="user-hp"></div></div>
    <div id="title">🥊 MUAY THAI · YOLO MODE</div>
    <div class="toolbar">
      <button class="btn on" id="tog-detect">detect: on</button>
      <button class="btn" id="tog-sound">sound</button>
      <button class="btn" id="tog-bell">round</button>
      <button class="btn" id="reset">reset</button>
    </div>
  </div>
<script>
  // -- toggle detection on the server (gate the punch detector) ----------
  fetch('/command', { method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({action:'boxing', on:true}) }).catch(()=>{});

  let detect = true;
  document.getElementById('tog-detect').addEventListener('click', e => {
    detect = !detect;
    e.target.classList.toggle('on', detect);
    e.target.textContent = 'detect: ' + (detect ? 'on' : 'off');
    fetch('/command', { method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({action:'boxing', on:detect}) }).catch(()=>{});
  });

  // Disable on page leave so the gesture loop doesn't keep classifying.
  window.addEventListener('beforeunload', () => {
    try {
      navigator.sendBeacon('/command',
        new Blob([JSON.stringify({action:'boxing', on:false})],
                 {type:'application/json'}));
    } catch(e) {}
  });

  // -- counters ----------------------------------------------------------
  const counts = { jab:0, cross:0, hook:0, uppercut:0 };
  let total = 0, peak = 0;
  let hp = 100;
  const elCounter = document.getElementById('counter');
  const elHp      = document.getElementById('hp');
  const elTarget  = document.getElementById('target');
  function updateHud() {
    elCounter.firstChild.nodeValue = total;
    for (const k of Object.keys(counts)) {
      document.getElementById('s-'+k).textContent = counts[k];
    }
    document.getElementById('s-peak').textContent = peak.toFixed(2);
    elHp.style.width = Math.max(0, hp) + '%';
  }

  document.getElementById('reset').addEventListener('click', () => {
    for (const k of Object.keys(counts)) counts[k] = 0;
    total = 0; peak = 0; hp = 100; updateHud();
  });

  // -- sounds (Web Audio synth thump) ------------------------------------
  let soundOn = true;
  const tgSound = document.getElementById('tog-sound');
  tgSound.classList.add('on'); tgSound.textContent = 'sound: on';
  tgSound.addEventListener('click', () => {
    soundOn = !soundOn;
    tgSound.classList.toggle('on', soundOn);
    tgSound.textContent = 'sound: ' + (soundOn ? 'on' : 'off');
  });
  let actx = null;
  function ac() {
    if (!actx) { try { actx = new (window.AudioContext||window.webkitAudioContext)(); }
                 catch(e) { return null; } }
    if (actx.state === 'suspended') actx.resume().catch(()=>{});
    return actx;
  }
  function thump(intensity, type) {
    if (!soundOn) return;
    const ctx = ac(); if (!ctx) return;
    const t0 = ctx.currentTime;
    // Body: low sine punch
    const osc = ctx.createOscillator();
    const g = ctx.createGain();
    osc.type = 'sine';
    const baseFreq = type === 'uppercut' ? 90 : type === 'hook' ? 110 : 140;
    osc.frequency.setValueAtTime(baseFreq * (1 + intensity * 0.6), t0);
    osc.frequency.exponentialRampToValueAtTime(40, t0 + 0.15);
    g.gain.setValueAtTime(0.0, t0);
    g.gain.linearRampToValueAtTime(0.45 * (0.4 + intensity), t0 + 0.005);
    g.gain.exponentialRampToValueAtTime(0.001, t0 + 0.22);
    osc.connect(g).connect(ctx.destination);
    osc.start(t0); osc.stop(t0 + 0.25);
    // Click: short noise burst
    try {
      const buf = ctx.createBuffer(1, ctx.sampleRate * 0.04, ctx.sampleRate);
      const d = buf.getChannelData(0);
      for (let i=0;i<d.length;i++) d[i] = (Math.random()*2-1) * Math.pow(1-i/d.length, 2);
      const src = ctx.createBufferSource(); src.buffer = buf;
      const hp = ctx.createBiquadFilter(); hp.type='highpass'; hp.frequency.value=1200;
      const ng = ctx.createGain(); ng.gain.value = 0.18 * (0.4 + intensity);
      src.connect(hp).connect(ng).connect(ctx.destination);
      src.start(t0);
    } catch(e){}
  }
  function bell() {
    if (!soundOn) return;
    const ctx = ac(); if (!ctx) return;
    const t0 = ctx.currentTime;
    [0,0.18,0.36].forEach((dt, i) => {
      const o = ctx.createOscillator();
      const g = ctx.createGain();
      o.type='triangle';
      o.frequency.setValueAtTime(880, t0+dt);
      o.frequency.exponentialRampToValueAtTime(440, t0+dt+0.6);
      g.gain.setValueAtTime(0.0, t0+dt);
      g.gain.linearRampToValueAtTime(0.22, t0+dt+0.01);
      g.gain.exponentialRampToValueAtTime(0.001, t0+dt+0.7);
      o.connect(g).connect(ctx.destination);
      o.start(t0+dt); o.stop(t0+dt+0.75);
    });
  }
  document.getElementById('tog-bell').addEventListener('click', bell);

  // -- punch FX ---------------------------------------------------------
  const STAGE = document.getElementById('stage');
  const WORDS = {
    jab:      ['POW!', 'BAM!', 'WHAP!', 'JAB!'],
    cross:    ['BOOM!', 'CRACK!', 'WHAM!', 'CROSS!'],
    hook:     ['THWACK!', 'KAPOW!', 'SOCK!', 'HOOK!'],
    uppercut: ['ZONK!', 'KABLAM!', 'BOFF!', 'UPPERCUT!'],
    straight: ['POW!', 'BAM!'],
  };
  function spawnPow(type, x, y) {
    const el = document.createElement('div');
    el.className = 'pow ' + type;
    const words = WORDS[type] || WORDS.jab;
    el.textContent = words[Math.floor(Math.random()*words.length)];
    // Camera is mirrored visually — flip x so the FX appears where you
    // actually punched (image-left x close to 0 → screen-right side).
    const px = (1 - x) * window.innerWidth;
    const py = y * window.innerHeight;
    el.style.left = px + 'px';
    el.style.top  = py + 'px';
    el.style.transform = 'translate(-50%,-50%) scale(1) rotate(0)';
    STAGE.appendChild(el);
    setTimeout(() => el.remove(), 740);
  }
  function hitTarget(type, intensity) {
    elTarget.classList.remove('hit');
    void elTarget.offsetWidth; // restart anim
    elTarget.classList.add('hit');
    // Damage scaled by intensity. Hooks + uppercuts hit harder.
    const mult = (type === 'hook' || type === 'uppercut') ? 1.6 : 1.0;
    hp = Math.max(0, hp - 4 * mult * (0.4 + intensity));
    if (hp <= 0) {
      elTarget.textContent = '😵';
      setTimeout(() => { hp = 100; elTarget.textContent = '🤡'; updateHud(); }, 1400);
    } else if (hp < 30) {
      elTarget.textContent = '🥴';
    } else if (hp < 60) {
      elTarget.textContent = '😣';
    } else {
      elTarget.textContent = '🤡';
    }
  }
  function flashScreen() {
    document.body.classList.remove('flash');
    void document.body.offsetWidth;
    document.body.classList.add('flash');
    document.body.classList.remove('shake');
    void document.body.offsetWidth;
    document.body.classList.add('shake');
    setTimeout(() => {
      document.body.classList.remove('flash');
      document.body.classList.remove('shake');
    }, 280);
  }

  // -- SSE: punch events from server -------------------------------------
  function connectSSE() {
    const es = new EventSource('/events');
    es.onmessage = (m) => {
      let msg; try { msg = JSON.parse(m.data); } catch (e) { return; }
      if (Array.isArray(msg.punches)) {
        for (const p of msg.punches) {
          const type = p.type || 'jab';
          const c = type === 'straight' ? 'jab' : type;
          if (counts[c] !== undefined) counts[c]++;
          else counts.jab++;
          total++;
          if (p.speed > peak) peak = p.speed;
          spawnPow(type, p.x, p.y);
          hitTarget(type, p.intensity || 0.5);
          flashScreen();
          thump(p.intensity || 0.5, type);
          updateHud();
        }
      }
    };
    es.onerror = () => { es.close(); setTimeout(connectSSE, 1500); };
  }
  connectSSE();
  updateHud();

  // -- round timer (3:00 / 1:00 rest, optional bell on transitions) -----
  let roundT = 180; // 3 minutes
  let onBreak = false;
  let roundN = 1;
  const elRound = document.getElementById('round');
  function fmt(t) {
    const m = Math.floor(t/60); const s = t%60;
    return m + ':' + (s<10?'0':'') + s;
  }
  setInterval(() => {
    roundT--;
    if (roundT <= 0) {
      bell();
      onBreak = !onBreak;
      if (onBreak) { roundT = 60; elRound.firstChild.nodeValue = '1:00'; }
      else { roundN++; roundT = 180; }
      elRound.querySelector('small').textContent =
        onBreak ? 'rest' : 'round ' + roundN;
    } else {
      elRound.firstChild.nodeValue = fmt(roundT);
    }
  }, 1000);

  // ============================================================
  // === McGREGOR FIGHTS BACK ===================================
  // ============================================================

  // -- Moving target: roam to a random spot in the safe zone ---
  elTarget.classList.add('bob');
  function roamTarget() {
    // Stay inside 18%–82% horizontally, 22%–62% vertically
    const x = 18 + Math.random() * 64;
    const y = 22 + Math.random() * 40;
    elTarget.style.left = x + '%';
    elTarget.style.top  = y + '%';
    // Move HP bar with the target
    document.getElementById('hp-wrap').style.left = x + '%';
    document.getElementById('hp-wrap').style.top  = (y - 18) + '%';
  }
  function scheduleRoam() {
    const delay = 2200 + Math.random() * 2800;
    setTimeout(() => { roamTarget(); scheduleRoam(); }, delay);
  }
  // Anchor HP bar to follow target via fixed offsets, then start roaming
  const hpWrap = document.getElementById('hp-wrap');
  hpWrap.style.transition = 'left 600ms cubic-bezier(.4,1.3,.5,1), top 600ms cubic-bezier(.4,1.3,.5,1)';
  hpWrap.style.transform = 'translate(-50%, 0)';
  setTimeout(scheduleRoam, 1500);

  // -- Speech-bubble taunts -----------------------------------
  const TAUNTS = [
    "is that all ya got?",
    "you're nothing!",
    "left hand, sleep!",
    "hit me harder, kid",
    "wake up!",
    "i bend the knee for nobody",
    "swing again, cupcake",
    "too slow!",
    "try the other hand",
    "you call that a hook?",
    "i'm right here",
    "where's the heat?",
    "i've taken naps harder",
    "easy money",
  ];
  const KO_LINES = [
    "lights out, kid",
    "and STILL!",
    "told ya",
    "any time, any place",
  ];
  const elTaunt = document.getElementById('taunt');
  let tauntT = null;
  function showTaunt(text) {
    elTaunt.textContent = text;
    // place above target's current position
    const r = elTarget.getBoundingClientRect();
    elTaunt.style.left = (r.left + r.width/2) + 'px';
    elTaunt.style.top  = (r.top - 30) + 'px';
    elTaunt.classList.add('show');
    if (tauntT) clearTimeout(tauntT);
    tauntT = setTimeout(() => elTaunt.classList.remove('show'), 1800);
  }
  function scheduleTaunt() {
    const delay = 4500 + Math.random() * 4500;
    setTimeout(() => {
      if (!document.body.classList.contains('ko')) {
        showTaunt(TAUNTS[Math.floor(Math.random()*TAUNTS.length)]);
      }
      scheduleTaunt();
    }, delay);
  }
  setTimeout(scheduleTaunt, 3500);

  // -- USER HP + KO cinematic ---------------------------------
  let userHp = 100;
  const elUserHp = document.getElementById('user-hp');
  function updateUserHp() {
    elUserHp.style.width = Math.max(0, userHp) + '%';
    if (userHp <= 0) youGotKod();
  }
  function youGotKod() {
    if (document.body.classList.contains('ko')) return;
    document.body.classList.add('ko');
    showTaunt(KO_LINES[Math.floor(Math.random()*KO_LINES.length)]);
    if (soundOn) {
      // big descending boom + bell
      const ctx = ac();
      if (ctx) {
        const t0 = ctx.currentTime;
        const o = ctx.createOscillator(); const g = ctx.createGain();
        o.type='sawtooth';
        o.frequency.setValueAtTime(220, t0);
        o.frequency.exponentialRampToValueAtTime(35, t0 + 1.2);
        g.gain.setValueAtTime(0.4, t0);
        g.gain.exponentialRampToValueAtTime(0.001, t0 + 1.4);
        o.connect(g).connect(ctx.destination);
        o.start(t0); o.stop(t0+1.4);
      }
      bell();
    }
    setTimeout(() => {
      document.body.classList.remove('ko');
      userHp = 100; updateUserHp();
    }, 2400);
  }

  // -- Incoming counter-punches: warn → swoop, dodge by punching --
  // Track recent punches so we can detect "block" within window.
  const recentPunches = []; // timestamps (ms)
  function noteBlockAttempt() { recentPunches.push(performance.now()); }
  function blockedRecently(windowMs) {
    const now = performance.now();
    while (recentPunches.length && now - recentPunches[0] > 1500) {
      recentPunches.shift();
    }
    return recentPunches.some(t => now - t < windowMs);
  }

  function spawnIncoming() {
    if (document.body.classList.contains('ko')) return;
    if (!detect) return;
    // origin = where target is; trajectory = down toward camera
    const r = elTarget.getBoundingClientRect();
    const ox = r.left + r.width/2;
    const oy = r.top  + r.height/2;
    const types = ['🥊','👊','🦶']; // punch / fist / kick
    const glyph = types[Math.floor(Math.random()*types.length)];

    // Warning glyph appears at target
    const warn = document.createElement('div');
    warn.className = 'incoming warn';
    warn.textContent = '⚠️';
    warn.style.left = ox + 'px';
    warn.style.top  = oy + 'px';
    STAGE.appendChild(warn);

    // Warning whoosh sound
    if (soundOn) {
      const ctx = ac();
      if (ctx) {
        const t0 = ctx.currentTime;
        const o = ctx.createOscillator(); const g = ctx.createGain();
        o.type='sawtooth';
        o.frequency.setValueAtTime(300, t0);
        o.frequency.exponentialRampToValueAtTime(800, t0+0.5);
        g.gain.setValueAtTime(0.0, t0);
        g.gain.linearRampToValueAtTime(0.10, t0+0.05);
        g.gain.exponentialRampToValueAtTime(0.001, t0+0.55);
        o.connect(g).connect(ctx.destination);
        o.start(t0); o.stop(t0+0.6);
      }
    }

    // 600ms warning window, then projectile launches
    setTimeout(() => {
      warn.remove();
      const proj = document.createElement('div');
      proj.className = 'incoming swoop';
      proj.textContent = glyph;
      proj.style.left = ox + 'px';
      proj.style.top  = oy + 'px';
      STAGE.appendChild(proj);

      // Impact resolves at end of swoop
      setTimeout(() => {
        proj.remove();
        // Did the user throw a punch in the dodge window? (≤900ms)
        if (blockedRecently(900)) {
          // BLOCKED!
          const b = document.createElement('div');
          b.className = 'blocked';
          b.textContent = 'BLOCKED!';
          b.style.left = (window.innerWidth/2) + 'px';
          b.style.top  = (window.innerHeight - 200) + 'px';
          STAGE.appendChild(b);
          setTimeout(() => b.remove(), 720);
          // tiny chime
          if (soundOn) {
            const ctx = ac();
            if (ctx) {
              const t0 = ctx.currentTime;
              const o = ctx.createOscillator(); const g = ctx.createGain();
              o.type='triangle';
              o.frequency.setValueAtTime(1200, t0);
              o.frequency.exponentialRampToValueAtTime(2000, t0+0.15);
              g.gain.setValueAtTime(0.18, t0);
              g.gain.exponentialRampToValueAtTime(0.001, t0+0.2);
              o.connect(g).connect(ctx.destination);
              o.start(t0); o.stop(t0+0.22);
            }
          }
        } else {
          // HIT — user takes damage
          const dmg = 8 + Math.random()*10;
          userHp = Math.max(0, userHp - dmg);
          updateUserHp();
          // red flash + heavy shake
          flashScreen();
          // low impact thump
          thump(0.9, 'hook');
          // also trigger a stronger camera shake
          document.body.classList.remove('shake');
          void document.body.offsetWidth;
          document.body.classList.add('shake');
          setTimeout(() => document.body.classList.remove('shake'), 280);
        }
      }, 480);
    }, 600);
  }
  function scheduleIncoming() {
    // Difficulty ramps with round number — interval shrinks 5–9s → 3–6s
    const base = Math.max(3000, 5000 - (roundN-1)*800);
    const delay = base + Math.random() * 4000;
    setTimeout(() => {
      spawnIncoming();
      scheduleIncoming();
    }, delay);
  }
  setTimeout(scheduleIncoming, 6000);
  updateUserHp();

  // -- Combo tracker ------------------------------------------
  const elCombo = document.getElementById('combo');
  let comboCount = 0;
  let comboTimer = null;
  function bumpCombo() {
    comboCount++;
    if (comboCount >= 3) {
      elCombo.textContent = comboCount + '× COMBO!';
      elCombo.classList.add('show');
    }
    if (comboTimer) clearTimeout(comboTimer);
    comboTimer = setTimeout(() => {
      elCombo.classList.remove('show');
      comboCount = 0;
    }, 1400);
  }

  // -- Hook into existing punch handler -----------------------
  // Wrap the SSE path so each user punch also notes a block-attempt
  // and bumps the combo counter. We do this by overriding hitTarget.
  const _origHitTarget = hitTarget;
  hitTarget = function(type, intensity) {
    noteBlockAttempt();
    bumpCombo();
    return _origHitTarget(type, intensity);
  };
</script>
</body></html>
"""


HTML = """<!doctype html>
<html><head>
<meta charset="utf-8">
<title>handsfree — play</title>
<style>
  :root { color-scheme: dark;
    --bg:#0c0c10; --panel:#14141a; --ink:#ececf1; --dim:#7a7a88;
    --accent:#6ee7b7; --warm:#fbbf24; --cool:#70b4ec; }
  * { box-sizing: border-box; }
  body { background: radial-gradient(1200px 800px at 50% -10%, #1b1f2a 0%, var(--bg) 60%); color: var(--ink);
    margin:0; min-height:100vh; font-family: ui-monospace, Menlo, monospace;
    display:flex; flex-direction:column; align-items:center; padding: 28px 24px 40px; gap:18px; }
  h1 { margin:0; font-size:12px; letter-spacing:0.16em; text-transform:uppercase; color: var(--dim); font-weight:500; }
  .stage { display:grid; grid-template-columns: auto 300px; gap:20px; align-items:start;
    max-width: 1200px; width: 100%; }
  img { width: 100%; max-width: 860px; border-radius: 14px; background:#000;
    box-shadow: 0 20px 60px rgba(0,0,0,0.55); }
  .side { display:flex; flex-direction:column; gap:18px; padding:20px;
    background: var(--panel); border-radius:14px; min-height:280px; }
  .status { font-size: 22px; font-weight:600; letter-spacing:0.04em; }
  .status.off  { color: var(--dim); }
  .status.on   { color: var(--accent); }
  .row { display:flex; flex-direction:column; gap:6px; }
  .row .label { font-size:10px; letter-spacing:0.14em; text-transform:uppercase; color: var(--dim); }
  .bar { height:10px; background:#0a0a0f; border-radius:6px; overflow:hidden; position:relative; }
  .bar .fill { height:100%; width:0%; border-radius:6px; transition: width 80ms linear; }
  .bar .fill.motion { background: linear-gradient(90deg, var(--cool), var(--accent)); }
  .bar .fill.smile  { background: linear-gradient(90deg, var(--warm), #f59e0b); }
  button { appearance:none; border:none; background: var(--accent); color:#05170f; font-weight:700;
    font-size:13px; letter-spacing:0.08em; text-transform:uppercase; padding:12px 16px;
    border-radius:10px; cursor:pointer; font-family: inherit; }
  button:disabled { background:#2a2a33; color: var(--dim); cursor: default; }
  .hint { color: var(--dim); font-size: 12px; max-width: 860px; text-align:center; line-height:1.6; }
  .clap-flash { position: fixed; inset: 0; pointer-events:none; background: radial-gradient(circle at center, rgba(110,231,183,0.18), transparent 60%);
    opacity:0; transition: opacity 200ms ease-out; }
  .clap-flash.on { opacity: 1; }
  body.clap-pulse { box-shadow: inset 0 0 120px rgba(110,231,183,0.35); transition: box-shadow 180ms ease-out; }
  body.atelier-on { box-shadow: inset 0 0 220px rgba(245,194,74,0.18), inset 0 0 80px rgba(122,167,255,0.12); }
  body.atelier-on::after {
    content:""; position:fixed; inset:0; pointer-events:none; z-index:1;
    background:
      radial-gradient(circle at 10% 90%, rgba(245,194,74,0.10), transparent 40%),
      radial-gradient(circle at 90% 10%, rgba(122,167,255,0.10), transparent 40%);
  }
  body.system-off { filter: grayscale(0.6) brightness(0.75); }
  body.system-off::before { content:"paused · double clap to resume";
    position:fixed; top:12px; left:50%; transform:translateX(-50%);
    background:#1a1a22; color:var(--dim); padding:6px 14px; border-radius:999px;
    font-size:11px; letter-spacing:0.12em; text-transform:uppercase;
    border:1px solid #2a2a38; z-index:9999; pointer-events:none; }
  #heard .final   { color: var(--accent); }
  #heard .interim { color: var(--dim); font-style: italic; }
  .vt-q { cursor:pointer; font-size:10px; padding:5px 10px;
    border-radius:999px; border:1px solid #2a3550; background:#15151c;
    color:var(--ink); }
  .vt-q:hover { background:#1b2530; border-color:#6ee7b7; color:var(--fg); }
  #vt-btn { cursor:pointer; font-size:10px; letter-spacing:0.16em;
    text-transform:uppercase; font-weight:700; padding:6px 14px;
    border-radius:999px; border:1px solid #2a2a38; background:#15151c;
    color:var(--dim); }
  #vt-btn.on { background:#6ee7b7; color:#05170f; border-color:#6ee7b7; }
  #jam-btn, #cc-btn { cursor:pointer; font-size:10px; letter-spacing:0.16em;
    text-transform:uppercase; font-weight:700; padding:6px 14px;
    border-radius:999px; border:1px solid #2a2a38; background:#15151c;
    color:var(--dim); }
  #jam-btn.on { background:#b48cff; color:#140a24; border-color:#b48cff; }
  #cc-btn.on  { background:#6ee7b7; color:#05170f; border-color:#6ee7b7; }

  /* ---- DJ Board (visible only in jam mode) ---- */
  #dj-board { order: 5; width:100%; max-width:1200px; display:none;
    margin-top:10px; padding:12px;
    background:linear-gradient(180deg, #150826 0%, #0a0a12 100%);
    border:1px solid #b48cff; border-radius:14px;
    box-shadow: 0 0 32px rgba(180,140,255,0.2); gap:10px; }
  body.jam #dj-board { display:flex; flex-wrap:wrap; }
  .dj-chan { flex:1 1 140px; min-width:0; background:#0a0a12;
    border:1px solid #2a2a38; border-radius:10px; padding:10px;
    display:flex; flex-direction:column; gap:8px;
    transition: border-color 160ms, box-shadow 160ms; }
  .dj-chan.on { border-color:#b48cff;
    box-shadow: 0 0 14px rgba(180,140,255,0.25); }
  .dj-chan.off { opacity: 0.55; }
  .dj-head { display:flex; justify-content:space-between; align-items:center;
    font-size:10px; letter-spacing:0.18em; text-transform:uppercase;
    color:var(--dim); font-weight:700; }
  .dj-chan.on .dj-head { color:#e5d8ff; }
  .dj-icon { font-size:14px; }
  .dj-toggle { cursor:pointer; background:#15151c; color:var(--dim);
    border:1px solid #2a2a38; border-radius:999px;
    font-size:10px; letter-spacing:0.16em; text-transform:uppercase;
    font-weight:700; padding:5px 0; text-align:center;
    transition: background 120ms, box-shadow 160ms; }
  .dj-toggle.on { background:#b48cff; color:#140a24; border-color:#b48cff;
    box-shadow: 0 0 12px rgba(180,140,255,0.55); }
  .dj-preset { background:#0f0f16; color:var(--ink);
    border:1px solid #262634; border-radius:6px;
    padding:5px 8px; font-size:11px; font-family:inherit; cursor:pointer; }
  .dj-preset:focus { outline:1px solid #b48cff; }
  .dj-meter { height:6px; background:#1a1a24; border-radius:4px;
    overflow:hidden; position:relative; }
  .dj-meter-fill { position:absolute; left:0; top:0; bottom:0; width:0%;
    background:linear-gradient(90deg, #6ee7b7, #fbbf24, #f87171);
    transition: width 70ms linear; }
  .dj-chan.off .dj-meter-fill { background:#2a2a38; }
  .dj-master { flex:0 0 120px; display:flex; flex-direction:column; gap:8px;
    padding:10px; background:#0a0a12; border:1px solid #b48cff;
    border-radius:10px; }
  .dj-vol { -webkit-appearance:none; appearance:none;
    width:100%; height:4px; background:#262634; border-radius:2px;
    outline:none; margin:4px 0; }
  .dj-vol::-webkit-slider-thumb { -webkit-appearance:none; appearance:none;
    width:14px; height:14px; border-radius:50%; background:#b48cff;
    cursor:pointer; box-shadow:0 0 10px rgba(180,140,255,0.75); }
  .dj-bpm { font-size:22px; font-weight:800; color:#b48cff;
    text-align:center; font-variant-numeric: tabular-nums;
    line-height:1; margin-top:2px; }
  .dj-bpm-label { font-size:9px; letter-spacing:0.2em; text-transform:uppercase;
    color:var(--dim); text-align:center; }
  /* Layout: camera at top, control center pinned to bottom.
     body is a flex column, so we reorder with flex `order` instead of
     moving DOM (keeps the rest of the JS untouched). */
  .stage    { order: 1; }
  #cc-panel { order: 10; width:100%; max-width:1200px;
    max-height:0; overflow:hidden; opacity:0;
    margin-top:0; padding:0 14px;
    background:#0a0a0f; border:1px solid transparent; border-radius:10px;
    transition: max-height 220ms ease, opacity 140ms ease,
                margin-top 220ms ease, padding 220ms ease,
                border-color 140ms ease;
    /* Skip rendering the panel's subtree when it's offscreen/closed so
       toggling doesn't force a massive layout pass. */
    content-visibility: auto; contain-intrinsic-size: 0 600px;
    will-change: max-height, opacity; }
  #cc-panel.open { max-height:80vh; overflow-y:auto; opacity:1;
    margin-top:10px; padding:14px; border-color:#262634; }
  .cc-row { display:flex; align-items:center; gap:10px;
    margin-bottom:10px; flex-wrap:wrap; }
  .cc-row:last-child { margin-bottom:0; }
  .cc-label { font-size:10px; letter-spacing:0.16em; text-transform:uppercase;
    color:var(--dim); width:72px; }
  .cc-opts { display:flex; gap:6px; flex-wrap:wrap; }
  .cc-opt { cursor:pointer; font-size:11px; padding:5px 12px;
    border-radius:999px; border:1px solid #2a2a38; background:#15151c;
    color:var(--ink); }
  .cc-opt.on { background:#6ee7b7; color:#05170f; border-color:#6ee7b7; }
  .cc-hint { font-size:10px; color:var(--dim); margin-top:4px; line-height:1.5; }
  body.jam #listening-pill, body.jam #cursor-pill, body.jam #voice-pill,
  body.jam #heard, body.jam .row:has(#notes), body.jam .hint,
  body.jam #status { display:none !important; }
  body.jam { background:radial-gradient(ellipse at top,#1a0f2a 0%,#07070b 70%); }
  #stars { position:fixed; inset:0; width:100%; height:100%;
    pointer-events:none; z-index:0; }
  body > *:not(#stars):not(#disco):not(.disco-beam) { position:relative; z-index:1; }

  /* Disco ball — visible only in jam mode. */
  #disco { position:fixed; top:40px; left:50%; transform:translateX(-50%);
    width:110px; height:110px; border-radius:50%;
    background:
      radial-gradient(circle at 35% 30%, rgba(255,255,255,0.95), rgba(180,200,255,0.35) 30%, transparent 55%),
      conic-gradient(from 0deg,
        #ffb4e6, #b4d4ff, #b4ffd4, #fff4b4, #ffb4e6);
    box-shadow:
      0 0 40px rgba(255,180,230,0.55),
      0 0 100px rgba(180,200,255,0.4),
      inset -18px -18px 40px rgba(0,0,0,0.55),
      inset  14px  14px 30px rgba(255,255,255,0.25);
    opacity:0; pointer-events:none; z-index:2;
    transition: opacity 420ms ease;
    animation: disco-spin 6s linear infinite;
    background-size: 10px 10px, 100% 100%;
    filter: saturate(1.3);
  }
  #disco::before { /* facet grid */
    content:""; position:absolute; inset:0; border-radius:50%;
    background:
      repeating-linear-gradient(45deg, rgba(0,0,0,0.25) 0 2px, transparent 2px 10px),
      repeating-linear-gradient(-45deg, rgba(255,255,255,0.18) 0 1px, transparent 1px 10px);
    mix-blend-mode: overlay;
  }
  #disco::after { /* hanging wire */
    content:""; position:absolute; top:-42px; left:50%;
    width:2px; height:42px; background:linear-gradient(#555, #222);
    transform:translateX(-50%);
  }
  body.jam #disco { opacity:1; }
  @keyframes disco-spin {
    from { filter: hue-rotate(0deg) saturate(1.3); }
    to   { filter: hue-rotate(360deg) saturate(1.3); }
  }

  /* Rotating disco beams — each a thin cone sweeping from ball to corners. */
  .disco-beam { position:fixed; top:96px; left:50%;
    width:2px; height:0; transform-origin:top center;
    background:linear-gradient(to bottom,
      hsla(var(--beam-hue, 300), 90%, 70%, 0.55), transparent);
    pointer-events:none; z-index:0; opacity:0;
    transition: opacity 420ms ease;
    box-shadow: 0 0 12px hsla(var(--beam-hue, 300), 90%, 70%, 0.4);
    animation: beam-sweep var(--beam-dur, 7s) linear infinite;
  }
  body.jam .disco-beam { opacity:0.85; height:120vh; }
  @keyframes beam-sweep {
    from { transform: translateX(-50%) rotate(var(--beam-start, 0deg)); }
    to   { transform: translateX(-50%) rotate(calc(var(--beam-start, 0deg) + 360deg)); }
  }

  body.jam { animation: jam-floor 3s ease-in-out infinite; }
  @keyframes jam-floor {
    0%,100% { background:radial-gradient(ellipse at top,#1a0f2a 0%,#07070b 70%); }
    50%     { background:radial-gradient(ellipse at top,#2a0f2a 0%,#0b0710 70%); }
  }
  details.sec { background:#0f0f16; border:1px solid #23232f;
    border-radius:10px; padding:0; }
  details.sec > summary { list-style:none; cursor:pointer;
    padding:10px 14px; font-size:10px; letter-spacing:0.16em;
    text-transform:uppercase; color:var(--dim); display:flex;
    align-items:center; gap:8px; user-select:none; }
  details.sec > summary::-webkit-details-marker { display:none; }
  details.sec > summary::before { content:'▸'; transition:transform 120ms;
    color:var(--dim); font-size:9px; }
  details.sec[open] > summary::before { transform:rotate(90deg); }
  details.sec[open] > summary { color:var(--ink); }
  details.sec > .sec-body { padding:0 14px 14px; display:flex;
    flex-direction:column; gap:12px; }
  @media (max-width: 900px) { .stage { grid-template-columns: 1fr; } .side { min-height:auto; } }
</style>
</head>
<body>
  <canvas id="stars"></canvas>
  <div id="disco"></div>
  <div class="disco-beam" style="--beam-hue:320; --beam-start:0deg;   --beam-dur:7s;"></div>
  <div class="disco-beam" style="--beam-hue:190; --beam-start:90deg;  --beam-dur:9s;"></div>
  <div class="disco-beam" style="--beam-hue:280; --beam-start:180deg; --beam-dur:8s;"></div>
  <div class="disco-beam" style="--beam-hue:60;  --beam-start:270deg; --beam-dur:10s;"></div>
  <div style="display:flex; gap:14px; align-items:center;">
    <h1>handsfree</h1>
    <div style="font-size:10px; color:var(--dim); letter-spacing:0.12em;">
      swipe L↔R → switch desktop • prayer hands → wispr
    </div>
    <div style="flex:1"></div>
    <button id="cc-btn"  type="button">control center</button>
    <button id="vt-btn"  type="button">voice test</button>
    <button id="jam-btn" type="button">jam mode</button>
    <button id="master-btn" type="button" title="Master on/off. Double-clap also toggles."
      style="font-size:11px; letter-spacing:0.18em; font-weight:800;
             padding:8px 18px; border-radius:999px; border:1px solid;">
      ON
    </button>
  </div>
  <!-- DJ Board: visible only in jam mode, above the Control Center. -->
  <div id="dj-board">
    <div class="dj-chan on" data-k="left">
      <div class="dj-head"><span>left hand</span><span class="dj-icon">✋</span></div>
      <div class="dj-toggle on" data-k="left">ON</div>
      <select class="dj-preset" data-k="left">
        <option value="warm">warm bass</option>
        <option value="sub">sub bass</option>
        <option value="acid">acid</option>
      </select>
      <div class="dj-meter"><div class="dj-meter-fill" data-k="left"></div></div>
    </div>
    <div class="dj-chan on" data-k="right">
      <div class="dj-head"><span>right hand</span><span class="dj-icon">🤚</span></div>
      <div class="dj-toggle on" data-k="right">ON</div>
      <select class="dj-preset" data-k="right">
        <option value="dreamy">dreamy lead</option>
        <option value="pluck">pluck</option>
        <option value="stab">chord stab</option>
      </select>
      <div class="dj-meter"><div class="dj-meter-fill" data-k="right"></div></div>
    </div>
    <div class="dj-chan on" data-k="smile">
      <div class="dj-head"><span>smile</span><span class="dj-icon">😊</span></div>
      <div class="dj-toggle on" data-k="smile">ON</div>
      <select class="dj-preset" data-k="smile">
        <option value="warm">warm pad</option>
        <option value="bright">bright pad</option>
        <option value="drone">drone</option>
      </select>
      <div class="dj-meter"><div class="dj-meter-fill" data-k="smile"></div></div>
    </div>
    <div class="dj-chan on" data-k="head">
      <div class="dj-head"><span>head bob</span><span class="dj-icon">🥁</span></div>
      <div class="dj-toggle on" data-k="head">ON</div>
      <select class="dj-preset" data-k="head">
        <option value="house">house kit</option>
        <option value="tight">tight kit</option>
        <option value="808">808</option>
      </select>
      <div class="dj-meter"><div class="dj-meter-fill" data-k="head"></div></div>
    </div>
    <div class="dj-master">
      <div class="dj-head"><span>master</span><span class="dj-icon">🎛️</span></div>
      <input id="dj-vol" type="range" class="dj-vol" min="0" max="100" value="75">
      <div>
        <div class="dj-bpm" id="dj-bpm">—</div>
        <div class="dj-bpm-label">bpm</div>
      </div>
    </div>
  </div>

  <div id="cc-panel">
<!-- cursor on/off is now the big master toggle in the top bar -->
    <div class="cc-row" style="border-bottom:1px dashed #333; padding-bottom:10px; margin-bottom:10px;">
      <div class="cc-label" style="color:#b48cff;">experiments</div>
      <div class="cc-opts" id="cc-exp-opts">
        <button class="cc-opt" data-exp="t_timeout" title="Make a T with both hands to toggle everything off / on">T ✋ timeout</button>
        <button class="cc-opt" data-exp="peace_rclick" title="Hold up a peace sign ✌️ to press-and-hold the mouse (drag / select). Release the sign to let go.">✌️ hold-to-drag</button>
        <button class="cc-opt" data-exp="head_copy" title="Bob your head up (chin-lift nod) to copy (Cmd+C). A native macOS toast confirms.">🙆 head-up copy</button>
        <button class="cc-opt" data-exp="thumbs_dclick" title="Thumbs up 👍 to paste (Cmd+V).">👍 paste</button>
        <button class="cc-opt" data-exp="fist_zoom" title="While the fist (grab) is held, push your hand toward the camera to zoom in, pull it back to zoom out. Side-to-side still pans.">🤛 fist depth zoom</button>
        <button class="cc-opt" data-exp="atelier" title="A-pose (fingertips together, wrists wide low) toggles ✨ Atelier mode — two-hand zoom & pan for Figma">✨ atelier</button>
        <button class="cc-opt" id="cc-atelier-manual" title="Force atelier mode on/off without doing the pose">atelier: off</button>
      </div>
    </div>
    <div class="cc-row" style="border-bottom:1px dashed #333; padding-bottom:10px; margin-bottom:10px;">
      <div class="cc-label" style="color:#fbbf24;">🙏 wispr quickstart</div>
      <div class="cc-opts" id="cc-wispr-preset-opts">
        <button class="cc-opt" data-preset="menu_toggle"
          title="Prayer hands click Wispr's menu bar icon to START listening. Un-pray clicks it again to STOP. Most reliable.">
          pray → toggle (menu)
        </button>
        <button class="cc-opt" data-preset="f19_double_toggle"
          title="Pray = double-tap F19. Requires F19 bound in Wispr + 'double-tap to toggle' enabled.">
          pray → toggle (F19×2)
        </button>
        <button class="cc-opt" data-preset="f19_hold"
          title="Pray = hold F19 down the whole time. Release = stop. Requires F19 bound in Wispr as press-and-hold.">
          pray → hold (F19)
        </button>
        <button class="cc-opt" data-preset="fn_hold"
          title="Pray = hold Fn (keycode 63). Original method. Works sometimes.">
          pray → hold (Fn)
        </button>
        <button class="cc-opt" data-preset="fn_tap_toggle"
          title="Pray = single Fn tap to toggle. If Wispr treats Fn as on/off, this starts+stops cleanly.">
          pray → tap (Fn)
        </button>
        <button class="cc-opt" data-preset="fn_double_toggle"
          title="Pray = double-tap Fn. Needs Wispr 'double-tap to toggle'.">
          pray → toggle (Fn×2)
        </button>
        <button class="cc-opt" data-preset="fn_hold_tap"
          title="Pray = hold Fn + extra Fn tap when hands release. Force-stops toggle-style apps.">
          pray → hold+tap (Fn)
        </button>
        <button class="cc-opt" data-preset="fn_hold_nuclear"
          title="Pray = hold Fn. On release fire EVERY stop signal: Fn-up, Fn-tap, Fn×2, F19, Escape, menu click. Use if nothing else stops Wispr.">
          pray → hold+💣 (Fn)
        </button>
        <button class="cc-opt" data-preset="off"
          title="Disable prayer → Wispr entirely">
          off
        </button>
      </div>
    </div>
    <div class="cc-row" style="border-bottom:1px dashed #333; padding-bottom:10px; margin-bottom:10px;">
      <div class="cc-label" style="color:#fbbf24;">🧪 Fn lab</div>
      <div class="cc-opts" id="cc-fn-diag-opts">
        <button class="cc-opt" data-sub="tap" title="Single Fn tap (keycode 63 + Fn flag)">tap</button>
        <button class="cc-opt" data-sub="double" title="Double Fn tap">double</button>
        <button class="cc-opt" data-sub="triple" title="Triple Fn tap">triple</button>
        <button class="cc-opt" data-sub="hold_500ms" title="Hold Fn for 500ms">hold 500ms</button>
        <button class="cc-opt" data-sub="hold_2s" title="Hold Fn for 2s">hold 2s</button>
        <button class="cc-opt" data-sub="down" title="Fn DOWN only (no up)">down</button>
        <button class="cc-opt" data-sub="up" title="Fn UP only">up</button>
        <button class="cc-opt" data-sub="escape" title="Tap Escape — may cancel an active Wispr recording">escape</button>
        <button class="cc-opt" data-sub="menu" title="Click Wispr menu bar icon via AppleScript">menu click</button>
        <button class="cc-opt" data-sub="nuclear" title="💣 Fire every plausible stop signal at once: Fn-up, Fn-tap, Fn×2, F19, Escape, menu click">💣 nuclear stop</button>
        <button class="cc-opt" id="cc-fn-release-tap" data-toggle="release_extra_tap"
          title="When 'hold' releases, also fire an extra Fn tap to force-stop toggle-style apps">
          release+tap
        </button>
        <button class="cc-opt" id="cc-fn-release-nuclear" data-toggle="release_nuclear"
          title="When 'hold' releases, fire the full 💣 nuclear stop cascade">
          release+💣
        </button>
      </div>
    </div>
    <div class="cc-row">
      <div class="cc-label">pointing</div>
      <div class="cc-opts" id="cc-point-opts">
        <button class="cc-opt" data-v="finger">finger</button>
        <button class="cc-opt" data-v="gaze">gaze</button>
        <button class="cc-opt" data-v="head">head</button>
      </div>
    </div>
    <div class="cc-row">
      <div class="cc-label">click</div>
      <div class="cc-opts" id="cc-click-opts">
        <button class="cc-opt" data-v="brow">brow raise</button>
        <button class="cc-opt" data-v="blink">blink</button>
        <button class="cc-opt" data-v="right_wink">right wink</button>
        <button class="cc-opt" data-v="wink">either wink</button>
        <button class="cc-opt" data-v="mouth">mouth open</button>
        <button class="cc-opt" data-v="pinch">pinch</button>
      </div>
    </div>
    <div class="cc-row">
      <div class="cc-label">wispr</div>
      <div class="cc-opts" id="cc-wispr-opts">
        <button class="cc-opt" data-v="off">off</button>
        <button class="cc-opt" data-v="cgevent_f19" title="Single F19 press — bind F19 in Wispr as your hotkey">cgevent f19</button>
        <button class="cc-opt" data-v="cgevent_fn" title="Synthetic Fn key — only works on some macOS versions">cgevent fn</button>
        <button class="cc-opt" data-v="double_tap_f19" title="Double-tap F19 — use with Wispr 'double-tap to toggle' mode">dbl-tap f19</button>
        <button class="cc-opt" data-v="double_tap_fn" title="Double-tap synthetic Fn — latches Wispr if it listens">dbl-tap fn</button>
        <button class="cc-opt" data-v="menu_click" title="Click Wispr menu bar icon via AppleScript — universal fallback">menu click</button>
        <button class="cc-opt" data-v="applescript_fn">applescript fn</button>
        <button class="cc-opt" data-v="apple_dictation">apple dictation</button>
        <button class="cc-opt" data-v="all" title="Fire every method in sequence — great for diagnosing which one works">all (test)</button>
        <button class="cc-opt" id="cc-wispr-fire">tap now</button>
        <button class="cc-opt" id="cc-wispr-holdtest">hold 2s test</button>
      </div>
    </div>
    <div class="cc-row">
      <div class="cc-label">dictation gesture</div>
      <div class="cc-opts" id="cc-dict-opts">
        <button class="cc-opt" data-v="off">off</button>
        <button class="cc-opt" data-v="prayer">prayer</button>
        <button class="cc-opt" data-v="fingertips">fingertips</button>
        <button class="cc-opt" data-v="fist">one-hand fist</button>
      </div>
    </div>
    <div class="cc-row">
      <div class="cc-label">dictation mode</div>
      <div class="cc-opts" id="cc-dictmode-opts">
        <button class="cc-opt" data-v="hold">hold · release to send</button>
        <button class="cc-opt" data-v="latch">latch · tap on / tap off</button>
      </div>
    </div>
    <div class="cc-row">
      <div class="cc-label">background</div>
      <div class="cc-opts" id="cc-bg-opts">
        <button class="cc-opt" data-v="off">off</button>
        <button class="cc-opt" data-v="stars">shooting stars</button>
        <button class="cc-opt" data-v="aurora">aurora</button>
        <button class="cc-opt" data-v="grid">retro grid</button>
        <button class="cc-opt" data-v="particles">dust motes</button>
        <button class="cc-opt" data-v="rain">matrix rain</button>
      </div>
    </div>
    <div class="cc-row">
      <div class="cc-label">clap toggle</div>
      <div class="cc-opts" id="cc-clap-opts">
        <button class="cc-opt" data-v="off">off</button>
        <button class="cc-opt" data-v="tight">A · tight</button>
        <button class="cc-opt" data-v="normal">B · normal</button>
        <button class="cc-opt" data-v="loose">C · loose</button>
        <button class="cc-opt" data-v="wide">D · wide</button>
        <button class="cc-opt" data-v="ironman">E · iron man</button>
      </div>
    </div>
    <div class="cc-row">
      <div class="cc-label">right click</div>
      <div class="cc-opts" id="cc-rclick-opts">
        <button class="cc-opt" data-v="off">off</button>
        <button class="cc-opt" data-v="smile">smile (hard)</button>
        <button class="cc-opt" data-v="pucker">pucker</button>
        <button class="cc-opt" data-v="furrow">furrow brow</button>
      </div>
    </div>
    <div class="cc-row">
      <div class="cc-label">double click</div>
      <div class="cc-opts" id="cc-dclick-opts">
        <button class="cc-opt" data-v="off">off</button>
        <button class="cc-opt" data-v="on">double-tap primary</button>
      </div>
    </div>
    <div class="cc-row">
      <div class="cc-label">sensitivity</div>
      <div class="cc-opts" style="flex:1; min-width:0;">
        <input type="range" id="cc-sens" min="0.3" max="3.0" step="0.05"
               value="1.5" style="flex:1; accent-color:var(--accent);
               min-width:140px;">
        <span id="cc-sens-val" style="font-size:11px; color:var(--dim);
              min-width:42px; text-align:right;">1.50×</span>
      </div>
    </div>
    <div class="cc-row">
      <div class="cc-label">scroll mode</div>
      <div class="cc-opts" id="cc-scroll-mode-opts">
        <button class="cc-opt" data-v="off">off</button>
        <button class="cc-opt" data-v="fist" title="Close one hand into a fist; move it to scroll">fist</button>
        <button class="cc-opt" data-v="two_hands" title="Raise both hands; move them together to scroll">two hands</button>
        <button class="cc-opt" data-v="head_lefthand" title="Raise left hand, then tilt chin up/down">✋ + head</button>
        <button class="cc-opt" data-v="head_mouth" title="Open mouth to gate, then tilt chin up/down">👄 + head</button>
        <button class="cc-opt" data-v="brow" title="Raise eyebrows = up, furrow = down">🙁 brows</button>
        <button class="cc-opt" data-v="head_always" title="Always on — head pitch scrolls any time. Drifty.">head only</button>
      </div>
    </div>
    <div class="cc-row">
      <div class="cc-label">scroll speed</div>
      <div class="cc-opts" style="flex:1; min-width:0;">
        <input type="range" id="cc-scroll-speed" min="0.5" max="6.0" step="0.1"
               value="3.0" style="flex:1; accent-color:var(--accent);
               min-width:140px;">
        <span id="cc-scroll-speed-val" style="font-size:11px; color:var(--dim);
              min-width:42px; text-align:right;">3.0×</span>
      </div>
    </div>
    <div class="cc-row">
      <div class="cc-label">tab swipe</div>
      <div class="cc-opts" id="cc-tabswipe-opts">
        <button class="cc-opt" data-v="off">off</button>
        <button class="cc-opt" data-v="tabs" title="Two hands raised + sweep → Cmd+Shift+]/[ (switch browser tabs)">tabs</button>
        <button class="cc-opt" data-v="spaces" title="Two hands raised + sweep → Ctrl+Left/Right (switch macOS Spaces, like 3-finger swipe)">spaces (3-finger)</button>
      </div>
    </div>
    <div class="cc-row">
      <div class="cc-label">voice daemon</div>
      <div class="cc-opts" id="cc-voicedaemon-opts">
        <button class="cc-opt" data-v="off">off</button>
        <button class="cc-opt" data-v="on">on</button>
      </div>
      <div id="cc-voice-status" style="font-size:10px; color:var(--dim);
           margin-left:10px;">idle</div>
    </div>
    <div id="cc-voice-log" style="display:none; margin-top:-4px;
         margin-bottom:10px; background:#05080f; border:1px solid #1a2030;
         border-radius:6px; padding:8px 10px; font-size:11px;
         line-height:1.5; font-family:ui-monospace,Menlo,monospace;
         max-height:160px; overflow-y:auto;"></div>
    <div class="cc-row">
      <div class="cc-label">swipe pages</div>
      <div class="cc-opts" id="cc-swipe-opts">
        <button class="cc-opt" data-v="off">off</button>
        <button class="cc-opt" data-v="on">on</button>
      </div>
    </div>
    <div class="cc-row">
      <div class="cc-label">voice commands</div>
      <div class="cc-opts" id="cc-voice-opts">
        <button class="cc-opt" data-v="off">off</button>
        <button class="cc-opt" data-v="on">on</button>
      </div>
    </div>
    <div class="cc-row" id="cc-voice-panel" style="display:none;">
      <div class="cc-label">mappings</div>
      <div class="cc-opts" style="flex:1; min-width:0; flex-direction:column;
           align-items:stretch; gap:4px;">
        <textarea id="cc-voice-map" spellcheck="false"
          style="width:100%; min-height:80px; background:#0f1420;
          color:var(--fg); border:1px solid #2a3550; border-radius:6px;
          font:11px/1.5 ui-monospace,Menlo,monospace; padding:6px 8px;
          resize:vertical;"
          placeholder="telegram: cmd+t&#10;arc: cmd+a"></textarea>
        <div style="font-size:10px; color:var(--dim); line-height:1.4;">
          format: <code>phrase: cmd+key</code> per line. heard text is matched
          as substring (case-insensitive). mods: cmd/shift/opt/ctrl.
          <span id="cc-voice-heard" style="color:var(--accent);
            margin-left:6px;"></span>
        </div>
      </div>
    </div>
    <div class="cc-row">
      <div class="cc-label">master</div>
      <div class="cc-opts">
        <span id="cc-master-state" style="font-size:10px; color:var(--dim);
              letter-spacing:0.14em; text-transform:uppercase;">on</span>
        <span id="cc-clap-indicator" style="font-size:10px; color:var(--dim);
              letter-spacing:0.14em;">claps: 0 — double clap to toggle</span>
      </div>
    </div>
    <div class="cc-hint">
      hands overhead → cursor on/off (re-calibrates each time).<br>
      head: hold still 1.2s facing center. finger: point right index at
      center. gaze: look straight at camera.<br>
      clicks: brow raise, deliberate blink, right-eye wink, either-eye wink,
      open mouth, or pinch — all edge-triggered.<br>
      <b>scroll:</b> raise both hands and move them up/down together. one
      hand = cursor, two hands = scroll.<br>
      <b>dictation:</b> <i>wispr</i> uses Wispr Flow hotkeys; <i>apple
      dictation</i> double-taps Control (set this in System Settings →
      Keyboard → Dictation → Shortcut = "Press Control key twice").<br>
      "test all" fires cgevent-F19, cgevent-Fn, and applescript-Fn
      in sequence to help you discover which one wakes Wispr.
    </div>
  </div>

  <!-- Voice Test panel. Dedicated UI for getting "open telegram" working
       end-to-end. Status, live transcript, quick-test buttons, text sim,
       event log. Toggled by the "voice test" button in the top bar. -->
  <div id="vt-panel" style="display:none; position:absolute; top:70px;
       right:16px; width:380px; max-width:90vw; background:#0f1420;
       border:1px solid #2a3550; border-radius:12px; padding:14px;
       box-shadow:0 10px 40px rgba(0,0,0,0.5); z-index:50;
       font-size:12px; line-height:1.5;">
    <div style="display:flex; align-items:center; gap:8px;
         margin-bottom:10px;">
      <b style="letter-spacing:0.1em; text-transform:uppercase;
         font-size:11px; color:var(--accent);">voice test</b>
      <div style="flex:1"></div>
      <button id="vt-close" type="button"
        style="background:none; border:none; color:var(--dim);
        font-size:16px; cursor:pointer; padding:0;">×</button>
    </div>

    <div style="display:flex; gap:6px; flex-wrap:wrap;
         margin-bottom:10px;">
      <span id="vt-mic-pill"
        style="font-size:10px; letter-spacing:0.14em;
        text-transform:uppercase; padding:3px 8px; border-radius:999px;
        background:#1a1a24; color:var(--dim);">mic: ?</span>
      <span id="vt-listen-pill"
        style="font-size:10px; letter-spacing:0.14em;
        text-transform:uppercase; padding:3px 8px; border-radius:999px;
        background:#1a1a24; color:var(--dim);">listening: off</span>
    </div>

    <div style="display:flex; gap:6px; margin-bottom:10px;">
      <button id="vt-request-mic" type="button"
        style="flex:1; padding:6px 10px; background:#1b2530;
        color:var(--fg); border:1px solid #2a3550; border-radius:6px;
        font-size:11px; cursor:pointer;">grant mic permission</button>
      <button id="vt-toggle" type="button"
        style="flex:1; padding:6px 10px; background:var(--accent);
        color:#05170f; border:none; border-radius:6px; font-size:11px;
        font-weight:700; cursor:pointer;">start listening</button>
    </div>

    <div style="font-size:10px; color:var(--dim); margin-bottom:4px;">
      local whisper (works when browser speech doesn't) —
      hold to record, release to fire
    </div>
    <div style="display:flex; gap:6px; margin-bottom:10px;">
      <button id="vt-ptt" type="button"
        style="flex:1; padding:10px; background:#1b2530; color:var(--fg);
        border:1px solid #2a3550; border-radius:6px; font-size:12px;
        font-weight:700; cursor:pointer; user-select:none;
        -webkit-user-select:none;">hold to record</button>
      <button id="vt-rec-toggle" type="button"
        style="padding:10px 14px; background:#1b2530; color:var(--fg);
        border:1px solid #2a3550; border-radius:6px; font-size:11px;
        cursor:pointer;">● rec</button>
    </div>

    <div style="font-size:10px; color:var(--dim); margin-bottom:4px;">
      live transcript
    </div>
    <div id="vt-transcript"
      style="min-height:44px; background:#05080f; border:1px solid #1a2030;
      border-radius:6px; padding:8px 10px; margin-bottom:10px;
      color:var(--fg); font-family:ui-monospace,Menlo,monospace;
      font-size:12px;">
      <span style="color:var(--dim);">(nothing yet)</span>
    </div>

    <div style="font-size:10px; color:var(--dim); margin-bottom:4px;">
      quick tests (click to fire — no voice needed)
    </div>
    <div id="vt-quick"
      style="display:flex; flex-wrap:wrap; gap:4px; margin-bottom:10px;">
      <button data-phrase="open telegram" class="vt-q">open telegram</button>
      <button data-phrase="open notion"   class="vt-q">open notion</button>
      <button data-phrase="open arc"      class="vt-q">open arc</button>
      <button data-phrase="open safari"   class="vt-q">open safari</button>
      <button data-phrase="open slack"    class="vt-q">open slack</button>
      <button data-phrase="open spotify"  class="vt-q">open spotify</button>
    </div>

    <div style="font-size:10px; color:var(--dim); margin-bottom:4px;">
      simulate a phrase (type instead of speaking)
    </div>
    <div style="display:flex; gap:4px; margin-bottom:10px;">
      <input id="vt-sim-input" type="text"
        placeholder='e.g. "open notion"'
        style="flex:1; background:#05080f; color:var(--fg);
        border:1px solid #2a3550; border-radius:6px; padding:6px 8px;
        font-size:12px;">
      <button id="vt-sim-fire" type="button"
        style="padding:6px 10px; background:#1b2530; color:var(--fg);
        border:1px solid #2a3550; border-radius:6px; font-size:11px;
        cursor:pointer;">fire</button>
    </div>

    <div style="font-size:10px; color:var(--dim); margin-bottom:4px;">
      event log
    </div>
    <div id="vt-log"
      style="max-height:180px; overflow-y:auto; background:#05080f;
      border:1px solid #1a2030; border-radius:6px; padding:6px 8px;
      font-family:ui-monospace,Menlo,monospace; font-size:11px;
      line-height:1.6;">
      <div style="color:var(--dim);">ready.</div>
    </div>
  </div>

  <div class="stage">
    <img src="/stream" alt="camera">
    <div class="side">
      <div id="status" class="status off">click page → allow mic → say "open arc"</div>
      <div style="display:flex; gap:8px; flex-wrap:wrap;">
        <div id="listening-pill" style="display:none;
             font-size:10px; letter-spacing:0.16em; text-transform:uppercase;
             color:#05170f; background:var(--warm); padding:4px 10px;
             border-radius:999px; font-weight:700;">listening…</div>
        <div id="cursor-pill" style="display:none;
             font-size:10px; letter-spacing:0.16em; text-transform:uppercase;
             color:#05170f; background:var(--cool); padding:4px 10px;
             border-radius:999px; font-weight:700;">cursor</div>
        <div id="voice-pill"
             style="font-size:10px; letter-spacing:0.16em; text-transform:uppercase;
             color:var(--dim); background:#1a1a24; padding:4px 10px;
             border-radius:999px; font-weight:700;">voice: off</div>
        <div id="atelier-pill" style="display:none;
             font-size:11px; letter-spacing:0.22em; text-transform:uppercase;
             color:#05170f;
             background:linear-gradient(90deg,#f5c24a,#ff7ad9,#7aa7ff);
             padding:5px 12px; border-radius:999px; font-weight:800;
             box-shadow:0 0 18px rgba(245,194,74,0.6);">✨ atelier</div>
      </div>

      <details class="sec">
        <summary>voice & heard</summary>
        <div class="sec-body">
          <div id="heard" style="font-size:14px; line-height:1.5; min-height:64px;
               max-height:120px; overflow:hidden; padding:10px 12px;
               background:#0a0a0f; border:1px solid #262634; border-radius:10px;
               color:var(--ink);">
            <span class="interim" style="color:var(--dim); font-style:italic;">
              say something…
            </span>
          </div>
        </div>
      </details>

      <details class="sec">
        <summary>activity</summary>
        <div class="sec-body">
          <div class="row"><div class="label">smile</div><div class="bar"><div id="smile-fill" class="fill smile"></div></div></div>
          <div class="row"><div class="label">left hand</div><div class="bar"><div id="left-fill" class="fill motion"></div></div></div>
          <div class="row"><div class="label">right hand</div><div class="bar"><div id="right-fill" class="fill motion"></div></div></div>
        </div>
      </details>

      <details class="sec">
        <summary>dictation notes</summary>
        <div class="sec-body">
          <textarea id="notes" placeholder="prayer hands → dictated text lands here"
            style="min-height:120px; resize:vertical; background:#0a0a0f;
                   color:var(--ink); border:1px solid #262634; border-radius:10px;
                   padding:10px; font-family:inherit; font-size:13px;
                   line-height:1.5;"></textarea>
        </div>
      </details>

      <details class="sec">
        <summary>cheat sheet</summary>
        <div class="sec-body">
          <div style="color:var(--dim); font-size:11px; line-height:1.7;">
            one hand → cursor<br>
            two hands → scroll<br>
            prayer (hold) → dictate<br>
            double clap → voice on/off<br>
            swipe → next/prev desktop<br>
            bob → drum · smile → pad
          </div>
        </div>
      </details>
    </div>
  </div>
  <div id="bob-flash" class="clap-flash"></div>

<script>
(function() {
  const statusEl  = document.getElementById('status');
  const smileFill = document.getElementById('smile-fill');
  const leftFill  = document.getElementById('left-fill');
  const rightFill = document.getElementById('right-fill');
  const flashEl   = document.getElementById('bob-flash');
  const listenPill= document.getElementById('listening-pill');
  const cursorPill= document.getElementById('cursor-pill');
  const voicePill = document.getElementById('voice-pill');
  const heardEl   = document.getElementById('heard');
  const notesEl   = document.getElementById('notes');

  // ---- Voice commands (Web Speech API) ----------------------------------
  // Freeform: "open <anything>" / "close <anything>". The backend hands the
  // raw app name to `open -a`, which fuzzy-matches installed apps.
  const OPEN_RE  = /\b(?:open|launch)\s+([a-z0-9][a-z0-9 \-\.]{0,30})/i;
  const CLOSE_RE = /\b(?:close|quit)\s+([a-z0-9][a-z0-9 \-\.]{0,30})/i;
  const INSTANT = [
    [/\bscroll\s+down\b/i,       { action: 'scroll', target: 'down' }],
    [/\bscroll\s+up\b/i,         { action: 'scroll', target: 'up' }],
    [/\bpage\s+down\b/i,         { action: 'scroll', target: 'down' }],
    [/\bpage\s+up\b/i,           { action: 'scroll', target: 'up' }],
    [/\b(?:click|tap)\b/i,       { action: 'click',  target: '' }],
    [/\bvolume\s+up\b/i,         { action: 'volume', target: 'up' }],
    [/\bvolume\s+down\b/i,       { action: 'volume', target: 'down' }],
    [/\b(?:mute|silence)\b/i,    { action: 'volume', target: 'mute' }],
    [/\bnext\s+desktop\b/i,      { action: 'desktop', target: 'next' }],
    [/\bprevious\s+desktop\b/i,  { action: 'desktop', target: 'prev' }],
  ];

  function cleanTarget(s) {
    return (s || '').trim()
      .replace(/^(the|up)\s+/i, '')
      .replace(/\s+(please|now|app|application)$/i, '')
      .trim();
  }

  function matchCommand(text) {
    const t = (text || '').toLowerCase().trim();
    if (!t) return null;
    for (const [re, cmd] of INSTANT) {
      if (re.test(t)) return cmd;
    }
    let m = t.match(OPEN_RE);
    if (m) return { action: 'open', target: cleanTarget(m[1]) };
    m = t.match(CLOSE_RE);
    if (m) return { action: 'close', target: cleanTarget(m[1]) };
    return null;
  }

  function sendCommand(cmd) {
    fetch('/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cmd),
    }).catch(() => {});
  }

  function setVoicePill(state, text) {
    voicePill.textContent = 'voice: ' + text;
    if (state === 'on')   { voicePill.style.background = '#6ee7b7'; voicePill.style.color = '#05170f'; }
    else if (state === 'err') { voicePill.style.background = '#f87171'; voicePill.style.color = '#1a0707'; }
    else                      { voicePill.style.background = '#1a1a24'; voicePill.style.color = 'var(--dim)'; }
  }

  // Small ack sound (two sines) so you know listening flipped on/off.
  function ackTone(up) {
    if (!ctx) return;
    const t = ctx.currentTime;
    const freqs = up ? [659.25, 987.77] : [987.77, 659.25];
    freqs.forEach((f, i) => {
      const o = ctx.createOscillator();
      o.type = 'sine'; o.frequency.value = f;
      const g = ctx.createGain();
      const s = t + i * 0.08;
      g.gain.setValueAtTime(0.0001, s);
      g.gain.exponentialRampToValueAtTime(0.18, s + 0.01);
      g.gain.exponentialRampToValueAtTime(0.001, s + 0.25);
      o.connect(g).connect(ctx.destination);
      o.start(s); o.stop(s + 0.3);
    });
  }

  let rec = null;
  let voiceWanted = false;  // user intent: should we be listening?
  let lastFired = 0;

  async function ensureMic() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach(t => t.stop());  // release — SR will reopen
      window.__lastMicErr = null;
      return true;
    } catch (e) {
      console.warn('mic denied', e);
      // Stash the real reason so the voice test panel can surface it.
      window.__lastMicErr = (e && (e.name || e.message)) || 'unknown';
      setVoicePill('err', 'mic denied');
      return false;
    }
  }

  function createRecognizer() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      setVoicePill('err', 'unsupported');
      if (window.__vtLog) window.__vtLog('sr', 'SpeechRecognition not available in this browser', '#f87171');
      return null;
    }
    const r = new SR();
    r.continuous = true;
    r.interimResults = true;
    r.lang = 'en-US';
    r.onstart = () => {
      setVoicePill('on', 'listening');
      if (window.__vtLog) window.__vtLog('sr', 'onstart — actually listening', '#6ee7b7');
    };
    r.onaudiostart    = () => { if (window.__vtLog) window.__vtLog('sr', 'audiostart',    'var(--dim)'); };
    r.onsoundstart    = () => { if (window.__vtLog) window.__vtLog('sr', 'soundstart',    'var(--dim)'); };
    r.onspeechstart   = () => { if (window.__vtLog) window.__vtLog('sr', 'speechstart',   'var(--dim)'); };
    r.onspeechend     = () => { if (window.__vtLog) window.__vtLog('sr', 'speechend',     'var(--dim)'); };
    r.onnomatch       = () => { if (window.__vtLog) window.__vtLog('sr', 'nomatch',       '#fbbf24'); };
    r.onerror = (e) => {
      const err = e && e.error ? e.error : 'unknown';
      const msg = e && e.message ? ' — ' + e.message : '';
      if (window.__vtLog) window.__vtLog('sr-err', err + msg, '#f87171');
      if (err === 'not-allowed' || err === 'service-not-allowed') {
        voiceWanted = false;
        setVoicePill('err', 'mic blocked');
      }
      // network / aborted / no-speech → let onend restart. Do not flicker pill.
    };
    r.onend = () => {
      if (window.__vtLog) window.__vtLog('sr', 'onend', 'var(--dim)');
      if (voiceWanted) {
        setTimeout(() => { try { r.start(); } catch {} }, 500);
      } else {
        setVoicePill('off', 'off');
      }
    };
    r.onresult = (ev) => {
      let interim = '', finalText = '';
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const res = ev.results[i];
        if (res.isFinal) finalText += res[0].transcript + ' ';
        else interim += res[0].transcript + ' ';
      }
      const text = (finalText + interim).trim();
      if (text) {
        heardEl.innerHTML = finalText
          ? `<span class="final">${escapeHtml(finalText)}</span>`
            + `<span class="interim">${escapeHtml(interim)}</span>`
          : `<span class="interim">${escapeHtml(interim)}</span>`;
      }
      if (window.__vtTranscript) window.__vtTranscript(finalText, interim);
      const phrase = (finalText || interim).trim();
      const cmd = matchCommand(phrase);
      const now = Date.now();
      if (cmd && cmd.target !== undefined && now - lastFired > 1500) {
        lastFired = now;
        flashBob();
        sendCommand(cmd);
        if (window.__vtFire) window.__vtFire(phrase, cmd, 'voice');
      }
    };
    return r;
  }

  function escapeHtml(s) {
    return (s || '').replace(/[&<>\"']/g, c => ({
      '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
    }[c]));
  }

  async function startVoice() {
    if (voiceWanted) return;
    const ok = await ensureMic();
    if (!ok) return;
    if (!rec) rec = createRecognizer();
    if (!rec) return;
    voiceWanted = true;
    setVoicePill('on', 'starting…');
    try { rec.start(); } catch {}
    ackTone(true);
  }

  function stopVoice() {
    if (!voiceWanted) return;
    voiceWanted = false;
    if (rec) { try { rec.stop(); } catch {} }
    setVoicePill('off', 'off');
    ackTone(false);
  }

  function toggleVoice() {
    if (voiceWanted) stopVoice(); else startVoice();
  }

  // Clickable voice pill doubles as on/off button.
  voicePill.style.cursor = 'pointer';
  voicePill.addEventListener('click', toggleVoice);

  // Jam Mode button: hides voice UI, disables swipe, routes hands to synths.
  const jamBtn = document.getElementById('jam-btn');
  function setJamMode(on) {
    jamMode = on;
    document.body.classList.toggle('jam', on);
    jamBtn.classList.toggle('on', on);
    jamBtn.textContent = on ? 'jam mode · on' : 'jam mode';
    if (on && voiceWanted) stopVoice();
    // Immediately silence whichever synth bank is now idle.
    if (on) {
      if (leftTh) leftTh.setHandY(null);
      if (rightTh) rightTh.setHandY(null);
    } else {
      if (jamBass) jamBass.setXY(null, null);
      if (jamLead) jamLead.setXY(null, null);
    }
    fetch('/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'jam', on }),
    }).catch(() => {});
  }
  jamBtn.addEventListener('click', () => {
    unlockAudio();
    setJamMode(!jamMode);
  });

  // ---------- DJ Board wiring ----------
  function djSetMeter(k, v) {
    const el = document.querySelector('.dj-meter-fill[data-k="'+k+'"]');
    if (!el) return;
    el.style.width = Math.round(Math.max(0, Math.min(1, v)) * 100) + '%';
  }
  let _djFlashT = {};
  function djFlashMeter(k) {
    const el = document.querySelector('.dj-meter-fill[data-k="'+k+'"]');
    if (!el) return;
    el.style.width = '100%';
    clearTimeout(_djFlashT[k]);
    _djFlashT[k] = setTimeout(() => { el.style.width = '0%'; }, 140);
  }
  function djTrackBPM() {
    const now = performance.now();
    bobTimes.push(now);
    while (bobTimes.length > 8) bobTimes.shift();
    if (bobTimes.length >= 3) {
      const intervals = [];
      for (let i = 1; i < bobTimes.length; i++) intervals.push(bobTimes[i] - bobTimes[i-1]);
      intervals.sort((a,b) => a - b);
      const med = intervals[Math.floor(intervals.length / 2)];
      if (med > 120 && med < 2000) {
        bobBPM = Math.round(60000 / med);
        const el = document.getElementById('dj-bpm');
        if (el) el.textContent = String(bobBPM);
      }
    }
  }
  // drop stale bpm after 4s of no bobs
  setInterval(() => {
    if (bobTimes.length && performance.now() - bobTimes[bobTimes.length-1] > 4000) {
      bobTimes.length = 0;
      bobBPM = 0;
      const el = document.getElementById('dj-bpm');
      if (el) el.textContent = '—';
    }
  }, 1000);
  document.querySelectorAll('.dj-toggle').forEach(btn => {
    btn.addEventListener('click', () => {
      const k = btn.dataset.k;
      djEnabled[k] = !djEnabled[k];
      btn.classList.toggle('on', djEnabled[k]);
      btn.textContent = djEnabled[k] ? 'ON' : 'OFF';
      const chan = btn.closest('.dj-chan');
      if (chan) chan.classList.toggle('on', djEnabled[k]) && chan.classList.toggle('off', !djEnabled[k]);
      if (chan) { chan.classList.toggle('off', !djEnabled[k]); chan.classList.toggle('on', djEnabled[k]); }
      // Silence immediately when turned off
      if (!djEnabled[k]) {
        if (k === 'left'  && jamBass) jamBass.setXY(null, null);
        if (k === 'right' && jamLead) jamLead.setXY(null, null);
        if (k === 'smile' && pad) pad.setAmount(0);
      }
    });
  });
  document.querySelectorAll('.dj-preset').forEach(sel => {
    sel.addEventListener('change', () => {
      const k = sel.dataset.k;
      const v = sel.value;
      djPreset[k] = v;
      if (k === 'left'  && jamBass && jamBass.setPreset) jamBass.setPreset(v);
      if (k === 'right' && jamLead && jamLead.setPreset) jamLead.setPreset(v);
      if (k === 'smile' && pad && pad.setPreset) pad.setPreset(v);
      if (k === 'head'  && drums && drums.setPreset) drums.setPreset(v);
    });
  });
  const djVol = document.getElementById('dj-vol');
  if (djVol) {
    djVol.addEventListener('input', () => {
      const v = Number(djVol.value) / 100;
      if (masterGain) masterGain.gain.value = v;
    });
  }

  // Voice Test panel: dedicated UI for verifying "open telegram" etc.
  (function initVoiceTest() {
    const btn       = document.getElementById('vt-btn');
    const panel     = document.getElementById('vt-panel');
    const closeBtn  = document.getElementById('vt-close');
    const micPill   = document.getElementById('vt-mic-pill');
    const listenPill= document.getElementById('vt-listen-pill');
    const reqMicBtn = document.getElementById('vt-request-mic');
    const toggleBtn = document.getElementById('vt-toggle');
    const transEl   = document.getElementById('vt-transcript');
    const quickWrap = document.getElementById('vt-quick');
    const simInput  = document.getElementById('vt-sim-input');
    const simFire   = document.getElementById('vt-sim-fire');
    const logEl     = document.getElementById('vt-log');
    if (!btn || !panel) return;

    let open = false;
    function setOpen(on) {
      open = on;
      panel.style.display = on ? 'block' : 'none';
      btn.classList.toggle('on', on);
    }
    btn.addEventListener('click', () => setOpen(!open));
    closeBtn.addEventListener('click', () => setOpen(false));

    function pill(el, state, text) {
      el.textContent = text;
      if (state === 'on')       { el.style.background = '#6ee7b7'; el.style.color = '#05170f'; }
      else if (state === 'err') { el.style.background = '#f87171'; el.style.color = '#1a0707'; }
      else                      { el.style.background = '#1a1a24'; el.style.color = 'var(--dim)'; }
    }

    function stamp() {
      const d = new Date();
      return d.toTimeString().slice(0, 8);
    }
    const MAX_LOG = 40;
    function log(kind, text, color) {
      const row = document.createElement('div');
      row.style.color = color || 'var(--fg)';
      row.textContent = `[${stamp()}] ${kind}: ${text}`;
      // On first entry, clear the "ready." placeholder.
      if (logEl.children.length === 1 && logEl.children[0].textContent === 'ready.') {
        logEl.innerHTML = '';
      }
      logEl.appendChild(row);
      while (logEl.children.length > MAX_LOG) logEl.removeChild(logEl.firstChild);
      logEl.scrollTop = logEl.scrollHeight;
    }
    // Expose so the recognizer (defined outside this IIFE) can log too.
    window.__vtLog = log;

    // Capability sniff so we know what browser reality we're in.
    const hasSR = !!(window.SpeechRecognition || window.webkitSpeechRecognition);
    log('env', (navigator.userAgent.match(/Chrome\/[\d.]+|Safari\/[\d.]+|Arc/g) || ['unknown']).join(' '), 'var(--dim)');
    log('env', 'SpeechRecognition: ' + (hasSR ? 'present' : 'MISSING'),
        hasSR ? 'var(--dim)' : '#f87171');

    // Mic permission detection — best-effort via Permissions API.
    async function refreshMicStatus() {
      try {
        if (navigator.permissions && navigator.permissions.query) {
          const p = await navigator.permissions.query({ name: 'microphone' });
          if (p.state === 'granted')       pill(micPill, 'on',  'mic: granted');
          else if (p.state === 'denied')   pill(micPill, 'err', 'mic: denied');
          else                             pill(micPill, 'off', 'mic: ask');
          p.onchange = refreshMicStatus;
          return;
        }
      } catch {}
      pill(micPill, 'off', 'mic: ?');
    }
    refreshMicStatus();

    reqMicBtn.addEventListener('click', async () => {
      log('mic', 'requesting permission…', 'var(--dim)');
      const ok = await ensureMic();
      if (ok) {
        log('mic', 'granted', '#6ee7b7');
      } else {
        const reason = window.__lastMicErr || 'unknown';
        log('mic', 'denied — ' + reason, '#f87171');
        if (reason === 'NotAllowedError') {
          log('fix', 'click the 🛡️/lock icon in Arc URL bar → ' +
              'set Microphone to Allow → reload this page', '#fbbf24');
        } else if (reason === 'NotFoundError') {
          log('fix', 'no mic device found — check macOS Sound settings',
              '#fbbf24');
        }
      }
      refreshMicStatus();
    });

    function paintListening() {
      if (voiceWanted) pill(listenPill, 'on',  'listening: on');
      else             pill(listenPill, 'off', 'listening: off');
      toggleBtn.textContent = voiceWanted ? 'stop listening' : 'start listening';
    }
    paintListening();

    toggleBtn.addEventListener('click', async () => {
      if (voiceWanted) {
        stopVoice();
        log('listen', 'stopped', 'var(--dim)');
      } else {
        await startVoice();
        log('listen', voiceWanted ? 'started' : 'failed to start',
            voiceWanted ? '#6ee7b7' : '#f87171');
      }
      paintListening();
      refreshMicStatus();
    });

    // Keep pill in sync if voice state changes via other paths.
    setInterval(paintListening, 750);

    // Expose hooks the recognizer's onresult calls into.
    window.__vtTranscript = (finalText, interim) => {
      const f = (finalText || '').trim();
      const i = (interim  || '').trim();
      if (!f && !i) return;
      transEl.innerHTML =
        (f ? `<span style="color:var(--fg)">${escapeHtml(f)}</span> ` : '') +
        (i ? `<span style="color:var(--dim)">${escapeHtml(i)}</span>` : '');
      if (f) log('heard', f, 'var(--fg)');
    };
    window.__vtFire = (phrase, cmd, source) => {
      const src = source || 'voice';
      if (!cmd) {
        log(src, `"${phrase}" → no match`, '#f87171');
        return;
      }
      log(src, `"${phrase}" → ${cmd.action} ${cmd.target || ''}`, '#6ee7b7');
    };

    // Quick-test buttons.
    quickWrap.querySelectorAll('.vt-q').forEach(b => {
      b.addEventListener('click', () => {
        const phrase = b.dataset.phrase || '';
        const cmd = matchCommand(phrase);
        window.__vtFire(phrase, cmd, 'click');
        if (cmd) sendCommand(cmd);
      });
    });

    // ---- Local Whisper push-to-talk --------------------------------------
    // Records audio via MediaRecorder → POSTs to /transcribe → server runs
    // faster-whisper locally, matches command, fires it, returns text +
    // result which we render in the log.
    const pttBtn    = document.getElementById('vt-ptt');
    const recToggle = document.getElementById('vt-rec-toggle');
    let mediaRec = null;
    let mediaStream = null;
    let recChunks = [];
    let recording = false;

    function paintRec(on) {
      recording = on;
      pttBtn.style.background = on ? '#f87171' : '#1b2530';
      pttBtn.style.color      = on ? '#1a0707' : 'var(--fg)';
      pttBtn.textContent      = on ? '● recording…' : 'hold to record';
      recToggle.textContent   = on ? '■ stop' : '● rec';
    }

    async function startRec() {
      if (recording) return;
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      } catch (e) {
        log('rec', 'mic denied: ' + (e && e.name), '#f87171');
        return;
      }
      recChunks = [];
      // webm/opus is the Chromium default and whisper (via ffmpeg/av) groks it.
      const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : (MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : '');
      try {
        mediaRec = mime ? new MediaRecorder(mediaStream, { mimeType: mime })
                        : new MediaRecorder(mediaStream);
      } catch (e) {
        log('rec', 'recorder err: ' + e.message, '#f87171');
        mediaStream.getTracks().forEach(t => t.stop());
        return;
      }
      mediaRec.ondataavailable = (ev) => {
        if (ev.data && ev.data.size > 0) recChunks.push(ev.data);
      };
      mediaRec.onstop = async () => {
        mediaStream.getTracks().forEach(t => t.stop());
        const blob = new Blob(recChunks, {
          type: mediaRec.mimeType || 'audio/webm',
        });
        log('rec', `captured ${(blob.size/1024).toFixed(1)} KB, transcribing…`, 'var(--dim)');
        try {
          const res = await fetch('/transcribe', {
            method: 'POST',
            headers: { 'Content-Type': blob.type },
            body: blob,
          });
          const j = await res.json();
          if (!j.ok) {
            log('whisper', 'error: ' + (j.error || 'unknown'), '#f87171');
            return;
          }
          const heard = j.text || '(silence)';
          transEl.innerHTML = `<span style="color:var(--fg)">${escapeHtml(heard)}</span>`;
          log('whisper', `(${j.ms}ms) "${heard}"`, 'var(--fg)');
          if (j.command) {
            const c = j.command;
            log('match', `${c.action} ${c.target || ''}`.trim(), '#6ee7b7');
          } else {
            log('match', 'no command', '#fbbf24');
          }
          if (j.fired) {
            log('fired', JSON.stringify(j.fired), j.fired.ok ? '#6ee7b7' : '#f87171');
          }
        } catch (e) {
          log('whisper', 'fetch err: ' + e.message, '#f87171');
        }
      };
      mediaRec.start();
      paintRec(true);
    }

    function stopRec() {
      if (!recording || !mediaRec) return;
      try { mediaRec.stop(); } catch {}
      paintRec(false);
    }

    // Push-to-talk: hold the button. Works with mouse and touch.
    pttBtn.addEventListener('pointerdown', (e) => { e.preventDefault(); startRec(); });
    pttBtn.addEventListener('pointerup',   (e) => { e.preventDefault(); stopRec();  });
    pttBtn.addEventListener('pointerleave',          () => { if (recording) stopRec(); });
    pttBtn.addEventListener('pointercancel',         () => { if (recording) stopRec(); });

    // Toggle record: click start, click stop. For when holding is awkward.
    recToggle.addEventListener('click', () => {
      if (recording) stopRec(); else startRec();
    });

    // Sim input — type + fire.
    function fireSim() {
      const phrase = (simInput.value || '').trim();
      if (!phrase) return;
      const cmd = matchCommand(phrase);
      window.__vtFire(phrase, cmd, 'typed');
      if (cmd) sendCommand(cmd);
    }
    simFire.addEventListener('click', fireSim);
    simInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') fireSim();
    });
  })();

  // Control Center: pointing/click method picker.
  const ccBtn   = document.getElementById('cc-btn');
  const ccPanel = document.getElementById('cc-panel');
  const ccPoint = document.getElementById('cc-point-opts');
  const ccClick = document.getElementById('cc-click-opts');
  let ccOpen = false;
  function setCCOpen(on) {
    ccOpen = on;
    ccPanel.classList.toggle('open', on);
    ccBtn.classList.toggle('on', on);
  }
  ccBtn.addEventListener('click', () => setCCOpen(!ccOpen));
  // Auto-open Control Center on page load so the user can see the
  // current state and tweak without hunting for a button.
  setCCOpen(true);

  function paintActive(container, value) {
    container.querySelectorAll('.cc-opt').forEach(b => {
      b.classList.toggle('on', b.dataset.v === value);
    });
  }
  function postMethod(payload) {
    fetch('/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(Object.assign({ action: 'cursor_method' }, payload)),
    }).catch(() => {});
  }
  const masterBtn = document.getElementById('master-btn');
  let masterOn = true;
  function paintMaster(on) {
    masterOn = !!on;
    masterBtn.textContent = on ? 'ON' : 'OFF';
    masterBtn.style.background = on ? 'var(--cool)' : '#2a1418';
    masterBtn.style.color = on ? '#05170f' : '#f5a3a3';
    masterBtn.style.borderColor = on ? 'var(--cool)' : '#6b2a35';
  }
  paintMaster(true);
  masterBtn.addEventListener('click', () => {
    const want = !masterOn;
    paintMaster(want);
    fetch('/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'master', on: want }),
    }).catch(() => {});
  });

  // Wispr quickstart presets: bundle dict_gesture + dict_mode + wispr_method
  // into a single one-click preset so the user doesn't have to reason about
  // the combinations.
  const ccWisprPreset = document.getElementById('cc-wispr-preset-opts');
  const WISPR_PRESETS = {
    menu_toggle:       { gesture: 'prayer', mode: 'latch', method: 'menu_click' },
    f19_double_toggle: { gesture: 'prayer', mode: 'latch', method: 'double_tap_f19' },
    f19_hold:          { gesture: 'prayer', mode: 'hold',  method: 'cgevent_f19' },
    fn_hold:           { gesture: 'prayer', mode: 'hold',  method: 'cgevent_fn',    extraTap: false },
    fn_tap_toggle:     { gesture: 'prayer', mode: 'latch', method: 'cgevent_fn' },
    fn_double_toggle: { gesture: 'prayer', mode: 'latch', method: 'double_tap_fn' },
    fn_hold_tap:       { gesture: 'prayer', mode: 'hold',  method: 'cgevent_fn',    extraTap: true  },
    fn_hold_nuclear:   { gesture: 'prayer', mode: 'hold',  method: 'cgevent_fn',    nuclear: true   },
    off:               { gesture: 'off',    mode: 'hold',  method: 'off' },
  };
  function paintWisprPreset(key) {
    if (!ccWisprPreset) return;
    ccWisprPreset.querySelectorAll('.cc-opt').forEach(b => {
      b.classList.toggle('on', b.dataset.preset === key);
    });
  }
  if (ccWisprPreset) {
    ccWisprPreset.addEventListener('click', async (e) => {
      const b = e.target.closest('.cc-opt'); if (!b) return;
      const p = WISPR_PRESETS[b.dataset.preset]; if (!p) return;
      paintWisprPreset(b.dataset.preset);
      // Fire sequentially so server state settles in a predictable order.
      try {
        await fetch('/command', { method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'dict_gesture', gesture: p.gesture }) });
        await fetch('/command', { method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'dict_mode', mode: p.mode }) });
        await fetch('/command', { method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'wispr_method', method: p.method }) });
        // Always explicitly set both release flags so switching presets
        // doesn't leave stale toggles from a previous preset.
        await fetch('/command', { method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'release_extra_tap', on: !!p.extraTap }) });
        await fetch('/command', { method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'release_nuclear', on: !!p.nuclear }) });
      } catch {}
    });
  }
  // Fn diagnostic lab — manual one-shot probes + release-extra-tap toggle.
  const ccFnDiag = document.getElementById('cc-fn-diag-opts');
  if (ccFnDiag) {
    ccFnDiag.addEventListener('click', (e) => {
      const b = e.target.closest('.cc-opt'); if (!b) return;
      if (b.dataset.toggle) {
        const turnOn = !b.classList.contains('on');
        b.classList.toggle('on', turnOn);
        fetch('/command', { method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: b.dataset.toggle, on: turnOn }),
        }).catch(() => {});
        return;
      }
      const sub = b.dataset.sub; if (!sub) return;
      // Flash the button briefly so the user knows it fired.
      b.classList.add('on');
      setTimeout(() => b.classList.remove('on'), 180);
      fetch('/command', { method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'fn_diag', sub }),
      }).catch(() => {});
    });
  }
  // Keep the preset row synced with whatever combo the server reports.
  function syncWisprPresetFromState(state) {
    if (!state) return;
    for (const [k, p] of Object.entries(WISPR_PRESETS)) {
      if (p.gesture === state.dictGesture
          && p.mode === state.dictMode
          && p.method === state.wispr
          && !!p.extraTap === !!state.releaseExtraTap
          && !!p.nuclear === !!state.releaseNuclear) {
        paintWisprPreset(k); return;
      }
    }
    paintWisprPreset(null);  // no matching preset → nothing highlighted
  }

  // Experimental toggles at top of CC panel.
  const ccExp = document.getElementById('cc-exp-opts');
  if (ccExp) {
    ccExp.addEventListener('click', (e) => {
      const b = e.target.closest('.cc-opt'); if (!b) return;
      // Manual atelier-mode force button (sibling in the exp row).
      if (b.id === 'cc-atelier-manual') {
        const turnOn = !b.classList.contains('on');
        b.classList.toggle('on', turnOn);
        b.textContent = 'atelier: ' + (turnOn ? 'ON' : 'off');
        fetch('/command', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'atelier_force', on: turnOn }),
        }).catch(() => {});
        return;
      }
      const key = b.dataset.exp; if (!key) return;
      const turnOn = !b.classList.contains('on');
      b.classList.toggle('on', turnOn);
      fetch('/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: key, on: turnOn }),
      }).catch(() => {});
    });
  }

  // ✨ Atelier sound: synthesized chord via Web Audio. No asset file needed.
  // Call atelierSound(true) on enter, atelierSound(false) on exit.
  let _audioCtx = null;
  function _ac() {
    if (!_audioCtx) {
      try { _audioCtx = new (window.AudioContext || window.webkitAudioContext)(); }
      catch (e) { return null; }
    }
    // Some browsers suspend until a user gesture — try to resume.
    if (_audioCtx.state === 'suspended') _audioCtx.resume().catch(() => {});
    return _audioCtx;
  }
  function atelierSound(on) {
    const ctx = _ac(); if (!ctx) return;
    const now = ctx.currentTime;
    // Arpeggiated chord: A minor → C major feel depending on direction.
    const notes = on
      ? [440, 659.25, 880, 1318.5]        // A4 E5 A5 E6 (rising, bright)
      : [880, 659.25, 523.25, 392.0];      // A5 E5 C5 G4 (falling, gentle)
    notes.forEach((f, i) => {
      const t = now + i * (on ? 0.06 : 0.08);
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = on ? 'triangle' : 'sine';
      osc.frequency.value = f;
      // Quick attack, gentle decay. Extra quiet overall.
      gain.gain.setValueAtTime(0.0, t);
      gain.gain.linearRampToValueAtTime(on ? 0.16 : 0.10, t + 0.01);
      gain.gain.exponentialRampToValueAtTime(0.001, t + (on ? 0.35 : 0.45));
      osc.connect(gain).connect(ctx.destination);
      osc.start(t);
      osc.stop(t + (on ? 0.4 : 0.5));
    });
    // Shimmer: a soft noise burst on enter for sparkle.
    if (on) {
      try {
        const buf = ctx.createBuffer(1, ctx.sampleRate * 0.25, ctx.sampleRate);
        const data = buf.getChannelData(0);
        for (let i = 0; i < data.length; i++) {
          data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / data.length, 3);
        }
        const src = ctx.createBufferSource();
        src.buffer = buf;
        const hp = ctx.createBiquadFilter();
        hp.type = 'highpass'; hp.frequency.value = 4000;
        const g = ctx.createGain(); g.gain.value = 0.06;
        src.connect(hp).connect(g).connect(ctx.destination);
        src.start(now);
      } catch (e) {}
    }
  }

  // ✨ Atelier toast — flashes when the mode toggles on or off.
  function atelierToast(on) {
    const t = document.createElement('div');
    t.textContent = on ? '✨ atelier mode' : 'atelier off';
    t.style.cssText = `
      position:fixed; top:50%; left:50%; transform:translate(-50%,-50%);
      z-index:99999; pointer-events:none;
      font-size:${on ? '42px' : '28px'}; font-weight:800;
      letter-spacing:0.08em; color:#05170f;
      background:linear-gradient(90deg,#f5c24a,#ff7ad9,#7aa7ff);
      padding:18px 36px; border-radius:18px;
      box-shadow:0 0 64px rgba(245,194,74,0.7), 0 0 32px rgba(122,167,255,0.5);
      opacity:0; transition:opacity 220ms ease, transform 380ms ease;
    `;
    document.body.appendChild(t);
    requestAnimationFrame(() => {
      t.style.opacity = '1';
      t.style.transform = 'translate(-50%,-50%) scale(1.05)';
    });
    setTimeout(() => {
      t.style.opacity = '0';
      setTimeout(() => t.remove(), 400);
    }, 900);
  }
  function atelierActionFlash(action) {
    // Little corner indicator for each atelier action (dev feedback).
    const pill = document.getElementById('atelier-pill');
    if (!pill) return;
    const prev = pill.textContent;
    pill.textContent = '✨ ' + action.replace('_', ' ');
    clearTimeout(pill._flashT);
    pill._flashT = setTimeout(() => { pill.textContent = prev; }, 300);
  }

  ccPoint.addEventListener('click', (e) => {
    const b = e.target.closest('.cc-opt'); if (!b) return;
    paintActive(ccPoint, b.dataset.v);
    postMethod({ pointing: b.dataset.v });
  });
  ccClick.addEventListener('click', (e) => {
    const b = e.target.closest('.cc-opt'); if (!b) return;
    paintActive(ccClick, b.dataset.v);
    postMethod({ click: b.dataset.v });
  });

  const ccWispr = document.getElementById('cc-wispr-opts');
  function postWispr(method) {
    fetch('/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'wispr_method', method }),
    }).catch(() => {});
  }
  function fireWispr() {
    fetch('/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'wispr_test' }),
    }).catch(() => {});
  }
  function holdTestWispr() {
    fetch('/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'wispr_hold_test' }),
    }).catch(() => {});
  }
  ccWispr.addEventListener('click', (e) => {
    const b = e.target.closest('.cc-opt'); if (!b) return;
    if (b.id === 'cc-wispr-fire') { fireWispr(); return; }
    if (b.id === 'cc-wispr-holdtest') { holdTestWispr(); return; }
    ccWispr.querySelectorAll('.cc-opt').forEach(x => {
      if (!x.id)
        x.classList.toggle('on', x.dataset.v === b.dataset.v);
    });
    postWispr(b.dataset.v);
  });

  const ccDict = document.getElementById('cc-dict-opts');
  ccDict.addEventListener('click', (e) => {
    const b = e.target.closest('.cc-opt'); if (!b) return;
    paintActive(ccDict, b.dataset.v);
    fetch('/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'dict_gesture', gesture: b.dataset.v }),
    }).catch(() => {});
  });

  const ccDictMode = document.getElementById('cc-dictmode-opts');
  ccDictMode.addEventListener('click', (e) => {
    const b = e.target.closest('.cc-opt'); if (!b) return;
    paintActive(ccDictMode, b.dataset.v);
    fetch('/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'dict_mode', mode: b.dataset.v }),
    }).catch(() => {});
  });

  const ccClap = document.getElementById('cc-clap-opts');
  ccClap.addEventListener('click', (e) => {
    const b = e.target.closest('.cc-opt'); if (!b) return;
    paintActive(ccClap, b.dataset.v);
    fetch('/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'clap_preset', preset: b.dataset.v }),
    }).catch(() => {});
  });

  const ccRClick = document.getElementById('cc-rclick-opts');
  ccRClick.addEventListener('click', (e) => {
    const b = e.target.closest('.cc-opt'); if (!b) return;
    paintActive(ccRClick, b.dataset.v);
    fetch('/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'right_click_method', method: b.dataset.v }),
    }).catch(() => {});
  });

  const ccDClick = document.getElementById('cc-dclick-opts');
  ccDClick.addEventListener('click', (e) => {
    const b = e.target.closest('.cc-opt'); if (!b) return;
    paintActive(ccDClick, b.dataset.v);
    fetch('/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'double_click', on: b.dataset.v === 'on' }),
    }).catch(() => {});
  });

  const ccScrollMode = document.getElementById('cc-scroll-mode-opts');
  ccScrollMode.addEventListener('click', (e) => {
    const b = e.target.closest('.cc-opt'); if (!b) return;
    paintActive(ccScrollMode, b.dataset.v);
    fetch('/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'scroll_mode', mode: b.dataset.v }),
    }).catch(() => {});
  });

  const ccScrollSpeed = document.getElementById('cc-scroll-speed');
  const ccScrollSpeedVal = document.getElementById('cc-scroll-speed-val');
  function paintScrollSpeed(v) {
    const n = Number(v);
    if (document.activeElement !== ccScrollSpeed) ccScrollSpeed.value = n;
    ccScrollSpeedVal.textContent = n.toFixed(1) + '×';
  }
  ccScrollSpeed.addEventListener('input', () => {
    paintScrollSpeed(ccScrollSpeed.value);
    fetch('/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'scroll_speed',
                             value: parseFloat(ccScrollSpeed.value) }),
    }).catch(() => {});
  });

  const ccTabSwipe = document.getElementById('cc-tabswipe-opts');
  ccTabSwipe.addEventListener('click', (e) => {
    const b = e.target.closest('.cc-opt'); if (!b) return;
    paintActive(ccTabSwipe, b.dataset.v);
    fetch('/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'tab_swipe', on: b.dataset.v }),
    }).catch(() => {});
  });

  // Voice daemon — direct-mic always-on command listening (Whisper).
  const ccVoiceDaemon = document.getElementById('cc-voicedaemon-opts');
  const ccVoiceStatus = document.getElementById('cc-voice-status');
  ccVoiceDaemon.addEventListener('click', (e) => {
    const b = e.target.closest('.cc-opt'); if (!b) return;
    paintActive(ccVoiceDaemon, b.dataset.v);
    fetch('/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'voice_daemon', on: b.dataset.v === 'on' }),
    }).catch(() => {});
  });

  const ccSwipe = document.getElementById('cc-swipe-opts');
  ccSwipe.addEventListener('click', (e) => {
    const b = e.target.closest('.cc-opt'); if (!b) return;
    paintActive(ccSwipe, b.dataset.v);
    fetch('/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'swipe_gesture', on: b.dataset.v === 'on' }),
    }).catch(() => {});
  });

  const ccSens = document.getElementById('cc-sens');
  const ccSensVal = document.getElementById('cc-sens-val');
  function paintSens(v) {
    const n = Number(v);
    if (document.activeElement !== ccSens) ccSens.value = n;
    ccSensVal.textContent = n.toFixed(2) + '×';
  }
  ccSens.addEventListener('input', () => {
    paintSens(ccSens.value);
    fetch('/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'cursor_sens', value: parseFloat(ccSens.value) }),
    }).catch(() => {});
  });

  let clapCount = 0, clapDecayT = null;
  let ctx = null, drums = null, pad = null;
  let leftTh = null, rightTh = null;
  let jamBass = null, jamLead = null;
  let jamMode = false;
  let audioReady = false;
  let masterGain = null;

  // DJ Board state — which channels are live when jam mode is on
  const djEnabled = { left: true, right: true, smile: true, head: true };
  const djPreset  = { left: 'warm', right: 'dreamy', smile: 'warm', head: 'house' };
  // BPM tracker for head bob
  const bobTimes = [];
  let bobBPM = 0;

  function noiseBuffer(ctx, seconds = 2) {
    const buf = ctx.createBuffer(1, seconds * ctx.sampleRate, ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    return buf;
  }

  // BobDrums: one kick/snare (alternating) + hat on each call to hit().
  // No scheduler — the rhythm *is* your head bobbing.
  class BobDrums {
    constructor(ctx, out) {
      this.ctx = ctx;
      this.count = 0;
      this.enabled = true;
      this.preset = 'house';

      this.filter = ctx.createBiquadFilter();
      this.filter.type = 'lowpass';
      this.filter.frequency.value = 2800;
      this.filter.Q.value = 0.4;

      this.master = ctx.createGain();
      this.master.gain.value = 0.8;
      this.filter.connect(this.master).connect(out || ctx.destination);

      this.noiseBuf = noiseBuffer(ctx, 1);
    }

    setPreset(name) {
      this.preset = name;
      // Tweak the shared filter to reshape the kit.
      if (name === 'tight') { this.filter.frequency.value = 4200; }
      else if (name === '808') { this.filter.frequency.value = 1600; }
      else { this.filter.frequency.value = 2800; } // house (default)
    }

    hit() {
      const t = this.ctx.currentTime;
      this.count++;
      if (this.count % 2 === 1) this._kick(t);
      else this._snare(t);
      this._hat(t, 0.4);
    }

    _kick(t) {
      const osc = this.ctx.createOscillator();
      const env = this.ctx.createGain();
      // Preset reshapes the kick envelope + pitch sweep.
      let startF = 140, endF = 40, decay = 0.28, stop = 0.32;
      if (this.preset === 'tight') {
        startF = 170; endF = 60; decay = 0.15; stop = 0.18;
      } else if (this.preset === '808') {
        startF = 110; endF = 32; decay = 0.55; stop = 0.65;
      }
      osc.frequency.setValueAtTime(startF, t);
      osc.frequency.exponentialRampToValueAtTime(endF, t + 0.12);
      env.gain.setValueAtTime(0.0001, t);
      env.gain.exponentialRampToValueAtTime(0.9, t + 0.003);
      env.gain.exponentialRampToValueAtTime(0.001, t + decay);
      osc.connect(env).connect(this.filter);
      osc.start(t); osc.stop(t + stop);
    }

    _snare(t) {
      const n = this.ctx.createBufferSource();
      n.buffer = this.noiseBuf;
      const bp = this.ctx.createBiquadFilter();
      bp.type = 'bandpass'; bp.frequency.value = 1700; bp.Q.value = 1.1;
      const env = this.ctx.createGain();
      env.gain.setValueAtTime(0.0001, t);
      env.gain.exponentialRampToValueAtTime(0.55, t + 0.002);
      env.gain.exponentialRampToValueAtTime(0.001, t + 0.14);
      n.connect(bp).connect(env).connect(this.filter);
      n.start(t); n.stop(t + 0.18);
    }

    _hat(t, vel) {
      const n = this.ctx.createBufferSource();
      n.buffer = this.noiseBuf;
      const hp = this.ctx.createBiquadFilter();
      hp.type = 'highpass'; hp.frequency.value = 6500;
      const env = this.ctx.createGain();
      env.gain.setValueAtTime(0.0001, t);
      env.gain.exponentialRampToValueAtTime(vel * 0.35, t + 0.001);
      env.gain.exponentialRampToValueAtTime(0.001, t + 0.04);
      n.connect(hp).connect(env).connect(this.filter);
      n.start(t); n.stop(t + 0.06);
    }

    // Rim tick — short resonant click, played on blink.
    rim() {
      const t = this.ctx.currentTime;
      const n = this.ctx.createBufferSource();
      n.buffer = this.noiseBuf;
      const bp = this.ctx.createBiquadFilter();
      bp.type = 'bandpass'; bp.frequency.value = 3800; bp.Q.value = 3.0;
      const env = this.ctx.createGain();
      env.gain.setValueAtTime(0.0001, t);
      env.gain.exponentialRampToValueAtTime(0.32, t + 0.001);
      env.gain.exponentialRampToValueAtTime(0.001, t + 0.06);
      n.connect(bp).connect(env).connect(this.filter);
      n.start(t); n.stop(t + 0.08);
    }

    // Gentle chime for acknowledgements (prayer, mode switch).
    chime() {
      const t = this.ctx.currentTime;
      [880, 1318.51].forEach((f, i) => {
        const o = this.ctx.createOscillator();
        o.type = 'sine';
        o.frequency.value = f;
        const env = this.ctx.createGain();
        env.gain.setValueAtTime(0.0001, t);
        env.gain.exponentialRampToValueAtTime(0.12, t + 0.01 + i * 0.03);
        env.gain.exponentialRampToValueAtTime(0.001, t + 0.6);
        o.connect(env).connect(this.master);
        o.start(t); o.stop(t + 0.65);
      });
    }
  }

  // Theremin: continuous pitched tone gated on hand being raised.
  class Theremin {
    constructor(ctx, notes, opts, out) {
      opts = opts || {};
      this.ctx = ctx;
      this.notes = notes;
      this.gainLevel = opts.gainLevel != null ? opts.gainLevel : 0.11;
      this.oscType  = opts.oscType  || 'sine';

      this.lpf = ctx.createBiquadFilter();
      this.lpf.type = 'lowpass';
      this.lpf.frequency.value = opts.lpfHz != null ? opts.lpfHz : 2600;

      this.gain = ctx.createGain(); this.gain.gain.value = 0;
      this.lpf.connect(this.gain).connect(out || ctx.destination);

      this.osc = ctx.createOscillator();
      this.osc.type = this.oscType;
      this.osc.frequency.value = notes[0];
      this.osc.connect(this.lpf);
      this.osc.start();

      // octave-up shimmer (skip if octaveMix = 0)
      const mix = opts.octaveMix != null ? opts.octaveMix : 0.18;
      if (mix > 0) {
        this.osc2 = ctx.createOscillator();
        this.osc2.type = 'triangle';
        this.osc2.frequency.value = notes[0] * 2;
        const g2 = ctx.createGain(); g2.gain.value = mix;
        this.osc2.connect(g2).connect(this.lpf);
        this.osc2.start();
      }
    }

    setHandY(y /* null or number in [0, HAND_THRESHOLD=0.55] */) {
      const now = this.ctx.currentTime;
      if (y == null || y < 0) {
        this.gain.gain.setTargetAtTime(0, now, 0.14);
        return;
      }
      const t = 1.0 - Math.max(0, Math.min(1, y / 0.55));
      const idx = Math.round(t * (this.notes.length - 1));
      const freq = this.notes[idx];
      this.osc.frequency.setTargetAtTime(freq, now, 0.05);
      if (this.osc2) this.osc2.frequency.setTargetAtTime(freq * 2, now, 0.05);
      this.gain.gain.setTargetAtTime(this.gainLevel, now, 0.08);
    }
  }

  // C major pentatonic in two ranges.
  const BASS_NOTES = [
    65.41, 73.42, 82.41, 98.00, 110.00,       // C2 D2 E2 G2 A2
    130.81, 146.83, 164.81, 196.00, 220.00,   // C3 D3 E3 G3 A3
  ];
  const LEAD_NOTES = [
    523.25, 587.33, 659.25, 783.99, 880.00,        // C5 D5 E5 G5 A5
    1046.50, 1174.66, 1318.51, 1567.98, 1760.00,   // C6 D6 E6 G6 A6
  ];

  class WarmPad {
    constructor(ctx, out) {
      this.ctx = ctx;
      this.enabled = true;
      this.preset = 'warm';
      // C major 7 (ish): C3, E3, G3, B3 — spacious, Rhodes-y
      const freqs = [130.81, 164.81, 196.00, 246.94];
      this.lpf = ctx.createBiquadFilter();
      this.lpf.type = 'lowpass'; this.lpf.frequency.value = 1600;
      this.lpf.Q.value = 0.6;

      const reverbDelay = ctx.createDelay();
      reverbDelay.delayTime.value = 0.11;
      const feedback = ctx.createGain(); feedback.gain.value = 0.32;
      reverbDelay.connect(feedback).connect(reverbDelay);

      this.gain = ctx.createGain();
      this.gain.gain.value = 0;

      this.lpf.connect(this.gain);
      this.lpf.connect(reverbDelay);
      reverbDelay.connect(this.gain);
      this.gain.connect(out || ctx.destination);

      this.oscs = [];
      // Warm preset = full C-maj7; drone preset uses only the root so we can
      // mute/unmute per preset via per-osc gain taps.
      this.oscGains = [];
      freqs.forEach((f, fi) => {
        for (const d of [-6, 6]) {
          const o = ctx.createOscillator();
          o.type = 'triangle';
          o.frequency.value = f;
          o.detune.value = d;
          const og = ctx.createGain(); og.gain.value = 1;
          o.connect(og).connect(this.lpf);
          o.start();
          this.oscs.push({ osc: o, gain: og, rootIdx: fi });
        }
      });
    }
    setAmount(v) {
      // v in [0,1]; target peak gain ~0.22
      const g = Math.max(0, Math.min(1, v)) * (this.preset === 'drone' ? 0.3 : 0.22);
      this.gain.gain.setTargetAtTime(g, this.ctx.currentTime, 0.18);
    }
    setPreset(name) {
      this.preset = name;
      const now = this.ctx.currentTime;
      if (name === 'bright') {
        this.lpf.frequency.setTargetAtTime(3200, now, 0.2);
        this.lpf.Q.setTargetAtTime(1.2, now, 0.2);
        this.oscs.forEach(o => o.gain.gain.setTargetAtTime(1, now, 0.2));
      } else if (name === 'drone') {
        // Only the root C3 pair plays; the rest mute.
        this.lpf.frequency.setTargetAtTime(900, now, 0.2);
        this.lpf.Q.setTargetAtTime(0.8, now, 0.2);
        this.oscs.forEach(o => {
          o.gain.gain.setTargetAtTime(o.rootIdx === 0 ? 1 : 0, now, 0.2);
        });
      } else { // warm (default)
        this.lpf.frequency.setTargetAtTime(1600, now, 0.2);
        this.lpf.Q.setTargetAtTime(0.6, now, 0.2);
        this.oscs.forEach(o => o.gain.gain.setTargetAtTime(1, now, 0.2));
      }
    }
  }

  // A minor pentatonic scales (darker, Tame Impala-ish).
  const JAM_BASS_NOTES = [
    55.00, 65.41, 73.42, 82.41, 98.00,         // A1 C2 D2 E2 G2
    110.00, 130.81, 146.83, 164.81, 196.00,    // A2 C3 D3 E3 G3
  ];
  const JAM_LEAD_NOTES = [
    220.00, 261.63, 293.66, 329.63, 392.00,        // A3 C4 D4 E4 G4
    440.00, 523.25, 587.33, 659.25, 783.99,        // A4 C5 D5 E5 G5
    880.00,                                         // A5
  ];

  // Left hand = warm sawtooth bass through a resonant lowpass.
  // Y → pitch (up = higher). X → filter cutoff (left = muted, right = open).
  class JamBass {
    constructor(ctx, out) {
      this.ctx = ctx;
      this.notes = JAM_BASS_NOTES;
      this.enabled = true;
      this.preset = 'warm';
      this.baseLpfQ = 4.5;
      this.baseLpfFloor = 250;
      this.baseLpfCeil = 2500;

      this.lpf = ctx.createBiquadFilter();
      this.lpf.type = 'lowpass';
      this.lpf.frequency.value = 500;
      this.lpf.Q.value = 4.5;

      this.drive = ctx.createWaveShaper();
      const curve = new Float32Array(512);
      for (let i = 0; i < 512; i++) {
        const x = i / 256 - 1;
        curve[i] = Math.tanh(x * 2.2);
      }
      this.drive.curve = curve;

      this.gain = ctx.createGain(); this.gain.gain.value = 0;

      this.osc = ctx.createOscillator();
      this.osc.type = 'sawtooth';
      this.osc.frequency.value = this.notes[0];

      // Sub osc an octave down for heft.
      this.sub = ctx.createOscillator();
      this.sub.type = 'sine';
      this.sub.frequency.value = this.notes[0] / 2;
      const subGain = ctx.createGain(); subGain.gain.value = 0.35;

      this.osc.connect(this.lpf);
      this.sub.connect(subGain).connect(this.lpf);
      this.lpf.connect(this.drive).connect(this.gain).connect(out || ctx.destination);
      this.osc.start(); this.sub.start();
    }
    setXY(x, y) {
      const now = this.ctx.currentTime;
      if (x == null || y == null) {
        this.gain.gain.setTargetAtTime(0, now, 0.18);
        return;
      }
      const t = 1.0 - Math.max(0, Math.min(1, y / 0.55));
      const idx = Math.round(t * (this.notes.length - 1));
      const f = this.notes[idx];
      this.osc.frequency.setTargetAtTime(f, now, 0.06);
      this.sub.frequency.setTargetAtTime(f / 2, now, 0.06);
      const xc = Math.max(0, Math.min(1, x));
      // baseLpfFloor + (ceil - floor) via exp mapping
      const cutoff = this.baseLpfFloor *
        Math.pow(this.baseLpfCeil / this.baseLpfFloor, xc);
      this.lpf.frequency.setTargetAtTime(cutoff, now, 0.06);
      this.lpf.Q.setTargetAtTime(this.baseLpfQ, now, 0.06);
      this.gain.gain.setTargetAtTime(0.18, now, 0.12);
    }
    setPreset(name) {
      this.preset = name;
      if (name === 'sub') {
        this.baseLpfQ = 1.2; this.baseLpfFloor = 80; this.baseLpfCeil = 600;
      } else if (name === 'acid') {
        this.baseLpfQ = 12; this.baseLpfFloor = 180; this.baseLpfCeil = 4500;
      } else { // warm (default)
        this.baseLpfQ = 4.5; this.baseLpfFloor = 250; this.baseLpfCeil = 2500;
      }
    }
  }

  // Right hand = phased sawtooth pad through 4-stage allpass phaser + delay.
  // Y → note in A-minor pent. X → phaser LFO rate + wet mix.
  class JamLead {
    constructor(ctx, out) {
      this.ctx = ctx;
      this.notes = JAM_LEAD_NOTES;
      this.enabled = true;
      this.preset = 'dreamy';

      // source: sawtooth + detuned copy for thickness
      this.osc = ctx.createOscillator();
      this.osc.type = 'sawtooth';
      this.osc.frequency.value = this.notes[0];

      this.osc2 = ctx.createOscillator();
      this.osc2.type = 'sawtooth';
      this.osc2.detune.value = 8;
      this.osc2.frequency.value = this.notes[0];

      const mixer = ctx.createGain(); mixer.gain.value = 0.5;
      this.osc.connect(mixer);
      this.osc2.connect(mixer);

      // gentle lowpass to tame highs
      const lpf = ctx.createBiquadFilter();
      lpf.type = 'lowpass'; lpf.frequency.value = 2800; lpf.Q.value = 0.5;
      mixer.connect(lpf);

      // 4-stage allpass phaser
      this.apf = [];
      const lfo = ctx.createOscillator();
      lfo.type = 'sine';
      lfo.frequency.value = 0.35;
      const lfoGain = ctx.createGain(); lfoGain.gain.value = 500;
      lfo.connect(lfoGain);
      let node = lpf;
      for (let i = 0; i < 4; i++) {
        const a = ctx.createBiquadFilter();
        a.type = 'allpass';
        a.frequency.value = 400 + i * 350;
        a.Q.value = 6;
        lfoGain.connect(a.frequency);
        node.connect(a);
        this.apf.push(a);
        node = a;
      }
      lfo.start();
      this.lfo = lfo;
      this.lfoGain = lfoGain;

      // dry + wet
      this.dryGain = ctx.createGain(); this.dryGain.gain.value = 0.6;
      this.wetGain = ctx.createGain(); this.wetGain.gain.value = 0.4;
      lpf.connect(this.dryGain);
      node.connect(this.wetGain);

      // delay reverb for depth
      const delay = ctx.createDelay();
      delay.delayTime.value = 0.22;
      const fb = ctx.createGain(); fb.gain.value = 0.38;
      delay.connect(fb).connect(delay);
      const wetRev = ctx.createGain(); wetRev.gain.value = 0.35;

      this.gain = ctx.createGain(); this.gain.gain.value = 0;
      this.dryGain.connect(this.gain);
      this.wetGain.connect(this.gain);
      this.wetGain.connect(delay);
      delay.connect(wetRev).connect(this.gain);
      this.gain.connect(out || ctx.destination);

      this.osc.start(); this.osc2.start();
      this._prevY = null;  // for pluck/stab edge detection
    }
    setXY(x, y) {
      const now = this.ctx.currentTime;
      if (x == null || y == null) {
        this.gain.gain.setTargetAtTime(0, now, 0.2);
        this._prevY = null;
        return;
      }
      const t = 1.0 - Math.max(0, Math.min(1, y / 0.55));
      const idx = Math.round(t * (this.notes.length - 1));
      const f = this.notes[idx];
      this.osc.frequency.setTargetAtTime(f, now, 0.05);
      this.osc2.frequency.setTargetAtTime(f, now, 0.05);
      const xc = Math.max(0, Math.min(1, x));
      const rate = 0.15 + xc * 3.5;
      this.lfo.frequency.setTargetAtTime(rate, now, 0.1);
      this.wetGain.gain.setTargetAtTime(0.2 + xc * 0.7, now, 0.1);
      this.dryGain.gain.setTargetAtTime(0.7 - xc * 0.4, now, 0.1);

      if (this.preset === 'pluck') {
        // Envelope retriggers when hand "jumps" to a new note (idx change).
        // Keeps a short decay so it reads as a pluck.
        const noteChanged = (this._prevIdx !== undefined && this._prevIdx !== idx);
        if (this._prevY == null || noteChanged) {
          this.gain.gain.cancelScheduledValues(now);
          this.gain.gain.setValueAtTime(0.001, now);
          this.gain.gain.exponentialRampToValueAtTime(0.22, now + 0.006);
          this.gain.gain.exponentialRampToValueAtTime(0.02, now + 0.35);
        }
        this._prevIdx = idx;
      } else if (this.preset === 'stab') {
        // Fire a quick chord when first raised; hold while held.
        if (this._prevY == null) {
          this.gain.gain.cancelScheduledValues(now);
          this.gain.gain.setValueAtTime(0.001, now);
          this.gain.gain.exponentialRampToValueAtTime(0.18, now + 0.01);
          this.gain.gain.exponentialRampToValueAtTime(0.07, now + 0.25);
        } else {
          this.gain.gain.setTargetAtTime(0.07, now, 0.15);
        }
      } else {
        // dreamy (default): continuous sustained tone
        this.gain.gain.setTargetAtTime(0.13, now, 0.12);
      }
      this._prevY = y;
    }
    setPreset(name) {
      this.preset = name;
      this._prevY = null;
      this._prevIdx = undefined;
    }
  }

  function unlockAudio() {
    if (audioReady) return;
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    masterGain = ctx.createGain();
    masterGain.gain.value = 0.9;
    masterGain.connect(ctx.destination);
    drums = new BobDrums(ctx, masterGain);
    pad = new WarmPad(ctx, masterGain);
    leftTh  = new Theremin(ctx, BASS_NOTES, {
      gainLevel: 0.14, oscType: 'triangle', octaveMix: 0.10, lpfHz: 1800,
    }, masterGain);
    rightTh = new Theremin(ctx, LEAD_NOTES, {
      gainLevel: 0.10, oscType: 'sine',     octaveMix: 0.20, lpfHz: 3200,
    }, masterGain);
    jamBass = new JamBass(ctx, masterGain);
    jamLead = new JamLead(ctx, masterGain);
    audioReady = true;
    statusEl.textContent = 'audio ready';
    statusEl.className = 'status on';
  }

  // Cheerful ascending chime — plays on double-clap boot.
  function bootChime() {
    if (!ctx) return;
    const t = ctx.currentTime;
    const notes = [523.25, 659.25, 783.99, 1046.50]; // C5 E5 G5 C6
    notes.forEach((f, i) => {
      const o = ctx.createOscillator();
      o.type = 'triangle';
      o.frequency.value = f;
      const g = ctx.createGain();
      const start = t + i * 0.08;
      g.gain.setValueAtTime(0.0001, start);
      g.gain.exponentialRampToValueAtTime(0.18, start + 0.01);
      g.gain.exponentialRampToValueAtTime(0.001, start + 0.5);
      o.connect(g).connect(ctx.destination);
      o.start(start); o.stop(start + 0.55);
    });
  }

  function flashBob() {
    flashEl.classList.add('on');
    setTimeout(() => flashEl.classList.remove('on'), 160);
  }

  function handBarPct(y) {
    if (y == null) return 0;
    return Math.round(Math.max(0, Math.min(1, 1 - y / 0.55)) * 100);
  }

  function connectEvents() {
    const es = new EventSource('/events');
    es.onmessage = (e) => {
      let msg; try { msg = JSON.parse(e.data); } catch { return; }

      if (msg.swipe) flashBob();

      if (typeof msg.smile === 'number') {
        smileFill.style.width = Math.round(msg.smile * 100) + '%';
        djSetMeter('smile', msg.smile);
        if (pad && (!jamMode || djEnabled.smile)) pad.setAmount(msg.smile);
        else if (pad && jamMode && !djEnabled.smile) pad.setAmount(0);
      }
      if ('leftHandY' in msg) {
        leftFill.style.width = handBarPct(msg.leftHandY) + '%';
        djSetMeter('left', Math.max(0, Math.min(1, 1 - msg.leftHandY / 0.55)));
        if (jamMode) {
          if (jamBass && djEnabled.left) jamBass.setXY(msg.leftHandX, msg.leftHandY);
          else if (jamBass) jamBass.setXY(null, null);
          if (leftTh) leftTh.setHandY(null);
        } else {
          if (leftTh) leftTh.setHandY(msg.leftHandY);
          if (jamBass) jamBass.setXY(null, null);
        }
      }
      if ('rightHandY' in msg) {
        rightFill.style.width = handBarPct(msg.rightHandY) + '%';
        djSetMeter('right', Math.max(0, Math.min(1, 1 - msg.rightHandY / 0.55)));
        if (jamMode) {
          if (jamLead && djEnabled.right) jamLead.setXY(msg.rightHandX, msg.rightHandY);
          else if (jamLead) jamLead.setXY(null, null);
          if (rightTh) rightTh.setHandY(null);
        } else {
          if (rightTh) rightTh.setHandY(msg.rightHandY);
          if (jamLead) jamLead.setXY(null, null);
        }
      }
      if (msg.bob) {
        if (drums && (!jamMode || djEnabled.head)) drums.hit();
        flashBob();
        djTrackBPM();
        djFlashMeter('head');
      }
      if (msg.blink && drums) drums.rim();

      if (msg.prayerStart) {
        listenPill.style.display = 'inline-block';
        notesEl.focus();
        if (drums) drums.chime();
        flashBob();
      }
      if (msg.prayerEnd) {
        listenPill.style.display = 'none';
      }
      if (msg.boot) {
        // Double-clap: toggle voice listening.
        toggleVoice();
        flashBob();
      }
      if ('pointing' in msg) paintActive(ccPoint, msg.pointing);
      if ('click'    in msg) paintActive(ccClick, msg.click);
      if ('wispr'    in msg) {
        ccWispr.querySelectorAll('.cc-opt').forEach(x => {
          if (x.id !== 'cc-wispr-fire')
            x.classList.toggle('on', x.dataset.v === msg.wispr);
        });
      }
      if ('scrollMode' in msg) paintActive(ccScrollMode, msg.scrollMode);
      if ('scrollSpeed' in msg) paintScrollSpeed(msg.scrollSpeed);
      if ('swipeGesture' in msg) paintActive(ccSwipe, msg.swipeGesture ? 'on' : 'off');
      if ('tabSwipe' in msg || 'tabSwipeAction' in msg) {
        const v = msg.tabSwipe ? (msg.tabSwipeAction || 'tabs') : 'off';
        paintActive(ccTabSwipe, v);
      }
      if ('voiceDaemon' in msg) {
        paintActive(ccVoiceDaemon, msg.voiceDaemon ? 'on' : 'off');
        const logEl = document.getElementById('cc-voice-log');
        if (logEl) logEl.style.display = msg.voiceDaemon ? 'block' : 'none';
      }
      if ('voiceState' in msg) {
        let s = msg.voiceState;
        if (msg.voiceErr) s = 'err: ' + msg.voiceErr;
        ccVoiceStatus.textContent = s;
      }
      if ('voiceTranscript' in msg) {
        const logEl = document.getElementById('cc-voice-log');
        if (logEl && Array.isArray(msg.voiceTranscript)) {
          logEl.innerHTML = msg.voiceTranscript.map(u => {
            const ok = u.result && u.result !== '(no match)';
            const color = ok ? '#6ee7b7' : 'var(--dim)';
            return '<div style="color:' + color + '">' +
                   '“' + (u.text || '').replace(/[<>&]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;'}[c])) + '” ' +
                   '<span style="color:var(--dim)">→ ' + (u.result || '') + '</span>' +
                   '</div>';
          }).reverse().join('');
        }
      }
      if ('cursorSens' in msg) paintSens(msg.cursorSens);
      if ('systemEnabled' in msg) {
        document.body.classList.toggle('system-off', !msg.systemEnabled);
        paintMaster(msg.systemEnabled);
        const ms = document.getElementById('cc-master-state');
        if (ms) {
          ms.textContent = msg.systemEnabled ? 'on' : 'off';
          ms.style.color = msg.systemEnabled
            ? 'var(--cool)' : 'var(--warm)';
        }
      }
      if ('tTimeout' in msg) {
        const b = document.querySelector('#cc-exp-opts [data-exp="t_timeout"]');
        if (b) b.classList.toggle('on', !!msg.tTimeout);
      }
      if ('mouthHold' in msg) {
        const b = document.querySelector('#cc-exp-opts [data-exp="mouth_hold"]');
        if (b) b.classList.toggle('on', !!msg.mouthHold);
      }
      if ('peaceRclick' in msg) {
        const b = document.querySelector('#cc-exp-opts [data-exp="peace_rclick"]');
        if (b) b.classList.toggle('on', !!msg.peaceRclick);
      }
      if ('thumbsDclick' in msg) {
        const b = document.querySelector('#cc-exp-opts [data-exp="thumbs_dclick"]');
        if (b) b.classList.toggle('on', !!msg.thumbsDclick);
      }
      if ('headCopy' in msg) {
        const b = document.querySelector('#cc-exp-opts [data-exp="head_copy"]');
        if (b) b.classList.toggle('on', !!msg.headCopy);
      }
      if ('fistZoom' in msg) {
        const b = document.querySelector('#cc-exp-opts [data-exp="fist_zoom"]');
        if (b) b.classList.toggle('on', !!msg.fistZoom);
      }
      if ('atelierEnabled' in msg) {
        const b = document.querySelector('#cc-exp-opts [data-exp="atelier"]');
        if (b) b.classList.toggle('on', !!msg.atelierEnabled);
      }
      if ('atelierMode' in msg) {
        const pill = document.getElementById('atelier-pill');
        if (pill) pill.style.display = msg.atelierMode ? 'inline-block' : 'none';
        const mb = document.getElementById('cc-atelier-manual');
        if (mb) {
          mb.classList.toggle('on', !!msg.atelierMode);
          mb.textContent = 'atelier: ' + (msg.atelierMode ? 'ON' : 'off');
        }
        document.body.classList.toggle('atelier-on', !!msg.atelierMode);
      }
      if (msg.atelierToggled) {
        const pill = document.getElementById('atelier-pill');
        const on = pill && pill.style.display !== 'none';
        atelierToast(on);
        atelierSound(on);
      }
      if (msg.atelierAction) atelierActionFlash(msg.atelierAction);
      if ('releaseExtraTap' in msg) {
        const b = document.getElementById('cc-fn-release-tap');
        if (b) b.classList.toggle('on', !!msg.releaseExtraTap);
      }
      if ('releaseNuclear' in msg) {
        const b = document.getElementById('cc-fn-release-nuclear');
        if (b) b.classList.toggle('on', !!msg.releaseNuclear);
      }
      if (msg.clapTick) {
        clapCount += 1;
        const ci = document.getElementById('cc-clap-indicator');
        if (ci) {
          ci.textContent = 'clap! (' + clapCount + ') double clap to toggle';
          ci.style.color = 'var(--cool)';
          clearTimeout(clapDecayT);
          clapDecayT = setTimeout(() => {
            ci.style.color = 'var(--dim)';
          }, 800);
        }
        document.body.classList.add('clap-pulse');
        setTimeout(() => document.body.classList.remove('clap-pulse'), 180);
      }
      if ('rightClick' in msg) paintActive(ccRClick, msg.rightClick);
      if ('doubleClick' in msg) paintActive(ccDClick, msg.doubleClick ? 'on' : 'off');
      if (msg.dictation) {
        const notesEl = document.getElementById('notes');
        if (notesEl) {
          // Expand the details section if collapsed, then focus.
          const parent = notesEl.closest('details.sec');
          if (parent && !parent.open) parent.open = true;
          try { notesEl.focus(); } catch (e) {}
        }
      }
      if ('cursor' in msg) {
        if (msg.cursor) {
          cursorPill.style.display = 'inline-block';
          const suffix = msg.pointing && msg.click
            ? ' · ' + msg.pointing + '/' + msg.click : '';
          cursorPill.textContent = (msg.cursorCalibrated
            ? 'cursor live' : 'cursor calibrating…') + suffix;
        } else {
          cursorPill.style.display = 'none';
        }
      }
      if ('clapPreset' in msg) paintActive(ccClap, msg.clapPreset);
      if ('dictGesture' in msg) paintActive(ccDict, msg.dictGesture);
      if ('dictMode' in msg) paintActive(ccDictMode, msg.dictMode);
      if ('dictGesture' in msg || 'dictMode' in msg || 'wispr' in msg) {
        syncWisprPresetFromState(msg);
      }
    };
  }

  // Auto-unlock audio on first page interaction (needed for jam mode
  // synths). Voice recognition is NOT auto-started — mic stays cold until
  // you click the voice pill or open the voice test panel. Keeps CPU/
  // battery low when you're not actively voice-testing.
  function firstInteraction() {
    unlockAudio();
    window.removeEventListener('pointerdown', firstInteraction);
    window.removeEventListener('keydown', firstInteraction);
  }
  window.addEventListener('pointerdown', firstInteraction);
  window.addEventListener('keydown', firstInteraction);

  connectEvents();

  // Animated background — switchable between several modes via Control Center.
  let __bgMode = localStorage.getItem('bgMode') || 'stars';
  (function bg() {
    const canvas = document.getElementById('stars');
    const ctx = canvas.getContext('2d');
    let w = 0, h = 0, dpr = window.devicePixelRatio || 1;
    function resize() {
      w = window.innerWidth; h = window.innerHeight;
      canvas.width  = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width  = w + 'px';
      canvas.style.height = h + 'px';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    window.addEventListener('resize', resize);
    resize();

    // --- stars ---
    const shooters = [];
    function spawnShooter() {
      const fromLeft = Math.random() < 0.5;
      shooters.push({
        x: fromLeft ? -40 : w + 40,
        y: Math.random() * h * 0.7,
        vx: (fromLeft ? 1 : -1) * (4 + Math.random() * 4),
        vy: 1.2 + Math.random() * 1.8,
        life: 1.0,
        hue: 180 + Math.random() * 120,
      });
    }
    function drawStars(jam) {
      const baseAlpha = jam ? 0.5 : 0.28;
      for (let i = 0; i < 60; i++) {
        const sx = (i * 97 + (performance.now() * 0.02)) % w;
        const sy = (i * 53) % h;
        const tw = 0.4 + 0.6 * Math.abs(Math.sin(performance.now()*0.001 + i));
        ctx.fillStyle = `rgba(200,220,255,${baseAlpha * tw * 0.35})`;
        ctx.fillRect(sx, sy, 1.2, 1.2);
      }
      if (Math.random() < (jam ? 0.035 : 0.012)) spawnShooter();
      for (let i = shooters.length - 1; i >= 0; i--) {
        const s = shooters[i];
        const grad = ctx.createLinearGradient(
          s.x, s.y, s.x - s.vx * 12, s.y - s.vy * 12);
        grad.addColorStop(0, `hsla(${s.hue}, 90%, 75%, ${0.9*s.life})`);
        grad.addColorStop(1, `hsla(${s.hue}, 90%, 70%, 0)`);
        ctx.strokeStyle = grad; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(s.x, s.y);
        ctx.lineTo(s.x - s.vx * 12, s.y - s.vy * 12); ctx.stroke();
        ctx.fillStyle = `hsla(${s.hue}, 100%, 92%, ${s.life})`;
        ctx.beginPath(); ctx.arc(s.x, s.y, 1.8, 0, Math.PI * 2); ctx.fill();
        s.x += s.vx; s.y += s.vy; s.life -= 0.01;
        if (s.life <= 0 || s.x < -80 || s.x > w + 80 || s.y > h + 80) {
          shooters.splice(i, 1);
        }
      }
    }

    // --- aurora: flowing ribbons of color ---
    function drawAurora(jam) {
      const t = performance.now() * 0.0004;
      const amp = jam ? 0.75 : 0.55;
      for (let band = 0; band < 3; band++) {
        const hue = (200 + band * 50 + t * 60) % 360;
        ctx.globalAlpha = amp * (0.22 + 0.10 * band);
        ctx.fillStyle = `hsl(${hue}, 80%, 55%)`;
        ctx.beginPath();
        ctx.moveTo(0, h);
        for (let x = 0; x <= w; x += 12) {
          const y = h * 0.4
            + Math.sin(x * 0.006 + t * 3 + band) * 90
            + Math.sin(x * 0.002 + t * 2 + band * 1.7) * 140;
          ctx.lineTo(x, y);
        }
        ctx.lineTo(w, h); ctx.closePath(); ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    // --- retro grid: vanishing-point perspective lines ---
    function drawGrid(jam) {
      const t = performance.now() * 0.001;
      const horizon = h * 0.55;
      ctx.strokeStyle = jam ? 'rgba(255,80,200,0.55)' : 'rgba(120,180,255,0.32)';
      ctx.lineWidth = 1;
      // horizontal lines receding toward horizon
      for (let i = 0; i < 18; i++) {
        const p = ((i + t * 0.6) % 18) / 18;
        const y = horizon + Math.pow(p, 2.5) * (h - horizon);
        ctx.globalAlpha = Math.min(1, p * 2);
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }
      ctx.globalAlpha = 0.6;
      // vertical lines converging to center
      for (let x = -10; x <= 10; x++) {
        ctx.beginPath();
        ctx.moveTo(w / 2 + x * (w / 18), h);
        ctx.lineTo(w / 2 + x * 4, horizon);
        ctx.stroke();
      }
      // sun
      const sunR = 120;
      const sunGrad = ctx.createLinearGradient(0, horizon - sunR, 0, horizon);
      sunGrad.addColorStop(0, 'rgba(255,180,80,0.9)');
      sunGrad.addColorStop(1, 'rgba(220,60,160,0.9)');
      ctx.fillStyle = sunGrad; ctx.globalAlpha = 1;
      ctx.beginPath(); ctx.arc(w / 2, horizon, sunR, Math.PI, 0); ctx.fill();
    }

    // --- dust motes: slow drifting particles ---
    const motes = Array.from({length: 80}, () => ({
      x: Math.random(), y: Math.random(),
      vx: (Math.random() - 0.5) * 0.0004,
      vy: (Math.random() - 0.5) * 0.0004,
      r: 0.6 + Math.random() * 1.8,
      hue: 180 + Math.random() * 160,
    }));
    function drawParticles(jam) {
      for (const m of motes) {
        m.x += m.vx; m.y += m.vy;
        if (m.x < 0 || m.x > 1) m.vx *= -1;
        if (m.y < 0 || m.y > 1) m.vy *= -1;
        const tw = 0.4 + 0.6 * Math.abs(
          Math.sin(performance.now() * 0.0008 + m.x * 10));
        ctx.fillStyle = `hsla(${m.hue}, 80%, 75%, ${jam ? 0.6 : 0.35 * tw})`;
        ctx.beginPath();
        ctx.arc(m.x * w, m.y * h, m.r * (jam ? 1.6 : 1), 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // --- matrix rain: falling glyph columns ---
    const rainCols = [];
    function ensureRain() {
      const cw = 14;
      const want = Math.ceil(w / cw);
      while (rainCols.length < want) {
        rainCols.push({
          y: -Math.random() * h,
          v: 1.2 + Math.random() * 3.5,
          len: 6 + Math.floor(Math.random() * 18),
        });
      }
    }
    function drawRain(jam) {
      ensureRain();
      ctx.fillStyle = 'rgba(0,0,0,0.18)';
      ctx.fillRect(0, 0, w, h);
      ctx.font = '12px monospace';
      const glyphs = 'ｱｲｳｴｵｶｷｸｹｺ01{}<>';
      for (let i = 0; i < rainCols.length; i++) {
        const c = rainCols[i];
        for (let k = 0; k < c.len; k++) {
          const y = c.y - k * 14;
          const a = jam ? (1 - k / c.len) * 0.9 : (1 - k / c.len) * 0.55;
          ctx.fillStyle = `hsla(${jam ? 300 : 140}, 80%, ${k === 0 ? 80 : 55}%, ${a})`;
          const g = glyphs[(i + k + ((performance.now() * 0.02) | 0)) % glyphs.length];
          ctx.fillText(g, i * 14, y);
        }
        c.y += c.v;
        if (c.y > h + c.len * 14) { c.y = -20; c.v = 1.2 + Math.random() * 3.5; }
      }
    }

    function tick() {
      ctx.clearRect(0, 0, w, h);
      const jam = document.body.classList.contains('jam');
      switch (__bgMode) {
        case 'aurora':    drawAurora(jam); break;
        case 'grid':      drawGrid(jam); break;
        case 'particles': drawParticles(jam); break;
        case 'rain':      drawRain(jam); break;
        case 'off':       break;
        case 'stars':
        default:          drawStars(jam);
      }
      requestAnimationFrame(tick);
    }
    tick();
  })();

  window.__setBgMode = (m) => {
    __bgMode = m;
    try { localStorage.setItem('bgMode', m); } catch (e) {}
  };

  const ccBg = document.getElementById('cc-bg-opts');
  if (ccBg) {
    paintActive(ccBg, __bgMode);
    ccBg.addEventListener('click', (e) => {
      const b = e.target.closest('.cc-opt'); if (!b) return;
      paintActive(ccBg, b.dataset.v);
      window.__setBgMode(b.dataset.v);
    });
  }

  // Voice commands (Web Speech API)
  (function initVoice() {
    const ccVoice = document.getElementById('cc-voice-opts');
    const panel = document.getElementById('cc-voice-panel');
    const mapTa = document.getElementById('cc-voice-map');
    const heard = document.getElementById('cc-voice-heard');
    if (!ccVoice || !mapTa) return;

    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    let enabled = false;
    let recog = null;
    let recentFire = 0;

    const savedMap = localStorage.getItem('handsfreeVoiceMap');
    mapTa.value = savedMap != null ? savedMap : 'telegram: cmd+t\\narc: cmd+a';
    mapTa.addEventListener('input', () => {
      try { localStorage.setItem('handsfreeVoiceMap', mapTa.value); } catch(e){}
    });

    function parseMap() {
      const out = [];
      for (const line of mapTa.value.split('\\n')) {
        const m = line.match(/^\s*([^:]+?)\s*:\s*(.+?)\s*$/);
        if (!m) continue;
        out.push({ phrase: m[1].toLowerCase(), combo: m[2] });
      }
      return out;
    }

    function maybeFire(transcript) {
      const txt = transcript.toLowerCase().trim();
      if (!txt) return;
      heard.textContent = '“' + txt.slice(0, 60) + '”';
      const now = Date.now();
      if (now - recentFire < 1200) return;
      for (const { phrase, combo } of parseMap()) {
        if (phrase && txt.includes(phrase)) {
          recentFire = now;
          fetch('/command', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'voice_hotkey', phrase, combo }),
          }).catch(() => {});
          heard.textContent = '→ ' + phrase + ' (' + combo + ')';
          return;
        }
      }
    }

    function startRecog() {
      if (!SR) { heard.textContent = 'Web Speech API unavailable'; return; }
      recog = new SR();
      recog.continuous = true;
      recog.interimResults = true;
      recog.lang = 'en-US';
      recog.onresult = (ev) => {
        for (let i = ev.resultIndex; i < ev.results.length; i++) {
          const r = ev.results[i];
          if (r.isFinal || r[0].confidence > 0.5) {
            maybeFire(r[0].transcript);
          }
        }
      };
      recog.onerror = (ev) => {
        heard.textContent = 'err: ' + ev.error;
      };
      recog.onend = () => { if (enabled) { try { recog.start(); } catch(e){} } };
      try { recog.start(); } catch (e) {}
    }

    function stopRecog() {
      if (recog) { try { recog.stop(); } catch (e) {} recog = null; }
      heard.textContent = '';
    }

    function setVoice(v) {
      enabled = (v === 'on');
      panel.style.display = enabled ? 'flex' : 'none';
      paintActive(ccVoice, v);
      try { localStorage.setItem('handsfreeVoiceOn', v); } catch(e) {}
      if (enabled) startRecog(); else stopRecog();
    }

    ccVoice.addEventListener('click', (e) => {
      const b = e.target.closest('.cc-opt'); if (!b) return;
      setVoice(b.dataset.v);
    });

    const savedOn = localStorage.getItem('handsfreeVoiceOn') || 'off';
    setVoice(savedOn);
  })();
})();
</script>
</body></html>
"""


# --- shared state ---------------------------------------------------------
_latest_jpeg: Optional[bytes] = None
_jpeg_lock = threading.Lock()
# Efficiency: count of active /stream HTTP clients. When zero, we skip
# the cv2.imencode step entirely — no one is watching, no point spending
# CPU on JPEG.
_stream_clients: int = 0
_stream_clients_lock = threading.Lock()
_stop = threading.Event()

_state_lock = threading.Lock()
_motion_val = 0.0   # still computed; not sent
_smile_val = 0.0
_bob_pending = False
_blink_pending = False
_prayer_start_pending = False
_prayer_end_pending = False
_prayer_active = False
_boot_pending = False
_clap_tick_pending = False
_swipe_pending: Optional[str] = None  # "left" or "right"
_atelier_toggle_pending: bool = False
_atelier_action_pending: Optional[str] = None

# 🥊 Muay Thai / boxing mode. When enabled, the capture loop classifies
# fast wrist motion into jabs / hooks / uppercuts and emits SSE punch
# events so the /boxing page can react. Cheap to compute, gated behind a
# flag so it doesn't add overhead when nobody's training.
_boxing_enabled: bool = False
_punch_pending: list = []  # list[dict] of punch events to flush this tick
_left_wrist_prev: Optional[tuple] = None
_right_wrist_prev: Optional[tuple] = None
_left_punch_last_at: float = 0.0
_right_punch_last_at: float = 0.0
PUNCH_VELOCITY_THR: float = 0.040     # normalized units / frame
PUNCH_COOLDOWN_S: float = 0.22
_dictation_pending = False
_left_hand_y: Optional[float] = None
_right_hand_y: Optional[float] = None
_left_hand_x: Optional[float] = None
_right_hand_x: Optional[float] = None
_jam_mode: bool = False

# Cursor prototyping: pointing method + click method, hot-swappable from UI.
_pointing_method: str = "finger"  # "head" | "finger" | "gaze"
_click_method: str = "mouth"      # primary (left) click gesture
_cursor_sens: float = 1.5         # multiplier on all pointing-method gains
_right_click_method: str = "off"  # "smile" | "pucker" | "furrow" | "off"
_double_click_on: bool = False    # double-tap primary within DBL window → double-click
_wispr_method: str = "off"  # "applescript_fn"|"cgevent_f19"|"cgevent_fn"|"all"|"apple_dictation"|"off"
# If True: after releasing a held Fn key, also fire a fresh tap. Useful when
# Wispr treats Fn as a toggle and the key-up alone doesn't stop recording.
_wispr_release_extra_tap: bool = False
_wispr_release_nuclear: bool = False   # fire nuclear-stop cascade on release

# Calibration centers (captured on cursor enable, per method).
_finger_center: Optional[tuple] = None   # (fx, fy) in [0..1]
_gaze_center: Optional[tuple] = None     # (gx, gy) in [0..1]

# Click detector state
_last_brow_click_at: float = 0.0
_pinch_was_closed: bool = False
_last_pinch_click_at: float = 0.0
_last_wink_click_at: float = 0.0
_wink_armed: bool = True    # edge-trigger gate so one wink = one click
_last_blink_click_at: float = 0.0
_blink_click_armed: bool = True
_last_right_wink_click_at: float = 0.0
_right_wink_armed: bool = True
_last_mouth_click_at: float = 0.0
_mouth_armed: bool = True
_last_smile_click_at: float = 0.0
_smile_click_armed: bool = True
_last_pucker_click_at: float = 0.0
_pucker_click_armed: bool = True
_last_furrow_click_at: float = 0.0
_furrow_click_armed: bool = True

# --- Experimental toggles (off by default) ---
# T-gesture = timeout: make a T with both hands to toggle master on/off.
_t_timeout_enabled: bool = True
_t_gesture_armed: bool = True
_t_last_trigger_at: float = 0.0
T_GESTURE_COOLDOWN_S: float = 1.2

# Mouth-hold: when mouth click is primary, mouth-open = mouseDown,
# mouth-close = mouseUp (press-and-hold instead of discrete click).
_mouth_hold_enabled: bool = False
_mouth_hold_down: bool = False

# Peace sign ✌️ → press-and-hold the mouse (hold-to-drag).
# While ✌️ is shown, mouse stays down; release → mouseUp.
# Lets you select text + drag in Figma without a rogue press-and-hold
# from any other gesture (e.g. mouth-hold) firing accidentally.
_peace_rclick_enabled: bool = True   # name kept for setting compat
_peace_armed: bool = True            # legacy, unused
_peace_last_at: float = 0.0          # legacy, unused
_peace_hold_down: bool = False
PEACE_HOLD_RELEASE_GRACE_S: float = 0.18  # keep mouse down briefly
_peace_hold_release_at: float = 0.0       # if hand momentarily lost
# Push-to-copy: while ✌️ is held, pushing the hand toward the camera
# (palm scale grows past PUSH_COPY_RATIO of the baseline) fires Cmd+C.
# Only fires once per peace-hold session; resets on release.
_peace_push_baseline: Optional[float] = None
_peace_push_baseline_t: float = 0.0
_peace_push_fired: bool = False
PUSH_COPY_BASELINE_S: float = 0.20   # average first ~200ms for baseline
PUSH_COPY_RATIO: float = 1.32        # 32% bigger ⇒ "shoved at camera"

# Thumbs up 👍 → paste (Cmd+V). (Setting name kept for state compat.)
_thumbs_dclick_enabled: bool = True

# Head bob UP (chin-lift) → Cmd+C copy.
_head_copy_enabled: bool = True

# Fist depth zoom: while fist is held, hand moving toward camera = Cmd+=
# (zoom in), away from camera = Cmd+- (zoom out). Lateral motion still
# pans/scrolls as before.
_fist_zoom_enabled: bool = True
_fist_zoom_baseline: Optional[float] = None
_fist_zoom_last_at: float = 0.0
FIST_ZOOM_RATIO_IN: float = 1.18    # 18% bigger ⇒ zoom in
FIST_ZOOM_RATIO_OUT: float = 0.85   # 15% smaller ⇒ zoom out
FIST_ZOOM_COOLDOWN_S: float = 0.16  # gap between zoom steps
_thumbs_armed: bool = True
_thumbs_last_at: float = 0.0

# ✨ Atelier mode — A-pose (fingertips together, wrists spread low, elbows
# wide) toggles a Figma-focused "design canvas" mode. While active, two-
# handed gestures map to zoom & pan: spreading hands = zoom in, squeezing
# = zoom out, translating both hands together = pan the canvas.
_atelier_enabled: bool = True       # feature on/off (user can disable)
_atelier_mode: bool = False         # currently IN atelier mode
_atelier_armed: bool = True
_atelier_last_at: float = 0.0
ATELIER_COOLDOWN_S: float = 1.2
_atelier_baseline_dist: Optional[float] = None
_atelier_baseline_cx: Optional[float] = None
_atelier_baseline_cy: Optional[float] = None
_atelier_last_action_at: float = 0.0
ATELIER_ACTION_COOLDOWN_S: float = 0.08
ATELIER_ZOOM_DELTA: float = 0.035    # wrist-dist change to trigger zoom step
ATELIER_PAN_DELTA: float = 0.025     # centroid change to trigger pan step
# Head dolly zoom — face size (eye-corner span) vs baseline. Leaning in
# shrinks the camera-to-face distance so eye span grows.
_atelier_face_baseline: Optional[float] = None
ATELIER_FACE_ZOOM_DELTA: float = 0.015
ATELIER_FACE_COOLDOWN_S: float = 0.22
_atelier_face_last_at: float = 0.0
# Pinch-to-grab: while atelier is on, pinching (thumb+index together)
# = mouseDown; releasing = mouseUp. Drag anything.
_atelier_pinch_down: bool = False
ATELIER_PINCH_ON_THR: float = 0.045
ATELIER_PINCH_OFF_THR: float = 0.075

GESTURE_COOLDOWN_S: float = 1.0
# Pending-click mechanism so two primaries within DOUBLE_WINDOW_S merge into
# one double-click instead of firing two singles.
_pending_click_at: float = 0.0
_pending_click_waiting: bool = False
DOUBLE_WINDOW_S: float = 0.35
# Thresholds for new right-click gestures
SMILE_CLICK_THRESHOLD = 0.75
SMILE_CLICK_OPEN_THR  = 0.45
SMILE_CLICK_COOLDOWN_S = 0.5
PUCKER_CLICK_THRESHOLD = 0.55
PUCKER_CLICK_OPEN_THR  = 0.25
PUCKER_CLICK_COOLDOWN_S = 0.5
FURROW_CLICK_THRESHOLD = 0.55
FURROW_CLICK_OPEN_THR  = 0.25
FURROW_CLICK_COOLDOWN_S = 0.5

# Two-hand vertical scroll: both hands in frame → scroll mode.
# While in scroll mode the cursor is frozen and avg-y velocity scrolls.
_scroll_prev_y: Optional[float] = None
_scroll_active: bool = False
_scroll_accum: float = 0.0
SCROLL_ZONE_Y_MAX  = 0.85   # both wrists above waist
SCROLL_FRAME_DEAD  = 0.0015 # ignore per-frame jitter below this
SCROLL_GAIN_MAP = {"gentle": 80.0, "normal": 180.0, "zippy": 360.0}
_scroll_sens: str = "normal"
# Continuous multiplier on top of the base scroll gain. Defaults to 3× so
# fist-scroll feels comparable to a real trackpad on first run; tweakable
# from the UI slider (`action: scroll_speed`).
BASE_SCROLL_GAIN: float = 180.0
_scroll_sens_mult: float = 3.0
# Scroll gesture mode. "fist" = close one hand into a fist and move up/down
# to scroll (new default — works single-handed, cursor freezes while fist
# is closed). "two_hands" = legacy mode (raise both hands then move them).
# "off" = disabled.
_scroll_mode: str = "fist"
_fist_scroll_prev: Optional[tuple] = None  # (y, x) normalized

# Head-pitch scroll: tilt chin up/down to scroll, gated by a chosen signal.
_head_scroll_baseline_y: Optional[float] = None
_head_scroll_accum: float = 0.0
HEAD_SCROLL_DEADBAND = 0.012  # ~1% of frame height around rest position

# Brow-raise / brow-furrow scroll
_brow_scroll_accum: float = 0.0
BROW_SCROLL_THRESHOLD = 0.18  # blendshape must exceed this to count
_fist_scroll_accum_y: float = 0.0
_fist_scroll_accum_x: float = 0.0

# Two-hand tab-swipe: both hands up, one stays still (anchor), the other
# sweeps sideways → fires Cmd+Shift+] / Cmd+Shift+[ to switch browser tabs.
TAB_SWIPE_WINDOW_S = 0.45
TAB_SWIPE_MIN_DX = 0.30
TAB_SWIPE_ANCHOR_MAX_DX = 0.10
TAB_SWIPE_Y_MAX = 0.60
TAB_SWIPE_COOLDOWN_S = 1.0
TAB_SWIPE_NEXT_COMBO = "cmd+shift+]"
TAB_SWIPE_PREV_COMBO = "cmd+shift+["
# macOS "Move between Spaces" shortcuts (built-in). Same feel as a
# three-finger trackpad swipe.
SPACE_SWIPE_NEXT_COMBO = "ctrl+right"
SPACE_SWIPE_PREV_COMBO = "ctrl+left"
# "tabs" → browser tabs, "spaces" → macOS Spaces, "off" → disabled.
_tab_swipe_action: str = "spaces"
_tab_swipe_hist: Deque[tuple] = deque(maxlen=60)
_last_tab_swipe_at: float = 0.0
_tab_swipe_enabled: bool = True
BROW_CLICK_THRESHOLD = 0.55
BROW_CLICK_COOLDOWN_S = 0.7
WINK_CLOSED_THRESHOLD = 0.55   # one eye must be this closed…
WINK_OPEN_THRESHOLD   = 0.3    # …while the other stays below this (loose rearm)
WINK_COOLDOWN_S       = 0.5
BLINK_CLICK_THRESHOLD = 0.55   # both eyes closed: score on both
BLINK_CLICK_OPEN_THR  = 0.3
BLINK_CLICK_COOLDOWN_S = 0.45
MOUTH_CLICK_THRESHOLD = 0.45
MOUTH_CLICK_OPEN_THR  = 0.2
MOUTH_CLICK_COOLDOWN_S = 0.4
PINCH_CLOSE_THRESHOLD = 0.055   # normalized (frame coords)
PINCH_OPEN_THRESHOLD = 0.085
PINCH_COOLDOWN_S = 0.35

# Finger pointing: remap a smaller box to full screen so he doesn't have to
# reach frame edges.
FINGER_GAIN_X = 2.2
FINGER_GAIN_Y = 2.0

# Gaze pointing: iris deltas are tiny → high gain.
GAZE_GAIN_X = 12.0
GAZE_GAIN_Y = 10.0

_prev_face_nose: Optional[tuple] = None
_prev_hand_wrists: list = []
_nose_y_history: Deque[float] = deque(maxlen=BOB_WINDOW_FRAMES)
_last_bob_at = 0.0

_blink_armed = True
_last_blink_at = 0.0

_swipe_xs: Deque[tuple] = deque(maxlen=40)
_last_swipe_at = 0.0

_prayer_hold_start: Optional[float] = None
_prayer_active_since: Optional[float] = None
_prayer_hands_lost_since: Optional[float] = None

# Dictation trigger gesture. "prayer" (palms together — classic, but one hand
# often occludes the other). "fingertips" (index tips touch — more robust
# because hands don't overlap). "fist" (one-handed closed fist — single-hand,
# no tracker fights). "off" disables gesture-triggered dictation entirely.
_dict_gesture = "off"

# How the gesture maps to Wispr/dictation key events:
#   "hold"  — press down on gesture ENTER, release on gesture EXIT
#             (Wispr Flow must be in hold-to-talk mode)
#   "latch" — tap once on gesture ENTER, do nothing on EXIT; tap again on next
#             ENTER (Wispr or Apple dictation must be in toggle/tap mode)
_dict_mode = "latch"

_hands_up_start: Optional[float] = None
_last_hands_up_at = 0.0

_clap_state = "far"  # or "close"
_clap_times: List[float] = []
_last_clap_boot_at = 0.0
_last_two_hands_at = 0.0  # last time we saw 2 hands close together

# Master enable — double clap toggles all gesture actions on/off.
# Detection still runs so we can catch the re-enable clap.
_system_enabled = True

# Per-gesture enables, toggleable from Control Center.
_scroll_gesture_enabled = True  # distinct from scroll speed preset
_swipe_gesture_enabled = False

_fn_pressed = False

_screen_w, _screen_h = pyautogui.size()

# Virtual-desktop bounding box across ALL monitors.  On macOS the cursor
# lives in a coord space where (0,0) is the top-left of the primary display
# and secondary displays can have negative or > _screen_w coordinates.
# We compute the union via AppKit so gesture-driven cursor can traverse
# every monitor the same way the trackpad does.
def _virtual_screen_bounds() -> tuple:
    try:
        from AppKit import NSScreen
        screens = NSScreen.screens()
        if not screens:
            return 0.0, 0.0, float(_screen_w), float(_screen_h)
        # NSScreen frames use a bottom-left origin where y grows upward.
        # Mouse coords use a top-left origin where y grows downward.
        # Flip y using the primary (index 0) screen height.
        primary_h = float(screens[0].frame().size.height)
        xs_min = []
        ys_min = []
        xs_max = []
        ys_max = []
        for s in screens:
            f = s.frame()
            x = float(f.origin.x)
            y_bl = float(f.origin.y)
            w = float(f.size.width)
            h = float(f.size.height)
            # Convert bottom-left-origin y to top-left-origin y.
            y_tl = primary_h - (y_bl + h)
            xs_min.append(x)
            ys_min.append(y_tl)
            xs_max.append(x + w)
            ys_max.append(y_tl + h)
        return min(xs_min), min(ys_min), max(xs_max), max(ys_max)
    except Exception as e:
        print(f"[viewer] virtual-screen probe failed: {e}", flush=True)
        return 0.0, 0.0, float(_screen_w), float(_screen_h)


_virt_x0, _virt_y0, _virt_x1, _virt_y1 = _virtual_screen_bounds()
_virt_w = _virt_x1 - _virt_x0
_virt_h = _virt_y1 - _virt_y0
print(f"[viewer] virtual screen: ({_virt_x0:.0f},{_virt_y0:.0f}) → "
      f"({_virt_x1:.0f},{_virt_y1:.0f})  size {_virt_w:.0f}x{_virt_h:.0f}  "
      f"(primary {_screen_w}x{_screen_h})", flush=True)

_cursor_enabled = True
_cursor_calibrated = False
_cursor_calib_start: Optional[float] = None
_yaw_center = 0.0
_pitch_center = 0.0
_cur_x = (_virt_x0 + _virt_x1) * 0.5
_cur_y = (_virt_y0 + _virt_y1) * 0.5


def _move_cursor_virtual(x: float, y: float) -> None:
    """Move the cursor to (x, y) in the FULL virtual-desktop coord space.
    pyautogui.moveTo uses CGEventPost which gets clamped to the primary
    display on macOS — so with multiple monitors the cursor gets stuck
    on the main screen. CGWarpMouseCursorPosition places the cursor at
    absolute virtual coords (supports negative x/y, >primary_w, etc.),
    and we also post a mouseMoved event so apps see hover updates."""
    if _QUARTZ_OK:
        try:
            CGWarpMouseCursorPosition((float(x), float(y)))
            ev = CGEventCreateMouseEvent(
                None, kCGEventMouseMoved,
                (float(x), float(y)), kCGMouseButtonLeft,
            )
            CGEventPost(kCGHIDEventTap, ev)
            return
        except Exception as e:
            print(f"[viewer] CG move failed, fallback pyautogui: {e}",
                  flush=True)
    try:
        pyautogui.moveTo(x, y, _pause=False)
    except Exception:
        pass


def _matrix_to_yaw_pitch(matrix) -> tuple:
    r = np.array(matrix).reshape(4, 4)[:3, :3]
    sy = float(np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2))
    if sy > 1e-6:
        yaw = float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))
        pitch = float(np.degrees(np.arctan2(-r[2, 0], sy)))
    else:
        yaw = 0.0
        pitch = float(np.degrees(np.arctan2(-r[2, 0], sy)))
    return yaw, pitch


def _draw_face_landmarks(frame_bgr: np.ndarray, landmarks) -> None:
    h, w = frame_bgr.shape[:2]
    for lm in landmarks[::4]:
        cv2.circle(frame_bgr, (int(lm.x * w), int(lm.y * h)), 1,
                   FACE_DOT_COLOR_BGR, -1)


def _draw_hand(frame_bgr: np.ndarray, hand_landmarks) -> None:
    h, w = frame_bgr.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame_bgr, pts[a], pts[b], HAND_LINE_COLOR_BGR, 1,
                 cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        r = 4 if i in (4, 8, 12, 16, 20) else 2
        cv2.circle(frame_bgr, (x, y), r, HAND_DOT_COLOR_BGR, -1, cv2.LINE_AA)


def _palm_center(hand):
    xs = [hand[i].x for i in PALM_IDXS]
    ys = [hand[i].y for i in PALM_IDXS]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _ensure_model(path: Path, url: str) -> bool:
    if path.exists():
        return True
    try:
        print(f"[viewer] downloading {path.name}...", flush=True)
        urllib.request.urlretrieve(url, path)
        return True
    except Exception as e:
        print(f"[viewer] download failed: {e}", flush=True)
        return False


_last_head_up_at: float = 0.0
HEAD_UP_ASCENT_THRESHOLD = 0.012   # bigger than bob-down so it's deliberate
HEAD_UP_COOLDOWN_S = 0.55


def _detect_head_bob_up(nose_y: Optional[float], now: float) -> bool:
    """Fire when the head bobs UP (chin-lift nod) and returns.

    Mirrors `_detect_bob` but inverts direction. Reads the same
    rolling nose-y history; uses a separate cooldown so up- and
    down-bobs don't fight.
    """
    global _last_head_up_at
    if nose_y is None:
        return False
    if len(_nose_y_history) < 5:
        return False
    hist = list(_nose_y_history)
    # Image coords: y decreases as head rises, so ascent = hist[-5]-hist[-3].
    earlier_ascent = hist[-5] - hist[-3]
    recent_vel = hist[-1] - hist[-3]   # positive = falling back down
    if (earlier_ascent > HEAD_UP_ASCENT_THRESHOLD
            and recent_vel >= 0
            and (now - _last_head_up_at) > HEAD_UP_COOLDOWN_S):
        _last_head_up_at = now
        return True
    return False


def _detect_bob(nose_y: Optional[float], now: float) -> bool:
    """Fire at the bottom of a head bob.

    Uses a rolling window of recent nose_y values. A bob has two halves:
    descent (nose_y rises — head moves down) then return (nose_y falls).
    We fire when the earlier half has non-trivial descent and the latest
    two frames are either flat or rising back up (head reached bottom).
    """
    global _last_bob_at
    if nose_y is None:
        return False
    _nose_y_history.append(nose_y)
    if len(_nose_y_history) < 5:
        return False
    hist = list(_nose_y_history)
    # Descent of the earlier 3 frames (positive = nose moved down).
    earlier_descent = hist[-3] - hist[-5]
    # Recent 2 frames velocity (positive = still falling down).
    recent_vel = hist[-1] - hist[-3]
    if (earlier_descent > BOB_DESCENT_THRESHOLD
            and recent_vel <= 0
            and (now - _last_bob_at) > BOB_COOLDOWN_S):
        _last_bob_at = now
        return True
    return False


def _detect_blink(blendshapes, now: float) -> bool:
    """Fire on a simultaneous both-eye blink."""
    global _blink_armed, _last_blink_at
    if not blendshapes:
        return False
    l = r = 0.0
    for b in blendshapes:
        if b.category_name == "eyeBlinkLeft":
            l = float(b.score)
        elif b.category_name == "eyeBlinkRight":
            r = float(b.score)
    combined = (l + r) * 0.5
    if (_blink_armed and combined > BLINK_ON_THRESHOLD
            and (now - _last_blink_at) > BLINK_COOLDOWN_S):
        _last_blink_at = now
        _blink_armed = False
        return True
    if not _blink_armed and combined < BLINK_OFF_THRESHOLD:
        _blink_armed = True
    return False


def _update_swipe(hands_list, now: float) -> Optional[str]:
    """Record one-handed X over time; fire 'left'/'right' on a full traversal."""
    global _last_swipe_at
    # Only sample when exactly one hand is in frame, so two-hand shuffling
    # can't produce phantom jumps in the tracked X.
    if len(hands_list) == 1:
        _swipe_xs.append((now, hands_list[0][0].x))
    else:
        # Drop stale history while multi-hand or no-hand.
        while _swipe_xs and now - _swipe_xs[0][0] > SWIPE_WINDOW_S:
            _swipe_xs.popleft()
        return None

    # Trim old samples.
    while _swipe_xs and now - _swipe_xs[0][0] > SWIPE_WINDOW_S:
        _swipe_xs.popleft()
    if len(_swipe_xs) < 4:
        return None

    first_x = _swipe_xs[0][1]
    last_x = _swipe_xs[-1][1]
    delta = last_x - first_x
    direction: Optional[str] = None
    if (delta > SWIPE_MIN_TRAVEL
            and first_x < SWIPE_EDGE_START and last_x > SWIPE_EDGE_END):
        direction = "right"
    elif (delta < -SWIPE_MIN_TRAVEL
            and first_x > SWIPE_EDGE_END and last_x < SWIPE_EDGE_START):
        direction = "left"
    if direction and (now - _last_swipe_at) > SWIPE_COOLDOWN_S:
        _last_swipe_at = now
        _swipe_xs.clear()
        return direction
    return None


def _is_fist(hand) -> bool:
    """Closed fist: ALL 4 fingers curl below their middle joint.
    Uses landmark y (lower = higher in image). For each finger, tip y
    should be greater (lower down) than the PIP joint y.

    Requires all 4 (not 3) so a pointing-index posture (3 curled,
    index extended) is NOT a fist — otherwise the cursor hand would
    spuriously trigger scroll.
    """
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for t, p in zip(tips, pips):
        if hand[t].y <= hand[p].y + 0.015:
            return False
    return True


def _fingertips_touching(hands_list) -> Optional[float]:
    """Distance between the two index fingertips if both hands visible."""
    if len(hands_list) < 2:
        return None
    p0 = hands_list[0][8]
    p1 = hands_list[1][8]
    return math.hypot(p0.x - p1.x, p0.y - p1.y)


def _update_dict_gesture(hands_list, now: float) -> Optional[bool]:
    """Dispatches to the active dictation gesture.
    Returns True on ENTER, False on EXIT, None otherwise."""
    if _dict_gesture == "off":
        # Ensure we release if someone turns it off mid-hold.
        if _prayer_active:
            return _update_prayer_like_release()
        return None
    if _dict_gesture == "fingertips":
        return _update_fingertips_gesture(hands_list, now)
    if _dict_gesture == "fist":
        return _update_fist_gesture(hands_list, now)
    return _update_prayer(hands_list, now)


def _update_prayer_like_release() -> bool:
    """Force-release the active dictation gesture (shared across detectors)."""
    global _prayer_active, _prayer_active_since, _prayer_hands_lost_since
    global _prayer_hold_start
    _prayer_active = False
    _prayer_active_since = None
    _prayer_hands_lost_since = None
    _prayer_hold_start = None
    print("[viewer] dict gesture release (mode-change)", flush=True)
    return False


def _update_fingertips_gesture(hands_list, now: float) -> Optional[bool]:
    """Touching index fingertips (both hands visible) → hold."""
    global _prayer_hold_start, _prayer_active
    global _prayer_active_since, _prayer_hands_lost_since

    CLOSE = 0.07
    OPEN  = 0.14

    if _prayer_active and _prayer_active_since is not None \
            and now - _prayer_active_since > PRAYER_MAX_HOLD_S:
        return _update_prayer_like_release()

    d = _fingertips_touching(hands_list)
    if d is None:
        _prayer_hold_start = None
        if _prayer_active:
            if _prayer_hands_lost_since is None:
                _prayer_hands_lost_since = now
            elif now - _prayer_hands_lost_since > PRAYER_LOST_HANDS_GRACE_S:
                return _update_prayer_like_release()
        return None

    _prayer_hands_lost_since = None
    if d < CLOSE:
        if _prayer_hold_start is None:
            _prayer_hold_start = now
        elif (not _prayer_active
                and now - _prayer_hold_start >= PRAYER_ENTER_HOLD_S):
            _prayer_active = True
            _prayer_active_since = now
            print(f"[viewer] fingertips enter d={d:.3f}", flush=True)
            return True
    elif d > OPEN:
        _prayer_hold_start = None
        if _prayer_active:
            return _update_prayer_like_release()
    return None


def _update_fist_gesture(hands_list, now: float) -> Optional[bool]:
    """One-handed closed fist → hold while the fist stays closed."""
    global _prayer_hold_start, _prayer_active
    global _prayer_active_since, _prayer_hands_lost_since

    if _prayer_active and _prayer_active_since is not None \
            and now - _prayer_active_since > PRAYER_MAX_HOLD_S:
        return _update_prayer_like_release()

    closed = any(_is_fist(h) for h in hands_list)

    if not hands_list:
        _prayer_hold_start = None
        if _prayer_active:
            if _prayer_hands_lost_since is None:
                _prayer_hands_lost_since = now
            elif now - _prayer_hands_lost_since > PRAYER_LOST_HANDS_GRACE_S:
                return _update_prayer_like_release()
        return None

    _prayer_hands_lost_since = None
    if closed:
        if _prayer_hold_start is None:
            _prayer_hold_start = now
        elif (not _prayer_active
                and now - _prayer_hold_start >= PRAYER_ENTER_HOLD_S):
            _prayer_active = True
            _prayer_active_since = now
            print("[viewer] fist enter", flush=True)
            return True
    else:
        _prayer_hold_start = None
        if _prayer_active:
            return _update_prayer_like_release()
    return None


def _update_prayer(hands_list, now: float) -> Optional[bool]:
    """Track prayer hold. Returns True on enter, False on exit, None otherwise.

    Release rules (any triggers an exit):
      - palms pull apart past PRAYER_OPEN_THRESHOLD
      - fewer than 2 hands visible for PRAYER_LOST_HANDS_GRACE_S
      - safety timeout after PRAYER_MAX_HOLD_S
    """
    global _prayer_hold_start, _prayer_active
    global _prayer_active_since, _prayer_hands_lost_since

    def _release(reason: str) -> bool:
        global _prayer_active, _prayer_active_since, _prayer_hands_lost_since
        global _prayer_hold_start
        _prayer_active = False
        _prayer_active_since = None
        _prayer_hands_lost_since = None
        _prayer_hold_start = None
        print(f"[viewer] prayer release ({reason})", flush=True)
        return False

    # Safety timeout.
    if (_prayer_active and _prayer_active_since is not None
            and now - _prayer_active_since > PRAYER_MAX_HOLD_S):
        return _release("timeout")

    if len(hands_list) < 2:
        _prayer_hold_start = None
        if _prayer_active:
            if _prayer_hands_lost_since is None:
                _prayer_hands_lost_since = now
            elif now - _prayer_hands_lost_since > PRAYER_LOST_HANDS_GRACE_S:
                return _release("hands lost")
        return None

    _prayer_hands_lost_since = None
    p0 = _palm_center(hands_list[0])
    p1 = _palm_center(hands_list[1])
    d = math.hypot(p0[0] - p1[0], p0[1] - p1[1])

    if d < PRAYER_CLOSE_THRESHOLD:
        if _prayer_hold_start is None:
            _prayer_hold_start = now
        elif not _prayer_active and now - _prayer_hold_start >= PRAYER_ENTER_HOLD_S:
            _prayer_active = True
            _prayer_active_since = now
            print(f"[viewer] prayer enter (d={d:.3f})", flush=True)
            return True
    elif d > PRAYER_OPEN_THRESHOLD:
        _prayer_hold_start = None
        if _prayer_active:
            return _release(f"open d={d:.3f}")
    return None


def _update_hands_up(hands_list, now: float) -> bool:
    """Fire True when both wrists held above HANDS_UP_Y_MAX for HANDS_UP_HOLD_S."""
    global _hands_up_start, _last_hands_up_at
    if len(hands_list) < 2:
        _hands_up_start = None
        return False
    y0 = hands_list[0][0].y
    y1 = hands_list[1][0].y
    if y0 < HANDS_UP_Y_MAX and y1 < HANDS_UP_Y_MAX:
        if _hands_up_start is None:
            _hands_up_start = now
        elif (now - _hands_up_start >= HANDS_UP_HOLD_S
                and now - _last_hands_up_at > HANDS_UP_COOLDOWN_S):
            _last_hands_up_at = now
            _hands_up_start = None
            return True
    else:
        _hands_up_start = None
    return False


def _update_double_clap(hands_list, now: float):
    """Returns (doubled: bool, single: bool).
    single=True on each individual clap (for UI feedback).
    doubled=True on valid double-clap."""
    global _clap_state, _clap_times, _last_clap_boot_at, _last_two_hands_at
    single = False
    doubled = False

    if len(hands_list) >= 2:
        p0 = _palm_center(hands_list[0])
        p1 = _palm_center(hands_list[1])
        d = math.hypot(p0[0] - p1[0], p0[1] - p1[1])
        _last_two_hands_at = now

        if _clap_state == "far" and d < CLAP_CLOSE_THRESHOLD:
            _clap_state = "close"
            _clap_times.append(now)
            _clap_times = [t for t in _clap_times
                           if now - t < CLAP_GAP_MAX_S * 2]
            single = True
            if len(_clap_times) >= 2:
                gap = _clap_times[-1] - _clap_times[-2]
                if (CLAP_GAP_MIN_S < gap < CLAP_GAP_MAX_S
                        and now - _last_clap_boot_at > CLAP_BOOT_COOLDOWN_S):
                    _last_clap_boot_at = now
                    _clap_times.clear()
                    doubled = True
        elif _clap_state == "close" and d > CLAP_FAR_THRESHOLD:
            _clap_state = "far"
    else:
        # Hand(s) lost during collision is common. Only drop back to "far"
        # after a grace window — otherwise we'd miss the release.
        if (_clap_state == "close"
                and now - _last_two_hands_at > CLAP_HAND_LOST_GRACE_S):
            _clap_state = "far"

    return doubled, single


def _tap_wispr_cgevent(keycode: int, fn_flag: bool = False) -> None:
    if not _QUARTZ_OK:
        print("[viewer] cgevent skipped: Quartz unavailable", flush=True)
        return
    try:
        down = CGEventCreateKeyboardEvent(None, keycode, True)
        up   = CGEventCreateKeyboardEvent(None, keycode, False)
        if fn_flag:
            CGEventSetFlags(down, kCGEventFlagMaskSecondaryFn)
            CGEventSetFlags(up,   kCGEventFlagMaskSecondaryFn)
        CGEventPost(kCGHIDEventTap, down)
        CGEventPost(kCGHIDEventTap, up)
        print(f"[viewer] wispr cgevent tapped (key={keycode} fn={fn_flag})",
              flush=True)
    except Exception as e:
        print(f"[viewer] cgevent tap failed: {e}", flush=True)


# macOS virtual key codes for voice-command hotkey firing.
_VOICE_KEYS = {
    "a": 0, "b": 11, "c": 8, "d": 2, "e": 14, "f": 3, "g": 5, "h": 4,
    "i": 34, "j": 38, "k": 40, "l": 37, "m": 46, "n": 45, "o": 31, "p": 35,
    "q": 12, "r": 15, "s": 1, "t": 17, "u": 32, "v": 9, "w": 13, "x": 7,
    "y": 16, "z": 6,
    "0": 29, "1": 18, "2": 19, "3": 20, "4": 21, "5": 23, "6": 22, "7": 26,
    "8": 28, "9": 25,
    "space": 49, "enter": 36, "return": 36, "tab": 48, "esc": 53,
    "escape": 53, "up": 126, "down": 125, "left": 123, "right": 124,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97, "f7": 98,
    "f8": 100, "f9": 101, "f10": 109, "f11": 103, "f12": 111, "f13": 105,
    "f14": 107, "f15": 113, "f16": 106, "f17": 64, "f18": 79, "f19": 80,
}
_VOICE_MODS = {
    "cmd": 0x00100000, "command": 0x00100000, "meta": 0x00100000,
    "shift": 0x00020000,
    "opt": 0x00080000, "option": 0x00080000, "alt": 0x00080000,
    "ctrl": 0x00040000, "control": 0x00040000,
}


def _fire_hotkey(combo: str) -> bool:
    """Parse 'cmd+t' / 'cmd+shift+a' style strings and fire via CGEvent."""
    if not _QUARTZ_OK:
        print("[viewer] hotkey skipped: Quartz unavailable", flush=True)
        return False
    parts = [p.strip().lower() for p in combo.split("+") if p.strip()]
    if not parts:
        return False
    flags = 0
    keycode: Optional[int] = None
    for p in parts:
        if p in _VOICE_MODS:
            flags |= _VOICE_MODS[p]
        elif p in _VOICE_KEYS:
            keycode = _VOICE_KEYS[p]
        else:
            print(f"[viewer] hotkey unknown token: {p}", flush=True)
            return False
    if keycode is None:
        return False
    try:
        down = CGEventCreateKeyboardEvent(None, keycode, True)
        up = CGEventCreateKeyboardEvent(None, keycode, False)
        if flags:
            CGEventSetFlags(down, flags)
            CGEventSetFlags(up, flags)
        CGEventPost(kCGHIDEventTap, down)
        CGEventPost(kCGHIDEventTap, up)
        print(f"[viewer] hotkey fired: {combo} (kc={keycode} flags={flags:#x})",
              flush=True)
        return True
    except Exception as e:
        print(f"[viewer] hotkey failed: {e}", flush=True)
        return False


def _tap_wispr_applescript_fn() -> None:
    """AppleScript 'key code 63' — the Fn key via System Events.
    Works in some apps but not all because Wispr uses IOKit HID."""
    script = 'tell application "System Events" to key code 63'
    try:
        subprocess.run(["osascript", "-e", script],
                       check=False, timeout=1.0)
        print("[viewer] wispr applescript tapped (key code 63)", flush=True)
    except Exception as e:
        print(f"[viewer] applescript tap failed: {e}", flush=True)


def _tap_apple_dictation() -> None:
    """Trigger built-in macOS Dictation. User must have set the dictation
    shortcut to 'Press Control key twice' in System Settings → Keyboard →
    Dictation. We tap left Control (key code 59) twice with a short gap.

    We also raise a `dictationPending` flag and try to activate a browser
    window + focus the notes textarea, so dictation lands in the page.
    """
    global _dictation_pending
    # Tell the browser to focus the notes textarea.
    with _state_lock:
        _dictation_pending = True
    # Best-effort: activate Arc (Jack's default). Falls through silently.
    try:
        subprocess.run(
            ["osascript", "-e", 'tell application "Arc" to activate'],
            check=False, timeout=0.8,
        )
    except Exception:
        pass
    # Small delay so the browser has time to focus before the trigger fires.
    time.sleep(0.25)
    if not _QUARTZ_OK:
        print("[viewer] apple dictation skipped: Quartz unavailable",
              flush=True)
        return
    try:
        for _ in range(2):
            d = CGEventCreateKeyboardEvent(None, 59, True)
            u = CGEventCreateKeyboardEvent(None, 59, False)
            CGEventPost(kCGHIDEventTap, d)
            CGEventPost(kCGHIDEventTap, u)
            time.sleep(0.08)
        print("[viewer] apple dictation: double-tapped Control", flush=True)
    except Exception as e:
        print(f"[viewer] apple dictation failed: {e}", flush=True)


def _double_tap_cgevent(keycode: int, fn_flag: bool = False,
                        gap_s: float = 0.08) -> None:
    """Rapid double-tap via CGEvent — triggers any app set up with
    'double-tap to toggle' for that key."""
    if not _QUARTZ_OK:
        return
    for _ in range(2):
        try:
            down = CGEventCreateKeyboardEvent(None, keycode, True)
            up   = CGEventCreateKeyboardEvent(None, keycode, False)
            if fn_flag:
                CGEventSetFlags(down, kCGEventFlagMaskSecondaryFn)
                CGEventSetFlags(up,   kCGEventFlagMaskSecondaryFn)
            CGEventPost(kCGHIDEventTap, down)
            CGEventPost(kCGHIDEventTap, up)
        except Exception as e:
            print(f"[viewer] double-tap failed: {e}", flush=True)
            return
        time.sleep(gap_s)
    print(f"[viewer] wispr double-tap (key={keycode} fn={fn_flag})", flush=True)


def _tap_wispr_menu_click() -> None:
    """Toggle Wispr Flow by clicking its menu bar icon via System Events.
    Works without any keyboard-injection layer — needs Accessibility
    permission for the process running handsfree. Universal fallback."""
    script = '''
tell application "System Events"
  if exists (process "Wispr Flow") then
    tell process "Wispr Flow"
      try
        click menu bar item 1 of menu bar 2
        return "clicked menu bar 2 item 1"
      on error
        try
          click menu bar item 1 of menu bar 1
          return "clicked menu bar 1 item 1"
        end try
      end try
    end tell
  else
    return "Wispr Flow not running"
  end if
end tell
'''
    try:
        r = subprocess.run(["osascript", "-e", script],
                           check=False, timeout=1.5,
                           capture_output=True, text=True)
        out = (r.stdout or r.stderr or "").strip()
        print(f"[viewer] wispr menu click: {out}", flush=True)
    except Exception as e:
        print(f"[viewer] wispr menu click failed: {e}", flush=True)


def _triple_tap_cgevent(keycode: int, fn_flag: bool = False,
                        gap_s: float = 0.08) -> None:
    if not _QUARTZ_OK:
        return
    for _ in range(3):
        try:
            d = CGEventCreateKeyboardEvent(None, keycode, True)
            u = CGEventCreateKeyboardEvent(None, keycode, False)
            if fn_flag:
                CGEventSetFlags(d, kCGEventFlagMaskSecondaryFn)
                CGEventSetFlags(u, kCGEventFlagMaskSecondaryFn)
            CGEventPost(kCGHIDEventTap, d)
            CGEventPost(kCGHIDEventTap, u)
        except Exception as e:
            print(f"[viewer] triple-tap failed: {e}", flush=True)
            return
        time.sleep(gap_s)
    print(f"[viewer] wispr triple-tap (key={keycode} fn={fn_flag})", flush=True)


def _tap_escape_cgevent() -> None:
    """Tap Escape. Many Mac apps treat Escape as 'cancel current action',
    which may stop an already-active Wispr recording."""
    if not _QUARTZ_OK:
        return
    try:
        d = CGEventCreateKeyboardEvent(None, 53, True)   # 53 = Escape
        u = CGEventCreateKeyboardEvent(None, 53, False)
        CGEventPost(kCGHIDEventTap, d)
        CGEventPost(kCGHIDEventTap, u)
        print("[viewer] escape tap", flush=True)
    except Exception as e:
        print(f"[viewer] escape tap failed: {e}", flush=True)


def _nuclear_stop_wispr() -> None:
    """Fire every plausible 'stop' signal at Wispr in sequence.
    Hypothesis: Wispr latches on Fn key-down at a layer the synthetic Fn
    key-up can't reach, so we spam multiple distinct stop channels. At
    least one should be heard. Runs on a background thread so the capture
    loop isn't stalled."""
    def _fire():
        # 1. Make sure any held Fn is released at the CGEvent layer.
        try:
            if _QUARTZ_OK:
                u = CGEventCreateKeyboardEvent(None, 63, False)
                CGEventSetFlags(u, kCGEventFlagMaskSecondaryFn)
                CGEventPost(kCGHIDEventTap, u)
        except Exception:
            pass
        time.sleep(0.05)
        # 2. A fresh Fn tap (in case Fn is a toggle).
        _tap_wispr_cgevent(63, fn_flag=True)
        time.sleep(0.08)
        # 3. A double-tap Fn (some Wispr configs use double-tap to toggle).
        _double_tap_cgevent(63, fn_flag=True)
        time.sleep(0.08)
        # 4. F19 tap (alternate hotkey some users bind).
        _tap_wispr_cgevent(80, fn_flag=False)
        time.sleep(0.08)
        # 5. Escape — many apps treat it as 'cancel/stop'.
        _tap_escape_cgevent()
        time.sleep(0.08)
        # 6. AppleScript click the Wispr menu bar icon.
        _tap_wispr_menu_click()
        print("[viewer] 💣 nuclear stop sequence complete", flush=True)
    threading.Thread(target=_fire, daemon=True).start()
    print("[viewer] 💣 nuclear stop fired", flush=True)


def _hold_cgevent_for(keycode: int, fn_flag: bool, hold_s: float) -> None:
    """Press key, sleep hold_s, release key. Simulates a deliberate human hold."""
    if not _QUARTZ_OK:
        return
    try:
        d = CGEventCreateKeyboardEvent(None, keycode, True)
        if fn_flag:
            CGEventSetFlags(d, kCGEventFlagMaskSecondaryFn)
        CGEventPost(kCGHIDEventTap, d)
        time.sleep(hold_s)
        u = CGEventCreateKeyboardEvent(None, keycode, False)
        if fn_flag:
            CGEventSetFlags(u, kCGEventFlagMaskSecondaryFn)
        CGEventPost(kCGHIDEventTap, u)
        print(f"[viewer] wispr hold {hold_s}s (key={keycode} fn={fn_flag})",
              flush=True)
    except Exception as e:
        print(f"[viewer] hold-for failed: {e}", flush=True)


def _tap_wispr_hotkey() -> None:
    """Dispatch on `_wispr_method`. 'all' fires each variant on a
    background thread with short gaps so we don't stall the capture loop."""
    method = _wispr_method
    if method == "off":
        return
    if method == "applescript_fn":
        _tap_wispr_applescript_fn()
    elif method == "cgevent_f19":
        _tap_wispr_cgevent(80, fn_flag=False)
    elif method == "cgevent_fn":
        _tap_wispr_cgevent(63, fn_flag=True)
    elif method == "double_tap_f19":
        _double_tap_cgevent(80, fn_flag=False)
    elif method == "double_tap_fn":
        _double_tap_cgevent(63, fn_flag=True)
    elif method == "triple_tap_fn":
        _triple_tap_cgevent(63, fn_flag=True)
    elif method == "menu_click":
        _tap_wispr_menu_click()
    elif method == "apple_dictation":
        _tap_apple_dictation()
    elif method == "all":
        def _fire_all():
            _tap_wispr_cgevent(80, fn_flag=False)
            time.sleep(0.2)
            _double_tap_cgevent(80, fn_flag=False)
            time.sleep(0.2)
            _tap_wispr_cgevent(63, fn_flag=True)
            time.sleep(0.2)
            _double_tap_cgevent(63, fn_flag=True)
            time.sleep(0.2)
            _tap_wispr_applescript_fn()
            time.sleep(0.2)
            _tap_wispr_menu_click()
        threading.Thread(target=_fire_all, daemon=True).start()
    else:
        _tap_wispr_cgevent(WISPR_KEYCODE, fn_flag=WISPR_USE_FN_FLAG)


_held_keycode: Optional[int] = None  # currently-held wispr key, if any


def _press_wispr_key_down() -> None:
    """Press (but don't release) the wispr key for the active method.
    Only cgevent methods support true hold — other methods fall back to tap."""
    global _held_keycode
    method = _wispr_method
    if method == "off":
        return
    if method == "cgevent_f19":
        kc, fn = 80, False
    elif method == "cgevent_fn":
        kc, fn = 63, True
    else:
        # Non-cgevent methods can't truly hold — tap once on entry.
        _tap_wispr_hotkey()
        return
    if not _QUARTZ_OK:
        return
    try:
        down = CGEventCreateKeyboardEvent(None, kc, True)
        if fn:
            CGEventSetFlags(down, kCGEventFlagMaskSecondaryFn)
        CGEventPost(kCGHIDEventTap, down)
        _held_keycode = kc
        print(f"[viewer] wispr HOLD down key={kc} fn={fn}", flush=True)
    except Exception as e:
        print(f"[viewer] wispr hold-down failed: {e}", flush=True)


def _release_wispr_key_up() -> None:
    """Release the held wispr key. No-op if nothing is held.
    If _wispr_release_extra_tap is on, we also fire a fresh tap after
    the key-up — useful when the app treats the keydown as a toggle and
    ignores the key-up, so we need a second tap to stop it."""
    global _held_keycode
    if _held_keycode is None or not _QUARTZ_OK:
        return
    try:
        kc = _held_keycode
        fn = (kc == 63)
        up = CGEventCreateKeyboardEvent(None, kc, False)
        if fn:
            CGEventSetFlags(up, kCGEventFlagMaskSecondaryFn)
        CGEventPost(kCGHIDEventTap, up)
        print(f"[viewer] wispr HOLD up   key={kc}", flush=True)
        if _wispr_release_extra_tap:
            # Brief gap so the app processes the release before the new tap.
            time.sleep(0.05)
            _tap_wispr_cgevent(kc, fn_flag=fn)
            print("[viewer] wispr HOLD + extra tap on release", flush=True)
        if _wispr_release_nuclear:
            # Synthetic Fn key-up is often ignored — spam every plausible
            # stop signal so at least one takes.
            _nuclear_stop_wispr()
    except Exception as e:
        print(f"[viewer] wispr hold-up failed: {e}", flush=True)
    finally:
        _held_keycode = None


def _hold_wispr_key() -> None:
    _press_wispr_key_down()


def _release_wispr_key() -> None:
    _release_wispr_key_up()


def _set_system_mute(mute: bool) -> None:
    """Mute / unmute system audio output via osascript. Fired in a
    daemon thread so the capture loop never blocks on osascript."""
    def _go():
        try:
            arg = ("set volume with output muted" if mute
                   else "set volume without output muted")
            subprocess.run(
                ["osascript", "-e", arg],
                check=False, timeout=2.0,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass
    threading.Thread(target=_go, daemon=True).start()


def _set_master(on: bool, source: str = "") -> None:
    """Master on/off — drives _system_enabled and _cursor_enabled together.
    When turning on, re-calibrates cursor so head position is re-centered.
    Auto-mutes system audio in timeout (off), unmutes on resume."""
    global _system_enabled, _cursor_enabled, _cursor_calibrated
    global _cursor_calib_start, _finger_center, _gaze_center
    was_on = _system_enabled
    _system_enabled = bool(on)
    _cursor_enabled = bool(on)
    if on:
        _cursor_calibrated = False
        _cursor_calib_start = time.time()
        _finger_center = None
        _gaze_center = None
        # Resume from timeout — unmute (only if we toggled, not on a
        # boot-time call where state didn't change).
        if not was_on:
            _set_system_mute(False)
    else:
        # Ensure any held dictation key isn't stuck after a hard off.
        _release_wispr_key_up()
        # Release any mouse button held via mouth-hold experiment.
        _mouth_hold_release()
        # Release any mouse button held via peace ✌️ hold-to-drag.
        _peace_hold_force_release()
        # Timeout = silence: mute system audio so videos / music pause
        # being audible while the user is heads-down.
        if was_on:
            _set_system_mute(True)
    print(f"[viewer] MASTER {'ON' if on else 'OFF'}"
          + (f" ({source})" if source else ""), flush=True)


def _fire_swipe_action(direction: str) -> None:
    """Swipe right → next desktop (Ctrl+Right). Swipe left → prev (Ctrl+Left).

    Requires System Settings → Keyboard Shortcuts → Mission Control →
    'Move left/right a space' to be enabled (they are by default).
    """
    key_code = 124 if direction == "right" else 123  # Right / Left arrow
    script = (
        'tell application "System Events" to '
        f'key code {key_code} using {{control down}}'
    )
    try:
        subprocess.run(
            ["osascript", "-e", script], check=False, timeout=1.0,
        )
        print(f"[viewer] swipe {direction} → ctrl+{'→' if direction=='right' else '←'}",
              flush=True)
    except Exception as e:
        print(f"[viewer] swipe action failed: {e}", flush=True)


def _split_hands_left_right(hand_landmarks_list):
    """Return (left_y, right_y) — wrist Y for the hand on each side of the
    (mirror-flipped) frame. Either can be None when a hand is missing or
    lowered below the play threshold."""
    lx, ly, rx, ry = _split_hands_xy(hand_landmarks_list)
    return ly, ry


def _split_hands_xy(hand_landmarks_list):
    """Return (left_x, left_y, right_x, right_y) — wrist XY for the hand
    on each side of the (mirror-flipped) frame. Any entry is None when
    that hand is missing or lowered below the play threshold."""
    raised = []
    for hand in hand_landmarks_list:
        x, y = hand[0].x, hand[0].y
        if y < HAND_PLAY_THRESHOLD_Y:
            raised.append((x, y))
    if not raised:
        return None, None, None, None
    if len(raised) == 1:
        x, y = raised[0]
        return (x, y, None, None) if x < 0.5 else (None, None, x, y)
    raised.sort(key=lambda xy: xy[0])
    lx, ly = raised[0]
    rx, ry = raised[1]
    return lx, ly, rx, ry


def _update_motion(face_nose, hand_wrists) -> None:
    """Rough "how much are you moving" score, 0..1, EMA-smoothed."""
    global _motion_val, _prev_face_nose, _prev_hand_wrists
    instant = 0.0
    samples = 0
    if face_nose is not None and _prev_face_nose is not None:
        dx = face_nose[0] - _prev_face_nose[0]
        dy = face_nose[1] - _prev_face_nose[1]
        instant += math.hypot(dx, dy) * 6.0
        samples += 1
    for cur, prev in zip(hand_wrists, _prev_hand_wrists):
        dx = cur[0] - prev[0]
        dy = cur[1] - prev[1]
        instant += math.hypot(dx, dy) * 3.5
        samples += 1
    if samples > 0:
        instant = min(1.0, instant / samples)
    else:
        instant = 0.0
    _motion_val = (1 - MOTION_EMA_ALPHA) * _motion_val + MOTION_EMA_ALPHA * instant
    _prev_face_nose = face_nose
    _prev_hand_wrists = hand_wrists


def _update_tab_swipe(hands_list, now: float) -> Optional[str]:
    """Two hands above the shoulder line. One holds still (anchor), the
    other sweeps sideways. Returns "next" or "prev", or None.

    Gate is loose on direction so either hand can be the sweeper — user
    just lifts both and moves one. Anchor-stillness check guards against
    two-hand scroll (both hands move together)."""
    global _last_tab_swipe_at
    if not _tab_swipe_enabled or len(hands_list) != 2:
        _tab_swipe_hist.clear()
        return None
    w0 = hands_list[0][0]
    w1 = hands_list[1][0]
    if w0.y > TAB_SWIPE_Y_MAX or w1.y > TAB_SWIPE_Y_MAX:
        _tab_swipe_hist.clear()
        return None
    # Sort L→R so anchor/sweeper labels are stable across frames.
    if w0.x < w1.x:
        lx, rx = float(w0.x), float(w1.x)
    else:
        lx, rx = float(w1.x), float(w0.x)
    _tab_swipe_hist.append((now, lx, rx))
    while _tab_swipe_hist and now - _tab_swipe_hist[0][0] > TAB_SWIPE_WINDOW_S:
        _tab_swipe_hist.popleft()
    if len(_tab_swipe_hist) < 5:
        return None
    dl = _tab_swipe_hist[-1][1] - _tab_swipe_hist[0][1]
    dr = _tab_swipe_hist[-1][2] - _tab_swipe_hist[0][2]
    direction: Optional[str] = None
    # Positive dx = hand moved toward the right side of the image. Because
    # the frame is mirrored for the UI, image-right = user's right hand
    # moving leftward in real space. Pick mapping that matches macOS tab
    # switch: sweep to the right → next tab, left → prev.
    if abs(dr) > TAB_SWIPE_MIN_DX and abs(dl) < TAB_SWIPE_ANCHOR_MAX_DX:
        direction = "next" if dr > 0 else "prev"
    elif abs(dl) > TAB_SWIPE_MIN_DX and abs(dr) < TAB_SWIPE_ANCHOR_MAX_DX:
        direction = "next" if dl > 0 else "prev"
    if direction is None:
        return None
    if now - _last_tab_swipe_at < TAB_SWIPE_COOLDOWN_S:
        return None
    _last_tab_swipe_at = now
    _tab_swipe_hist.clear()
    return direction


def _update_fist_scroll(hands_list, now: float) -> tuple[int, int]:
    """Single-hand fist scroll. Close one hand → wrist Y/X motion scrolls.
    Returns (vertical, horizontal) scroll amounts. Sets `_scroll_active`
    so the cursor freezes while scrolling.

    Use any hand that's currently a fist. The horizontal channel is
    softer (gain/3) since horizontal scroll is noisier.
    """
    global _fist_scroll_prev, _fist_scroll_accum_y, _fist_scroll_accum_x
    global _scroll_active
    if _jam_mode or not hands_list:
        _fist_scroll_prev = None
        _fist_scroll_accum_y = 0.0
        _fist_scroll_accum_x = 0.0
        _scroll_active = False
        return 0, 0

    fist_hand = None
    for h in hands_list:
        if _is_fist(h):
            fist_hand = h
            break
    if fist_hand is None:
        _fist_scroll_prev = None
        _fist_scroll_accum_y = 0.0
        _fist_scroll_accum_x = 0.0
        _scroll_active = False
        return 0, 0

    _scroll_active = True
    wrist = fist_hand[0]
    cur = (float(wrist.y), float(wrist.x))
    if _fist_scroll_prev is None:
        _fist_scroll_prev = cur
        return 0, 0
    dy = cur[0] - _fist_scroll_prev[0]
    dx = cur[1] - _fist_scroll_prev[1]
    _fist_scroll_prev = cur

    gain = BASE_SCROLL_GAIN * _scroll_sens_mult
    v_amt = 0
    h_amt = 0
    if abs(dy) >= SCROLL_FRAME_DEAD:
        _fist_scroll_accum_y += -dy * gain
        if abs(_fist_scroll_accum_y) >= 1.0:
            v_amt = int(_fist_scroll_accum_y)
            _fist_scroll_accum_y -= v_amt
    if abs(dx) >= SCROLL_FRAME_DEAD:
        # Horizontal channel was previously gain/3 (too soft). Figma's
        # horizontal pan needs comparable throw to vertical to feel right,
        # so we use gain/1.5 — still slightly damped vs vertical because
        # wrist x-jitter is noisier than y-jitter, but close enough that
        # the diagonal grab-pan feels intuitive.
        _fist_scroll_accum_x += -dx * (gain / 1.5)
        if abs(_fist_scroll_accum_x) >= 1.0:
            h_amt = int(_fist_scroll_accum_x)
            _fist_scroll_accum_x -= h_amt
    return v_amt, h_amt


def _update_head_scroll(hands_list, face_nose, blendshapes,
                        now: float, gate: str) -> int:
    """Head-pitch scroll. Continuous vertical scroll based on how far
    your nose has moved from the 'center' captured when the gate first
    engaged.

    gate: 'lefthand' → left hand raised above threshold
          'mouth'    → mouth-open hold (jawOpen > threshold)
          'always'   → no gate (head pitch always scrolls — careful)

    Returns signed scroll amount; negative = down, positive = up.
    Chin up (nose y decreasing in image coords) → scroll up, and vice versa.
    """
    global _head_scroll_baseline_y, _head_scroll_accum, _scroll_active
    if _jam_mode or face_nose is None:
        _head_scroll_baseline_y = None
        _head_scroll_accum = 0.0
        _scroll_active = False
        return 0
    # --- gate check ---
    active = False
    if gate == "lefthand":
        lx, ly, _rx, _ry = _split_hands_xy(hands_list)
        active = (ly is not None and ly < HAND_THRESHOLD)
    elif gate == "mouth":
        if blendshapes:
            for b in blendshapes:
                if b.category_name == "jawOpen":
                    active = float(b.score) > MOUTH_CLICK_THRESHOLD
                    break
    elif gate == "always":
        active = True
    if not active:
        _head_scroll_baseline_y = None
        _head_scroll_accum = 0.0
        _scroll_active = False
        return 0
    _scroll_active = True
    ny = float(face_nose[1])
    if _head_scroll_baseline_y is None:
        _head_scroll_baseline_y = ny
        _head_scroll_accum = 0.0
        return 0
    # Dead-zone near baseline so resting head position doesn't drift.
    delta = ny - _head_scroll_baseline_y
    if abs(delta) < HEAD_SCROLL_DEADBAND:
        return 0
    # Slowly re-center baseline so the user can "hold" a direction
    # without running out of neck range (drift correction).
    _head_scroll_baseline_y += delta * 0.003
    gain = BASE_SCROLL_GAIN * _scroll_sens_mult
    # Nose y decreases when chin goes up → scroll up (positive amount)
    per_frame = -delta * gain * 0.8
    _head_scroll_accum += per_frame
    amt = 0
    if abs(_head_scroll_accum) >= 1.0:
        amt = int(_head_scroll_accum)
        _head_scroll_accum -= amt
    return amt


def _update_brow_scroll(blendshapes, now: float) -> int:
    """Scroll via eyebrow blendshapes.
    browInnerUp → scroll up, browDownLeft/Right (average) → scroll down.
    Proportional to blendshape intensity above a deadband."""
    global _brow_scroll_accum, _scroll_active
    if _jam_mode or not blendshapes:
        _brow_scroll_accum = 0.0
        _scroll_active = False
        return 0
    up = 0.0; down_l = 0.0; down_r = 0.0
    for b in blendshapes:
        n = b.category_name
        if n == "browInnerUp":
            up = float(b.score)
        elif n == "browDownLeft":
            down_l = float(b.score)
        elif n == "browDownRight":
            down_r = float(b.score)
    down = (down_l + down_r) * 0.5
    signal = 0.0
    if up > BROW_SCROLL_THRESHOLD and up > down:
        signal = (up - BROW_SCROLL_THRESHOLD)
    elif down > BROW_SCROLL_THRESHOLD and down > up:
        signal = -(down - BROW_SCROLL_THRESHOLD)
    if signal == 0.0:
        _brow_scroll_accum = 0.0
        _scroll_active = False
        return 0
    _scroll_active = True
    gain = BASE_SCROLL_GAIN * _scroll_sens_mult
    _brow_scroll_accum += signal * gain * 0.08
    amt = 0
    if abs(_brow_scroll_accum) >= 1.0:
        amt = int(_brow_scroll_accum)
        _brow_scroll_accum -= amt
    return amt


def _update_two_hand_scroll(hands_list, now: float) -> int:
    """Two hands visible → scroll mode. Returns signed scroll amount
    (positive = up, negative = down, 0 = no scroll this frame).
    Also flips `_scroll_active` so the cursor can freeze."""
    global _scroll_prev_y, _scroll_active, _scroll_accum
    if _jam_mode or len(hands_list) < 2:
        _scroll_prev_y = None
        _scroll_active = False
        _scroll_accum = 0.0
        return 0
    y0 = hands_list[0][0].y
    y1 = hands_list[1][0].y
    if y0 > SCROLL_ZONE_Y_MAX or y1 > SCROLL_ZONE_Y_MAX:
        _scroll_prev_y = None
        _scroll_active = False
        _scroll_accum = 0.0
        return 0
    _scroll_active = True
    avg = (y0 + y1) * 0.5
    if _scroll_prev_y is None:
        _scroll_prev_y = avg
        return 0
    dy = avg - _scroll_prev_y
    _scroll_prev_y = avg
    if abs(dy) < SCROLL_FRAME_DEAD:
        return 0
    gain = BASE_SCROLL_GAIN * _scroll_sens_mult
    # hands moving up (dy < 0) → scroll up (positive pyautogui amount)
    _scroll_accum += -dy * gain
    if abs(_scroll_accum) < 1.0:
        return 0
    amount = int(_scroll_accum)
    _scroll_accum -= amount
    return amount


def _pick_right_hand(hands_lm_list):
    """Return the hand landmark list for the rightmost hand (mirror-flipped
    frame → user's right). Falls back to any visible hand so pinch works
    even with a single hand in view."""
    if not hands_lm_list:
        return None
    return max(hands_lm_list, key=lambda h: h[0].x)


def _target_from_head(face_matrix) -> Optional[tuple]:
    """Head pose → screen (target_x, target_y) or None."""
    if face_matrix is None:
        return None
    yaw, pitch = _matrix_to_yaw_pitch(face_matrix)
    # The matrix decomposition in this file labels the axes as (yaw, pitch)
    # but empirically `yaw` drives vertical and `pitch` drives horizontal
    # cursor movement on Jack's camera. Swap them to match reality.
    d_h = pitch - _pitch_center
    d_v = yaw - _yaw_center
    if abs(d_h) < CURSOR_DEAD_ZONE_DEG:
        d_h = 0.0
    if abs(d_v) < CURSOR_DEAD_ZONE_DEG:
        d_v = 0.0
    mid_x = (_virt_x0 + _virt_x1) * 0.5
    mid_y = (_virt_y0 + _virt_y1) * 0.5
    tx = mid_x + d_h * CURSOR_SENSITIVITY_X * _cursor_sens
    ty = mid_y - d_v * CURSOR_SENSITIVITY_Y * _cursor_sens
    return tx, ty


def _target_from_finger(hands_lm_list) -> Optional[tuple]:
    """Right-hand index fingertip (landmark 8) → screen."""
    hand = _pick_right_hand(hands_lm_list)
    if hand is None or _finger_center is None:
        return None
    fx, fy = hand[8].x, hand[8].y
    cx, cy = _finger_center
    dx = (fx - cx) * FINGER_GAIN_X * _cursor_sens
    dy = (fy - cy) * FINGER_GAIN_Y * _cursor_sens
    mid_x = (_virt_x0 + _virt_x1) * 0.5
    mid_y = (_virt_y0 + _virt_y1) * 0.5
    tx = mid_x + dx * _virt_w
    ty = mid_y + dy * _virt_h
    return tx, ty


def _iris_center(face_landmarks) -> Optional[tuple]:
    """Average of left iris (468) and right iris (473) landmark centers."""
    if face_landmarks is None or len(face_landmarks) < 478:
        return None
    a = face_landmarks[468]
    b = face_landmarks[473]
    return ((a.x + b.x) * 0.5, (a.y + b.y) * 0.5)


def _target_from_gaze(face_landmarks) -> Optional[tuple]:
    iris = _iris_center(face_landmarks)
    if iris is None or _gaze_center is None:
        return None
    gx, gy = iris
    cx, cy = _gaze_center
    dx = (gx - cx) * GAZE_GAIN_X * _cursor_sens
    dy = (gy - cy) * GAZE_GAIN_Y * _cursor_sens
    mid_x = (_virt_x0 + _virt_x1) * 0.5
    mid_y = (_virt_y0 + _virt_y1) * 0.5
    tx = mid_x + dx * _virt_w
    ty = mid_y + dy * _virt_h
    return tx, ty


def _detect_brow_click(blendshapes, now: float) -> bool:
    global _last_brow_click_at
    if not blendshapes:
        return False
    v = 0.0
    for b in blendshapes:
        if b.category_name in ("browInnerUp", "browOuterUpLeft",
                               "browOuterUpRight"):
            v = max(v, float(b.score))
    if v > BROW_CLICK_THRESHOLD:
        if now - _last_brow_click_at > BROW_CLICK_COOLDOWN_S:
            _last_brow_click_at = now
            return True
    return False


def _detect_wink_click(blendshapes, now: float) -> bool:
    """Edge-triggered wink: one eye closed, the other open. Accepts either
    eye so the user doesn't have to think about which to use."""
    global _last_wink_click_at, _wink_armed
    if not blendshapes:
        return False
    left = right = 0.0
    for b in blendshapes:
        if b.category_name == "eyeBlinkLeft":
            left = float(b.score)
        elif b.category_name == "eyeBlinkRight":
            right = float(b.score)
    hi, lo = max(left, right), min(left, right)
    winking = hi > WINK_CLOSED_THRESHOLD and lo < WINK_OPEN_THRESHOLD
    fired = False
    if winking and _wink_armed:
        if now - _last_wink_click_at > WINK_COOLDOWN_S:
            _last_wink_click_at = now
            _wink_armed = False
            fired = True
    elif not winking and lo < WINK_OPEN_THRESHOLD:
        # Asymmetry gone (the formerly-open eye is still open enough) → rearm.
        # Using `lo` here means we don't need BOTH eyes fully open, just the
        # steady-open one — friendlier for frequent blinkers.
        _wink_armed = True
    return fired


def _detect_right_wink_click(blendshapes, now: float) -> bool:
    """Edge-triggered right-eye-only wink."""
    global _last_right_wink_click_at, _right_wink_armed
    if not blendshapes:
        return False
    left = right = 0.0
    for b in blendshapes:
        if b.category_name == "eyeBlinkLeft":
            left = float(b.score)
        elif b.category_name == "eyeBlinkRight":
            right = float(b.score)
    winking = right > WINK_CLOSED_THRESHOLD and left < WINK_OPEN_THRESHOLD
    fired = False
    if winking and _right_wink_armed:
        if now - _last_right_wink_click_at > WINK_COOLDOWN_S:
            _last_right_wink_click_at = now
            _right_wink_armed = False
            fired = True
    elif not winking and right < WINK_OPEN_THRESHOLD:
        _right_wink_armed = True
    return fired


def _detect_blink_click(blendshapes, now: float) -> bool:
    """Edge-triggered deliberate blink: BOTH eyes closed hard together."""
    global _last_blink_click_at, _blink_click_armed
    if not blendshapes:
        return False
    left = right = 0.0
    for b in blendshapes:
        if b.category_name == "eyeBlinkLeft":
            left = float(b.score)
        elif b.category_name == "eyeBlinkRight":
            right = float(b.score)
    both_closed = left > BLINK_CLICK_THRESHOLD and right > BLINK_CLICK_THRESHOLD
    fired = False
    if both_closed and _blink_click_armed:
        if now - _last_blink_click_at > BLINK_CLICK_COOLDOWN_S:
            _last_blink_click_at = now
            _blink_click_armed = False
            fired = True
    elif left < BLINK_CLICK_OPEN_THR and right < BLINK_CLICK_OPEN_THR:
        _blink_click_armed = True
    return fired


def _detect_mouth_click(blendshapes, now: float) -> bool:
    """Edge-triggered mouth-open click (jawOpen blendshape)."""
    global _last_mouth_click_at, _mouth_armed
    if not blendshapes:
        return False
    jaw = 0.0
    for b in blendshapes:
        if b.category_name == "jawOpen":
            jaw = float(b.score)
            break
    fired = False
    if jaw > MOUTH_CLICK_THRESHOLD and _mouth_armed:
        if now - _last_mouth_click_at > MOUTH_CLICK_COOLDOWN_S:
            _last_mouth_click_at = now
            _mouth_armed = False
            fired = True
    elif jaw < MOUTH_CLICK_OPEN_THR:
        _mouth_armed = True
    return fired


def _update_mouth_hold(blendshapes, now: float) -> None:
    """Mouth-open = mouseDown, mouth-close = mouseUp. Gated by
    _mouth_hold_enabled and _click_method == 'mouth'. Safe to call every frame."""
    global _mouth_hold_down
    if not blendshapes:
        return
    jaw = 0.0
    for b in blendshapes:
        if b.category_name == "jawOpen":
            jaw = float(b.score)
            break
    if jaw > MOUTH_CLICK_THRESHOLD and not _mouth_hold_down:
        try:
            pyautogui.mouseDown(_pause=False)
            _mouth_hold_down = True
            print("[viewer] mouth-hold → mouseDown", flush=True)
        except Exception as e:
            print(f"[viewer] mouseDown failed: {e}", flush=True)
    elif jaw < MOUTH_CLICK_OPEN_THR and _mouth_hold_down:
        try:
            pyautogui.mouseUp(_pause=False)
            _mouth_hold_down = False
            print("[viewer] mouth-hold → mouseUp", flush=True)
        except Exception as e:
            print(f"[viewer] mouseUp failed: {e}", flush=True)


def _mouth_hold_release() -> None:
    """Release any held mouse button — called on disable or master-off."""
    global _mouth_hold_down
    if _mouth_hold_down:
        try:
            pyautogui.mouseUp(_pause=False)
        except Exception:
            pass
        _mouth_hold_down = False


def _detect_smile_rightclick(blendshapes, now: float) -> bool:
    global _last_smile_click_at, _smile_click_armed
    if not blendshapes:
        return False
    v = 0.0
    for b in blendshapes:
        if b.category_name in ("mouthSmileLeft", "mouthSmileRight"):
            v = max(v, float(b.score))
    fired = False
    if v > SMILE_CLICK_THRESHOLD and _smile_click_armed:
        if now - _last_smile_click_at > SMILE_CLICK_COOLDOWN_S:
            _last_smile_click_at = now
            _smile_click_armed = False
            fired = True
    elif v < SMILE_CLICK_OPEN_THR:
        _smile_click_armed = True
    return fired


def _detect_pucker_rightclick(blendshapes, now: float) -> bool:
    global _last_pucker_click_at, _pucker_click_armed
    if not blendshapes:
        return False
    v = 0.0
    for b in blendshapes:
        if b.category_name == "mouthPucker":
            v = float(b.score)
            break
    fired = False
    if v > PUCKER_CLICK_THRESHOLD and _pucker_click_armed:
        if now - _last_pucker_click_at > PUCKER_CLICK_COOLDOWN_S:
            _last_pucker_click_at = now
            _pucker_click_armed = False
            fired = True
    elif v < PUCKER_CLICK_OPEN_THR:
        _pucker_click_armed = True
    return fired


def _detect_furrow_rightclick(blendshapes, now: float) -> bool:
    global _last_furrow_click_at, _furrow_click_armed
    if not blendshapes:
        return False
    v = 0.0
    for b in blendshapes:
        if b.category_name in ("browDownLeft", "browDownRight"):
            v = max(v, float(b.score))
    fired = False
    if v > FURROW_CLICK_THRESHOLD and _furrow_click_armed:
        if now - _last_furrow_click_at > FURROW_CLICK_COOLDOWN_S:
            _last_furrow_click_at = now
            _furrow_click_armed = False
            fired = True
    elif v < FURROW_CLICK_OPEN_THR:
        _furrow_click_armed = True
    return fired


def _detect_right_click(blendshapes, now: float) -> bool:
    if _right_click_method == "smile":
        return _detect_smile_rightclick(blendshapes, now)
    if _right_click_method == "pucker":
        return _detect_pucker_rightclick(blendshapes, now)
    if _right_click_method == "furrow":
        return _detect_furrow_rightclick(blendshapes, now)
    return False


def _detect_pinch_click(hands_lm_list, now: float) -> bool:
    """Edge-triggered pinch: thumb tip (4) to index tip (8) closes."""
    global _pinch_was_closed, _last_pinch_click_at
    hand = _pick_right_hand(hands_lm_list)
    if hand is None:
        _pinch_was_closed = False
        return False
    dx = hand[4].x - hand[8].x
    dy = hand[4].y - hand[8].y
    d = math.hypot(dx, dy)
    fired = False
    if not _pinch_was_closed and d < PINCH_CLOSE_THRESHOLD:
        _pinch_was_closed = True
        if now - _last_pinch_click_at > PINCH_COOLDOWN_S:
            _last_pinch_click_at = now
            fired = True
    elif _pinch_was_closed and d > PINCH_OPEN_THRESHOLD:
        _pinch_was_closed = False
    return fired


def _detect_t_gesture(hands_lm_list, now: float) -> bool:
    """Edge-triggered T-shape: one hand vertical (fingers up), the other
    hand horizontal, with horizontal hand's midpoint near vertical hand's
    fingertips. Returns True once per gesture (armed/unarmed like pinch)."""
    global _t_gesture_armed, _t_last_trigger_at
    if not hands_lm_list or len(hands_lm_list) < 2:
        _t_gesture_armed = True
        return False
    # Use first two hands seen.
    h1, h2 = hands_lm_list[0], hands_lm_list[1]

    def axis(h):
        # wrist (0) → middle fingertip (12)
        dx = h[12].x - h[0].x
        dy = h[12].y - h[0].y
        return dx, dy

    def is_vertical(dx, dy):
        # fingers up → dy negative (y grows downward), |dy| >> |dx|
        return dy < -0.05 and abs(dy) > abs(dx) * 1.7

    def is_horizontal(dx, dy):
        return abs(dx) > 0.06 and abs(dx) > abs(dy) * 1.7

    ax1 = axis(h1); ax2 = axis(h2)
    vert = horiz = None
    if is_vertical(*ax1) and is_horizontal(*ax2):
        vert, horiz = h1, h2
    elif is_vertical(*ax2) and is_horizontal(*ax1):
        vert, horiz = h2, h1
    if vert is None:
        _t_gesture_armed = True
        return False
    # Horizontal hand should cross near the TOP of the vertical hand.
    # midpoint of horizontal hand's axis:
    hmx = (horiz[0].x + horiz[12].x) / 2
    hmy = (horiz[0].y + horiz[12].y) / 2
    # vertical hand's fingertip region:
    vtx = vert[12].x; vty = vert[12].y
    # distance from horizontal midpoint to vertical fingertip area
    d = math.hypot(hmx - vtx, hmy - vty)
    if d > 0.18:
        _t_gesture_armed = True
        return False
    # Now it's a T.
    if not _t_gesture_armed:
        return False
    if now - _t_last_trigger_at < T_GESTURE_COOLDOWN_S:
        return False
    _t_gesture_armed = False
    _t_last_trigger_at = now
    return True


def _detect_a_pose(hands_lm_list, now: float) -> bool:
    """Edge-triggered A-pose: both hands visible, index+middle fingertips
    of the two hands meet near each other up top, wrists spread wider
    apart and lower (forming an 'A' with the arms). Returns True once per
    pose (armed/unarmed)."""
    global _atelier_armed, _atelier_last_at
    if not hands_lm_list or len(hands_lm_list) < 2:
        _atelier_armed = True
        return False
    h1, h2 = hands_lm_list[0], hands_lm_list[1]
    # Fingertip centroid = avg of index(8) + middle(12) tips
    def tip_centroid(h):
        return ((h[8].x + h[12].x) / 2, (h[8].y + h[12].y) / 2)
    t1 = tip_centroid(h1)
    t2 = tip_centroid(h2)
    tip_dist = math.hypot(t1[0] - t2[0], t1[1] - t2[1])
    wrist_dist = abs(h1[0].x - h2[0].x)
    # A-pose conditions:
    # - fingertips close together (< 0.09)
    # - wrists meaningfully wider apart than fingertips (>= 2x, and > 0.18)
    # - fingertips above wrists (y smaller = higher on screen)
    tips_y = (t1[1] + t2[1]) / 2
    wrists_y = (h1[0].y + h2[0].y) / 2
    tips_above = tips_y < wrists_y - 0.04
    if not (tip_dist < 0.09
            and wrist_dist > max(0.18, tip_dist * 2.5)
            and tips_above):
        _atelier_armed = True
        return False
    if not _atelier_armed:
        return False
    if now - _atelier_last_at < ATELIER_COOLDOWN_S:
        return False
    _atelier_armed = False
    _atelier_last_at = now
    return True


def _update_atelier(hands_lm_list, now: float) -> Optional[str]:
    """While atelier mode is active, read two-hand geometry and drive
    Figma-friendly zoom + pan. Returns a short action label for feedback
    ('zoom_in' / 'zoom_out' / 'pan_left' / 'pan_right' / 'pan_up' /
    'pan_down'), or None.

    Zoom = change in wrist-to-wrist distance vs. baseline.
    Pan  = change in wrist-centroid position vs. baseline.
    Baselines reset after each action so gestures feel incremental."""
    global _atelier_baseline_dist, _atelier_baseline_cx, _atelier_baseline_cy
    global _atelier_last_action_at
    if not _atelier_mode:
        return None
    if not hands_lm_list or len(hands_lm_list) < 2:
        # Drop baselines so we don't snap on re-acquire.
        _atelier_baseline_dist = None
        _atelier_baseline_cx = None
        _atelier_baseline_cy = None
        return None
    h1, h2 = hands_lm_list[0], hands_lm_list[1]
    w1 = h1[0]; w2 = h2[0]
    dist = math.hypot(w1.x - w2.x, w1.y - w2.y)
    cx = (w1.x + w2.x) / 2
    cy = (w1.y + w2.y) / 2
    if _atelier_baseline_dist is None:
        _atelier_baseline_dist = dist
        _atelier_baseline_cx = cx
        _atelier_baseline_cy = cy
        return None
    if now - _atelier_last_action_at < ATELIER_ACTION_COOLDOWN_S:
        return None
    ddist = dist - _atelier_baseline_dist
    dcx   = cx   - _atelier_baseline_cx
    dcy   = cy   - _atelier_baseline_cy
    # Zoom takes priority when distance change dominates.
    if abs(ddist) > ATELIER_ZOOM_DELTA and abs(ddist) > max(abs(dcx), abs(dcy)):
        action = "zoom_in" if ddist > 0 else "zoom_out"
        _atelier_baseline_dist = dist
        _atelier_baseline_cx = cx
        _atelier_baseline_cy = cy
        _atelier_last_action_at = now
        return action
    if abs(dcx) > ATELIER_PAN_DELTA and abs(dcx) > abs(dcy):
        # Camera is mirrored in the UI. Moving image-right = user's right hand
        # goes left in screen space — but users sweep toward the direction
        # they want the canvas to travel. Flip sign so pan matches intent.
        action = "pan_left" if dcx > 0 else "pan_right"
        _atelier_baseline_dist = dist
        _atelier_baseline_cx = cx
        _atelier_baseline_cy = cy
        _atelier_last_action_at = now
        return action
    if abs(dcy) > ATELIER_PAN_DELTA:
        action = "pan_down" if dcy > 0 else "pan_up"
        _atelier_baseline_dist = dist
        _atelier_baseline_cx = cx
        _atelier_baseline_cy = cy
        _atelier_last_action_at = now
        return action
    return None


def _detect_punches(hands_lm_list, now: float) -> list:
    """Classify fast wrist motion into jab / cross / hook / uppercut.
    Returns list of dicts: {hand: 'left'|'right', type, x, y, intensity}.

    'left'/'right' refers to image-space (camera is mirrored — image-left
    is the user's right hand). The boxing UI labels punches by screen
    side so this matches the user's mental model when they look at the
    feed. Velocity = wrist motion in normalized coords per frame.
    """
    global _left_wrist_prev, _right_wrist_prev
    global _left_punch_last_at, _right_punch_last_at
    out: list = []
    if not _boxing_enabled or not hands_lm_list:
        # Drop history when no hands so we don't spike on re-acquire.
        if not hands_lm_list:
            _left_wrist_prev = None
            _right_wrist_prev = None
        return out
    # Sort by wrist x: smaller x = image-left side.
    pairs = sorted(((float(h[0].x), float(h[0].y), h) for h in hands_lm_list),
                   key=lambda p: p[0])
    left = pairs[0] if pairs else None
    right = pairs[1] if len(pairs) > 1 else None

    def _classify(prev, cur, last_at):
        if prev is None:
            return None
        if now - last_at < PUNCH_COOLDOWN_S:
            return None
        vx = cur[0] - prev[0]
        vy = cur[1] - prev[1]
        speed = math.hypot(vx, vy)
        if speed < PUNCH_VELOCITY_THR:
            return None
        # Classify by dominant axis. y grows DOWNWARD in image coords →
        # uppercut = vy strongly negative (hand moves up).
        if vy < 0 and abs(vy) > abs(vx) * 1.15:
            ptype = "uppercut"
        elif abs(vx) > abs(vy) * 1.15:
            ptype = "hook"
        else:
            ptype = "straight"  # jab/cross — distinguished by hand below
        intensity = min(1.0, speed / 0.12)
        return {
            "type": ptype,
            "x": cur[0],
            "y": cur[1],
            "intensity": round(intensity, 3),
            "speed": round(speed, 4),
        }

    if left is not None:
        evt = _classify(_left_wrist_prev, (left[0], left[1]),
                        _left_punch_last_at)
        if evt is not None:
            evt["hand"] = "left"
            # Image-left = user's right hand → cross (rear hand for orthodox).
            if evt["type"] == "straight":
                evt["type"] = "cross"
            out.append(evt)
            _left_punch_last_at = now
        _left_wrist_prev = (left[0], left[1])
    else:
        _left_wrist_prev = None

    if right is not None:
        evt = _classify(_right_wrist_prev, (right[0], right[1]),
                        _right_punch_last_at)
        if evt is not None:
            evt["hand"] = "right"
            # Image-right = user's left hand → jab (lead hand for orthodox).
            if evt["type"] == "straight":
                evt["type"] = "jab"
            out.append(evt)
            _right_punch_last_at = now
        _right_wrist_prev = (right[0], right[1])
    else:
        _right_wrist_prev = None

    return out


def _face_size_proxy(face_landmarks) -> Optional[float]:
    """Return a size-proxy for the face (outer-eye-corner span). Grows as
    the user leans toward the camera — used for head-dolly zoom."""
    if face_landmarks is None or len(face_landmarks) < 264:
        return None
    try:
        a = face_landmarks[33]   # right eye outer corner
        b = face_landmarks[263]  # left eye outer corner
        return math.hypot(a.x - b.x, a.y - b.y)
    except Exception:
        return None


def _update_atelier_head_zoom(face_landmarks, now: float) -> Optional[str]:
    """Head-dolly zoom: when face grows (user leans in) fire zoom-in;
    when it shrinks (leans back) fire zoom-out. Baseline resets after
    each action so the gesture feels incremental."""
    global _atelier_face_baseline, _atelier_face_last_at
    if not _atelier_mode:
        return None
    size = _face_size_proxy(face_landmarks)
    if size is None:
        _atelier_face_baseline = None
        return None
    if _atelier_face_baseline is None:
        _atelier_face_baseline = size
        return None
    if now - _atelier_face_last_at < ATELIER_FACE_COOLDOWN_S:
        return None
    d = size - _atelier_face_baseline
    if abs(d) < ATELIER_FACE_ZOOM_DELTA:
        return None
    action = "zoom_in" if d > 0 else "zoom_out"
    _atelier_face_baseline = size
    _atelier_face_last_at = now
    return action


def _update_atelier_pinch_grab(hands_lm_list) -> Optional[str]:
    """Pinch-and-drag inside atelier mode. Thumb tip (4) ↔ index tip (8)
    close → mouseDown. Release → mouseUp. Returns event label for feedback."""
    global _atelier_pinch_down
    if not _atelier_mode:
        if _atelier_pinch_down:
            try:
                pyautogui.mouseUp(_pause=False)
            except Exception:
                pass
            _atelier_pinch_down = False
        return None
    if not hands_lm_list:
        if _atelier_pinch_down:
            try:
                pyautogui.mouseUp(_pause=False)
            except Exception:
                pass
            _atelier_pinch_down = False
            return "drop"
        return None
    # Use closest (smallest pinch-dist) hand so either hand can grab.
    best = None
    for h in hands_lm_list:
        try:
            d = math.hypot(h[4].x - h[8].x, h[4].y - h[8].y)
            if best is None or d < best:
                best = d
        except Exception:
            continue
    if best is None:
        return None
    if not _atelier_pinch_down and best < ATELIER_PINCH_ON_THR:
        try:
            pyautogui.mouseDown(_pause=False)
            _atelier_pinch_down = True
            return "grab"
        except Exception as e:
            print(f"[viewer] atelier grab failed: {e}", flush=True)
    elif _atelier_pinch_down and best > ATELIER_PINCH_OFF_THR:
        try:
            pyautogui.mouseUp(_pause=False)
            _atelier_pinch_down = False
            return "drop"
        except Exception as e:
            print(f"[viewer] atelier drop failed: {e}", flush=True)
    return None


def _atelier_fire(action: str) -> None:
    """Turn an atelier action label into real Figma input.
    - zoom: Cmd+= / Cmd+- (works in Figma + most apps)
    - pan : mouse scroll (Figma pans on wheel scroll)
    """
    try:
        if action == "zoom_in":
            _fire_hotkey("cmd+=")
        elif action == "zoom_out":
            _fire_hotkey("cmd+-")
        elif action == "pan_left":
            pyautogui.hscroll(-8, _pause=False)
        elif action == "pan_right":
            pyautogui.hscroll(8, _pause=False)
        elif action == "pan_up":
            pyautogui.scroll(5, _pause=False)
        elif action == "pan_down":
            pyautogui.scroll(-5, _pause=False)
    except Exception as e:
        print(f"[viewer] atelier fire {action} failed: {e}", flush=True)


def _finger_extended(hand, tip_idx: int, pip_idx: int) -> bool:
    """True if the fingertip is meaningfully farther from the wrist than
    the PIP joint — a reliable extension check that works at any hand angle."""
    wrist = hand[0]
    def d(lm):
        return math.hypot(lm.x - wrist.x, lm.y - wrist.y)
    return d(hand[tip_idx]) > d(hand[pip_idx]) * 1.15


def _finger_curled(hand, tip_idx: int, pip_idx: int) -> bool:
    wrist = hand[0]
    def d(lm):
        return math.hypot(lm.x - wrist.x, lm.y - wrist.y)
    return d(hand[tip_idx]) < d(hand[pip_idx]) * 1.05


def _is_peace_sign(hand) -> bool:
    """✌️: index + middle extended, ring + pinky curled."""
    idx_ext  = _finger_extended(hand, 8, 6)
    mid_ext  = _finger_extended(hand, 12, 10)
    ring_cur = _finger_curled(hand, 16, 14)
    pink_cur = _finger_curled(hand, 20, 18)
    return idx_ext and mid_ext and ring_cur and pink_cur


def _is_thumbs_up(hand) -> bool:
    """👍: thumb tip well above wrist, all four fingers curled."""
    thumb_tip = hand[4]
    wrist = hand[0]
    # Thumb clearly above wrist (y decreases upward in image coords)
    thumb_up = (wrist.y - thumb_tip.y) > 0.10
    idx_cur  = _finger_curled(hand, 8, 6)
    mid_cur  = _finger_curled(hand, 12, 10)
    ring_cur = _finger_curled(hand, 16, 14)
    pink_cur = _finger_curled(hand, 20, 18)
    return thumb_up and idx_cur and mid_cur and ring_cur and pink_cur


def _edge_trigger_gesture(match: bool, armed_flag: str, last_at_flag: str,
                          now: float, cooldown: float = GESTURE_COOLDOWN_S) -> bool:
    """Generic edge-trigger: only fires on the transition into `match`,
    with per-gesture cooldown. Uses global state via getattr."""
    g = globals()
    armed = g[armed_flag]
    last_at = g[last_at_flag]
    if not match:
        g[armed_flag] = True
        return False
    if not armed:
        return False
    if now - last_at < cooldown:
        return False
    g[armed_flag] = False
    g[last_at_flag] = now
    return True


def _detect_peace_rclick(hands_lm_list, now: float) -> bool:
    """Legacy edge-trigger right-click; kept for callers that haven't
    migrated. Peace is now a press-and-hold gesture — see
    `_update_peace_hold`."""
    if not hands_lm_list:
        globals()['_peace_armed'] = True
        return False
    match = any(_is_peace_sign(h) for h in hands_lm_list)
    return _edge_trigger_gesture(match, '_peace_armed', '_peace_last_at', now)


def _toast(title: str, message: str = "") -> None:
    """Native macOS notification (pops outside the browser).

    Fired in a daemon thread so the capture loop never blocks on
    the osascript call (which can take 50–200ms cold).
    """
    def _go():
        try:
            esc_t = title.replace('"', '\\"')
            esc_m = message.replace('"', '\\"')
            script = (f'display notification "{esc_m}" '
                      f'with title "{esc_t}"')
            subprocess.run(
                ["osascript", "-e", script],
                check=False, timeout=2.0,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass
    threading.Thread(target=_go, daemon=True).start()


def _update_fist_zoom(hands_list, now: float) -> Optional[str]:
    """While a fist is held, push toward camera = zoom in, pull back =
    zoom out. Uses wrist↔middle-MCP distance as a depth proxy. Returns
    'in' / 'out' / None. Resets baseline whenever the fist drops."""
    global _fist_zoom_baseline, _fist_zoom_last_at
    fist_hand = None
    if hands_list:
        for h in hands_list:
            if _is_fist(h):
                fist_hand = h
                break
    if fist_hand is None:
        _fist_zoom_baseline = None
        return None
    scale = _hand_scale(fist_hand)
    if _fist_zoom_baseline is None:
        _fist_zoom_baseline = scale
        return None
    if now - _fist_zoom_last_at < FIST_ZOOM_COOLDOWN_S:
        return None
    ratio = scale / max(_fist_zoom_baseline, 1e-6)
    if ratio >= FIST_ZOOM_RATIO_IN:
        _fist_zoom_baseline = scale  # rebaseline so a long push keeps zooming
        _fist_zoom_last_at = now
        return "in"
    if ratio <= FIST_ZOOM_RATIO_OUT:
        _fist_zoom_baseline = scale
        _fist_zoom_last_at = now
        return "out"
    return None


def _hand_scale(hand) -> float:
    """Hand-size proxy: distance from wrist (0) to middle-finger MCP (9).
    Grows when the hand moves toward the camera, shrinks as it pulls back."""
    a = hand[0]; b = hand[9]
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def _update_peace_hold(hands_lm_list, now: float) -> Optional[str]:
    """While ✌️ is held, mouse is held down (drag/select). On release,
    mouseUp. Includes a small grace window so a momentarily-lost hand
    doesn't drop the mouse mid-drag. Returns 'down' / 'up' on transitions,
    else None.
    """
    global _peace_hold_down, _peace_hold_release_at
    match = bool(hands_lm_list) and any(
        _is_peace_sign(h) for h in hands_lm_list
    )
    if match:
        _peace_hold_release_at = 0.0
        if not _peace_hold_down:
            try:
                pyautogui.mouseDown(_pause=False)
                _peace_hold_down = True
                return "down"
            except Exception as e:
                print(f"[viewer] peace mouseDown failed: {e}", flush=True)
        return None
    if _peace_hold_down:
        if _peace_hold_release_at == 0.0:
            _peace_hold_release_at = now
            return None
        if now - _peace_hold_release_at < PEACE_HOLD_RELEASE_GRACE_S:
            return None
        try:
            pyautogui.mouseUp(_pause=False)
        except Exception as e:
            print(f"[viewer] peace mouseUp failed: {e}", flush=True)
        _peace_hold_down = False
        _peace_hold_release_at = 0.0
        return "up"
    return None


def _peace_hold_force_release() -> None:
    """Release any held mouse — call on master-off / disable."""
    global _peace_hold_down, _peace_hold_release_at
    if _peace_hold_down:
        try:
            pyautogui.mouseUp(_pause=False)
        except Exception:
            pass
    _peace_hold_down = False
    _peace_hold_release_at = 0.0


def _detect_thumbs_dclick(hands_lm_list, now: float) -> bool:
    if not hands_lm_list:
        globals()['_thumbs_armed'] = True
        return False
    match = any(_is_thumbs_up(h) for h in hands_lm_list)
    return _edge_trigger_gesture(match, '_thumbs_armed', '_thumbs_last_at', now)


def _update_cursor(face_matrix, face_landmarks, hands_lm_list,
                   blendshapes, now: float, hands_up_toggle: bool) -> str:
    """Move the cursor per `_pointing_method`. Return 'primary', 'right',
    or '' for the click event type this frame."""
    global _cursor_enabled, _cursor_calibrated, _cursor_calib_start
    global _yaw_center, _pitch_center, _cur_x, _cur_y
    global _finger_center, _gaze_center

    # hands-overhead is now wired to master toggle in the capture loop; leave
    # this hook here only for calibration state reset when master flips.

    # ---- calibration: capture neutral pose per method ---------------------
    if _cursor_enabled and not _cursor_calibrated:
        if _cursor_calib_start is None:
            return ""
        if now - _cursor_calib_start < CURSOR_CALIB_S:
            return ""
        ok = False
        if _pointing_method == "head":
            if face_matrix is not None:
                y, p = _matrix_to_yaw_pitch(face_matrix)
                _yaw_center, _pitch_center = y, p
                ok = True
        elif _pointing_method == "finger":
            hand = _pick_right_hand(hands_lm_list)
            if hand is not None:
                _finger_center = (hand[8].x, hand[8].y)
                ok = True
        elif _pointing_method == "gaze":
            iris = _iris_center(face_landmarks)
            if iris is not None:
                _gaze_center = iris
                ok = True
        if ok:
            _cursor_calibrated = True
            print(f"[viewer] cursor calibrated ({_pointing_method})",
                  flush=True)
        else:
            # missing data — keep trying each frame
            _cursor_calib_start = now
        return ""

    if not _cursor_enabled or not _cursor_calibrated:
        return ""

    # ---- pointing ---------------------------------------------------------
    # Freeze the cursor while two-hand scroll is engaged so the finger
    # pointer doesn't chase the scrolling hand.
    if _scroll_active:
        return ""
    target = None
    if _pointing_method == "head":
        target = _target_from_head(face_matrix)
    elif _pointing_method == "finger":
        target = _target_from_finger(hands_lm_list)
    elif _pointing_method == "gaze":
        target = _target_from_gaze(face_landmarks)

    if target is not None:
        tx = max(_virt_x0 + 2.0, min(_virt_x1 - 2.0, target[0]))
        ty = max(_virt_y0 + 2.0, min(_virt_y1 - 2.0, target[1]))
        _cur_x += (tx - _cur_x) * CURSOR_SMOOTHING
        _cur_y += (ty - _cur_y) * CURSOR_SMOOTHING
        _move_cursor_virtual(_cur_x, _cur_y)

    # ---- click ------------------------------------------------------------
    primary = False
    if _click_method == "brow":
        primary = _detect_brow_click(blendshapes, now)
    elif _click_method == "pinch":
        primary = _detect_pinch_click(hands_lm_list, now)
    elif _click_method == "wink":
        primary = _detect_wink_click(blendshapes, now)
    elif _click_method == "right_wink":
        primary = _detect_right_wink_click(blendshapes, now)
    elif _click_method == "blink":
        primary = _detect_blink_click(blendshapes, now)
    elif _click_method == "mouth":
        if _mouth_hold_enabled:
            _update_mouth_hold(blendshapes, now)
        else:
            primary = _detect_mouth_click(blendshapes, now)
    # Mouth-hold works regardless of click method — so you can still
    # use e.g. pinch or brow-raise for normal clicks, and use mouth-open
    # as a dedicated press-and-hold.
    if _mouth_hold_enabled and _click_method != "mouth":
        _update_mouth_hold(blendshapes, now)
    right = _detect_right_click(blendshapes, now)
    if right:
        return "right"
    if primary:
        return "primary"
    return ""


def _update_smile(blendshapes) -> None:
    global _smile_val
    if not blendshapes:
        return
    l = r = 0.0
    for b in blendshapes:
        if b.category_name == "mouthSmileLeft":
            l = float(b.score)
        elif b.category_name == "mouthSmileRight":
            r = float(b.score)
    v = max(l, r)
    _smile_val = (1 - SMILE_EMA_ALPHA) * _smile_val + SMILE_EMA_ALPHA * v


def _capture_loop() -> None:
    global _latest_jpeg, _bob_pending, _blink_pending
    global _prayer_start_pending, _prayer_end_pending, _boot_pending
    global _clap_tick_pending
    global _swipe_pending, _left_hand_y, _right_hand_y
    global _left_hand_x, _right_hand_x, _system_enabled
    global _atelier_mode, _atelier_baseline_dist
    global _atelier_baseline_cx, _atelier_baseline_cy
    global _atelier_toggle_pending, _atelier_action_pending
    global _punch_pending
    cam = Camera()
    for _ in range(30):
        ok, _ = cam.read()
        if ok:
            break
        time.sleep(0.1)
    print(f"[viewer] camera warm: {cam.name}", flush=True)

    # Auto-start cursor calibration on boot so gestures work without the
    # user having to toggle master off/on first. Cursor defaults to enabled
    # so this just kicks off the neutral-pose capture countdown.
    global _cursor_calib_start
    if _cursor_enabled and _cursor_calib_start is None:
        _cursor_calib_start = time.time()
        print("[viewer] auto-calibrating cursor on boot", flush=True)

    face_landmarker = None
    if FACE_MODEL_PATH.exists():
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(FACE_MODEL_PATH)),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        face_landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        print("[viewer] face landmarker loaded", flush=True)

    hand_landmarker = None
    if _ensure_model(HAND_MODEL_PATH, HAND_MODEL_URL):
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(HAND_MODEL_PATH)),
            num_hands=2,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        hand_landmarker = mp_vision.HandLandmarker.create_from_options(opts)
        print("[viewer] hand landmarker loaded", flush=True)

    t0 = time.time()
    # Efficiency: cap capture loop so we don't burn CPU running MediaPipe
    # and JPEG-encoding at 60+ fps when 15 is plenty for gesture input.
    TARGET_FPS = 15
    FRAME_DT = 1.0 / TARGET_FPS
    next_frame_at = time.time()
    while not _stop.is_set():
        # Throttle to TARGET_FPS.
        now_throttle = time.time()
        if now_throttle < next_frame_at:
            time.sleep(next_frame_at - now_throttle)
        next_frame_at = time.time() + FRAME_DT

        ok, frame = cam.read()
        if not ok:
            time.sleep(0.01)
            continue
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.time() - t0) * 1000)

        # Efficiency: only run the models whose output the current settings
        # actually consume. Saves ~half the CPU when, e.g., click is off.
        need_face = (
            _click_method != "off"
            or _right_click_method != "off"
            or _pointing_method == "head"
            or _atelier_mode  # head-dolly zoom needs face landmarks
        )
        need_hand = (
            _pointing_method == "finger"
            or _scroll_gesture_enabled
            or _tab_swipe_enabled
            or _dict_gesture != "off"
        )

        face_nose = None
        face_blendshapes = None
        face_matrix = None
        face_landmarks_lm = None  # full landmark list for modules needing it
        fres = None
        if face_landmarker is not None and need_face:
            try:
                fres = face_landmarker.detect_for_video(mp_img, ts_ms)
                if fres.face_landmarks:
                    lm = fres.face_landmarks[0]
                    face_landmarks_lm = lm
                    _draw_face_landmarks(frame, lm)
                    face_nose = (lm[1].x, lm[1].y)  # nose tip
                if fres.face_blendshapes:
                    face_blendshapes = fres.face_blendshapes[0]
                    _update_smile(face_blendshapes)
                else:
                    _update_smile(None)
                if fres.facial_transformation_matrixes:
                    face_matrix = fres.facial_transformation_matrixes[0]
            except Exception as e:
                print(f"[viewer] face err: {e}", flush=True)

        hand_wrists = []
        hands_lm_list = []
        if hand_landmarker is not None and need_hand:
            try:
                hres = hand_landmarker.detect_for_video(mp_img, ts_ms)
                for hand in (hres.hand_landmarks or []):
                    _draw_hand(frame, hand)
                    hand_wrists.append((hand[0].x, hand[0].y))
                    hands_lm_list.append(hand)
            except Exception as e:
                print(f"[viewer] hand err: {e}", flush=True)

        _update_motion(face_nose, hand_wrists)

        now = time.time()
        nose_y_only = face_nose[1] if face_nose is not None else None
        bobbed = _detect_bob(nose_y_only, now)
        # Head bob UP (chin lift) → Cmd+C copy + native toast.
        if (_head_copy_enabled and _system_enabled
                and _detect_head_bob_up(nose_y_only, now)):
            print("[viewer] 🙆 head-up → cmd+c", flush=True)
            try:
                _fire_hotkey("cmd+c")
                _toast("handsfree", "copied ✂︎")
            except Exception as e:
                print(f"[viewer] head-up copy failed: {e}", flush=True)
        blinked = _detect_blink(face_blendshapes, now)
        prayer_change = _update_dict_gesture(hands_lm_list, now)
        booted, clap_tick = _update_double_clap(hands_lm_list, now)
        swipe_dir = _update_swipe(hands_lm_list, now)
        left_x, left_y, right_x, right_y = _split_hands_xy(hands_lm_list)

        hands_up_toggle = _update_hands_up(hands_lm_list, now)
        # T-gesture master toggle (experimental, off by default)
        if _t_timeout_enabled and _detect_t_gesture(hands_lm_list, now):
            print("[viewer] T-gesture → master toggle", flush=True)
            _set_master(not _system_enabled, source="t-gesture")
        # ✨ A-pose → atelier mode toggle
        atelier_toggled = False
        if _atelier_enabled and _detect_a_pose(hands_lm_list, now):
            _atelier_mode = not _atelier_mode
            _atelier_baseline_dist = None
            _atelier_baseline_cx = None
            _atelier_baseline_cy = None
            atelier_toggled = True
            print(f"[viewer] ✨ atelier mode = "
                  f"{'ON' if _atelier_mode else 'off'}", flush=True)
            with _state_lock:
                _atelier_toggle_pending = True
        # While atelier is ON, two-hand geometry drives zoom + pan, the
        # head leans drive head-dolly zoom, and pinching = grab-and-drag.
        atelier_action = None
        if _atelier_mode and _system_enabled:
            atelier_action = _update_atelier(hands_lm_list, now)
            if atelier_action is None:
                atelier_action = _update_atelier_head_zoom(
                    face_landmarks_lm, now,
                )
            if atelier_action is not None:
                _atelier_fire(atelier_action)
                print(f"[viewer] atelier {atelier_action}", flush=True)
                with _state_lock:
                    _atelier_action_pending = atelier_action
            grab_evt = _update_atelier_pinch_grab(hands_lm_list)
            if grab_evt is not None:
                print(f"[viewer] atelier {grab_evt}", flush=True)
                with _state_lock:
                    _atelier_action_pending = grab_evt
        else:
            # Safety: release any held mouse if we exited atelier mid-grab.
            _update_atelier_pinch_grab(hands_lm_list)

        # 🥊 Boxing punch detection — feeds /boxing page via SSE.
        if _boxing_enabled:
            punches = _detect_punches(hands_lm_list, now)
            if punches:
                with _state_lock:
                    _punch_pending.extend(punches)
        # Peace sign ✌️ → press-and-hold the mouse (hold-to-drag).
        # Push the held ✌️ toward the camera → Cmd+C (copy).
        if _peace_rclick_enabled and _system_enabled:
            evt = _update_peace_hold(hands_lm_list, now)
            if evt == "copy":
                print("[viewer] peace ✌️ + push → cmd+c", flush=True)
            elif evt is not None:
                print(f"[viewer] peace ✌️ → mouse {evt}", flush=True)
        else:
            # Safety: drop any held mouse if disabled mid-hold.
            _peace_hold_force_release()
        # Thumbs up 👍 → paste (Cmd+V).
        if (_thumbs_dclick_enabled and _system_enabled
                and _detect_thumbs_dclick(hands_lm_list, now)):
            print("[viewer] thumbs 👍 → cmd+v", flush=True)
            try:
                _fire_hotkey("cmd+v")
                _toast("handsfree", "pasted 📋")
            except Exception as e:
                print(f"[viewer] cmd+v failed: {e}", flush=True)
        # Two-hand tab swipe → Cmd+Shift+]/[.
        tab_dir = _update_tab_swipe(hands_lm_list, now)
        if tab_dir is not None and _system_enabled and not _jam_mode:
            if _tab_swipe_action == "spaces":
                combo = (SPACE_SWIPE_NEXT_COMBO if tab_dir == "next"
                         else SPACE_SWIPE_PREV_COMBO)
            else:
                combo = (TAB_SWIPE_NEXT_COMBO if tab_dir == "next"
                         else TAB_SWIPE_PREV_COMBO)
            _fire_hotkey(combo)
            print(f"[viewer] swipe {tab_dir} ({_tab_swipe_action}) → "
                  f"{combo}", flush=True)
            with _state_lock:
                _swipe_pending = f"tab-{tab_dir}"
        if _scroll_mode == "fist":
            # Fist depth → zoom: push toward camera = Cmd+=, pull away = Cmd+-.
            # Lateral motion still drives pan/scroll below.
            if _fist_zoom_enabled and _system_enabled:
                zdir = _update_fist_zoom(hands_lm_list, now)
                if zdir is not None:
                    try:
                        _fire_hotkey("cmd+=" if zdir == "in" else "cmd+-")
                        print(f"[viewer] 🤛 fist-zoom {zdir}", flush=True)
                    except Exception as e:
                        print(f"[viewer] fist-zoom failed: {e}", flush=True)
            v_amt, h_amt = _update_fist_scroll(hands_lm_list, now)
            if _system_enabled and _scroll_gesture_enabled and (v_amt or h_amt):
                # Emit a single native two-axis scroll event so Figma sees
                # diagonal pans like a real trackpad does. Falls back to
                # pyautogui if Quartz isn't available.
                try:
                    if _QUARTZ_OK:
                        ev = CGEventCreateScrollWheelEvent(
                            None,
                            kCGScrollEventUnitPixel,
                            2,             # axis count
                            int(v_amt),    # axis 1 = vertical
                            int(h_amt),    # axis 2 = horizontal
                        )
                        CGEventPost(kCGHIDEventTap, ev)
                    else:
                        if v_amt:
                            pyautogui.scroll(v_amt, _pause=False)
                        if h_amt:
                            pyautogui.hscroll(h_amt, _pause=False)
                except Exception as e:
                    print(f"[viewer] fist-scroll failed: {e}", flush=True)
        elif _scroll_mode == "two_hands":
            scroll_amt = _update_two_hand_scroll(hands_lm_list, now)
            if scroll_amt != 0 and _system_enabled and _scroll_gesture_enabled:
                try:
                    pyautogui.scroll(scroll_amt, _pause=False)
                except Exception as e:
                    print(f"[viewer] scroll failed: {e}", flush=True)
        elif _scroll_mode in ("head_lefthand", "head_mouth", "head_always"):
            gate = {
                "head_lefthand": "lefthand",
                "head_mouth":    "mouth",
                "head_always":   "always",
            }[_scroll_mode]
            scroll_amt = _update_head_scroll(
                hands_lm_list, face_nose, face_blendshapes, now, gate,
            )
            if scroll_amt != 0 and _system_enabled and _scroll_gesture_enabled:
                try:
                    pyautogui.scroll(scroll_amt, _pause=False)
                except Exception as e:
                    print(f"[viewer] head-scroll failed: {e}", flush=True)
        elif _scroll_mode == "brow":
            scroll_amt = _update_brow_scroll(face_blendshapes, now)
            if scroll_amt != 0 and _system_enabled and _scroll_gesture_enabled:
                try:
                    pyautogui.scroll(scroll_amt, _pause=False)
                except Exception as e:
                    print(f"[viewer] brow-scroll failed: {e}", flush=True)
        face_lms = None
        if fres is not None:
            try:
                if fres.face_landmarks:
                    face_lms = fres.face_landmarks[0]
            except Exception:
                face_lms = None
        click_event = _update_cursor(
            face_matrix, face_lms, hands_lm_list,
            face_blendshapes, now, hands_up_toggle,
        )
        if _cursor_enabled and _cursor_calibrated and _system_enabled:
            _dispatch_click(click_event, now)

        if (swipe_dir is not None and not _jam_mode
                and _system_enabled and _swipe_gesture_enabled):
            _fire_swipe_action(swipe_dir)
        # Master toggle is UI-click only — gesture triggers (hands-overhead,
        # double-clap) were unreliable and have been removed.
        _ = hands_up_toggle  # kept for potential future use

        if prayer_change is True and _system_enabled:
            if _dict_mode == "latch":
                print("[viewer] dict gesture → latch START tap", flush=True)
                _tap_wispr_hotkey()
            else:
                print("[viewer] dict gesture START → holding key", flush=True)
                _hold_wispr_key()
        elif prayer_change is False and _system_enabled:
            if _dict_mode == "hold":
                print("[viewer] dict gesture END → releasing key", flush=True)
                _release_wispr_key()
            elif _dict_mode == "latch":
                # Also tap on EXIT so Wispr (or Apple) in toggle-mode gets
                # the second press to stop recording.
                print("[viewer] dict gesture → latch STOP tap", flush=True)
                _tap_wispr_hotkey()
        # Double-clap master toggle removed — button-only from now on.
        _ = booted

        with _state_lock:
            if bobbed:
                _bob_pending = True
            if blinked:
                _blink_pending = True
            if prayer_change is True:
                _prayer_start_pending = True
            elif prayer_change is False:
                _prayer_end_pending = True
            if booted:
                _boot_pending = True
            if clap_tick:
                _clap_tick_pending = True
            if swipe_dir is not None:
                _swipe_pending = swipe_dir
            _left_hand_y = left_y
            _right_hand_y = right_y
            _left_hand_x = left_x
            _right_hand_x = right_x

        # Only JPEG-encode if someone is actually watching the stream.
        with _stream_clients_lock:
            has_viewers = _stream_clients > 0
        if has_viewers:
            okj, buf = cv2.imencode(".jpg", frame,
                                    [cv2.IMWRITE_JPEG_QUALITY, 75])
            if okj:
                with _jpeg_lock:
                    _latest_jpeg = buf.tobytes()

    cam.release()
    print("[viewer] capture loop stopped", flush=True)


# --- http handler ---------------------------------------------------------
# A few common misspellings / homophones. Everything else is passed to
# `open -a` as-is and macOS fuzzy-matches app names.
APP_ALIASES = {
    # Arc — Whisper loves homophones
    "ark":           "Arc",
    "arc browser":   "Arc",
    # Telegram
    "telegrams":     "Telegram",
    "telegraph":     "Telegram",
    "tell a gram":   "Telegram",
    "tell him":      "Telegram",
    # Notion
    "notions":       "Notion",
    "ocean":         "Notion",
    "motion":        "Notion",
    "emotion":       "Notion",
    "lotion":        "Notion",
    # Figma
    "figure":        "Figma",
    "figure ma":     "Figma",
    "sigma":         "Figma",
    # Slack
    "slacks":        "Slack",
    # Spotify
    "spotty":        "Spotify",
    "spot ify":      "Spotify",
    # Cursor (editor)
    "cursor":        "Cursor",
    # Wispr
    "wispr":         "Wispr Flow",
    "whisper":       "Wispr Flow",
    "wisper":        "Wispr Flow",
    "whisper flow":  "Wispr Flow",
    # Misc
    "chat gpt":      "ChatGPT",
    "chatgpt":       "ChatGPT",
    "vs code":       "Visual Studio Code",
    "vscode":        "Visual Studio Code",
    "zoom":          "zoom.us",
}

# Canonical app names we try to fuzzy-match raw targets against when the
# alias table misses. Kept lowercase; _resolve_app compares prefixes and
# substrings. Small-edit-distance matches are preferred.
_KNOWN_APPS = [
    "Telegram", "Notion", "Arc", "Slack", "Spotify", "Figma", "Safari",
    "Chrome", "Cursor", "Terminal", "Finder", "Messages", "Mail",
    "Calendar", "ChatGPT", "Visual Studio Code", "Wispr Flow",
]


def _scroll(direction: str) -> None:
    try:
        amount = 10 if direction == "up" else -10
        pyautogui.scroll(amount, _pause=False)
        print(f"[viewer] scroll {direction}", flush=True)
    except Exception as e:
        print(f"[viewer] scroll failed: {e}", flush=True)


def _dispatch_click(event: str, now: float) -> None:
    """Turn detector events into pyautogui clicks, honoring double-click mode.

    event: 'primary' | 'right' | ''
    When `_double_click_on`, two primaries within DOUBLE_WINDOW_S collapse
    into one native double-click. Otherwise primary fires immediately.
    """
    global _pending_click_at, _pending_click_waiting
    # Right click bypasses the double-click delay.
    if event == "right":
        try:
            pyautogui.rightClick(_pause=False)
            print(f"[viewer] {_right_click_method} → right-click", flush=True)
        except pyautogui.FailSafeException:
            pass
        return
    if event == "primary":
        if _double_click_on:
            if _pending_click_waiting and (now - _pending_click_at) < DOUBLE_WINDOW_S:
                _pending_click_waiting = False
                try:
                    pyautogui.doubleClick(_pause=False)
                    print(f"[viewer] {_click_method} → double-click", flush=True)
                except pyautogui.FailSafeException:
                    pass
            else:
                _pending_click_waiting = True
                _pending_click_at = now
        else:
            try:
                pyautogui.click(_pause=False)
                print(f"[viewer] {_click_method} → click", flush=True)
            except pyautogui.FailSafeException:
                pass
    # If a single is pending and its double window expired, fire it now.
    if _pending_click_waiting and (now - _pending_click_at) >= DOUBLE_WINDOW_S:
        _pending_click_waiting = False
        try:
            pyautogui.click(_pause=False)
            print(f"[viewer] {_click_method} → click (delayed)", flush=True)
        except pyautogui.FailSafeException:
            pass


def _click() -> None:
    try:
        pyautogui.click(_pause=False)
        print("[viewer] click", flush=True)
    except Exception as e:
        print(f"[viewer] click failed: {e}", flush=True)


def _volume(direction: str) -> None:
    if direction == "mute":
        script = ('set volume output muted '
                  '(not (output muted of (get volume settings)))')
    elif direction == "up":
        script = ('set volume output volume '
                  '((output volume of (get volume settings)) + 10)')
    else:
        script = ('set volume output volume '
                  '((output volume of (get volume settings)) - 10)')
    try:
        subprocess.Popen(["osascript", "-e", script])
        print(f"[viewer] volume {direction}", flush=True)
    except Exception as e:
        print(f"[viewer] volume failed: {e}", flush=True)


def _resolve_app(target: str) -> str:
    """Turn a noisy spoken-ish target into a real app name, in 3 tiers:
    1. Exact alias match against APP_ALIASES (handles mishearings).
    2. Fuzzy substring / edit-distance match against _KNOWN_APPS.
    3. Fall through to the raw target; macOS `open -a` will fuzzy-match
       installed apps in /Applications on its own."""
    raw = (target or "").strip()
    if not raw:
        return raw
    t = raw.lower()
    if t in APP_ALIASES:
        return APP_ALIASES[t]
    # Substring hit on a known app (e.g. "tele" → Telegram).
    for name in _KNOWN_APPS:
        if t == name.lower() or t in name.lower() or name.lower() in t:
            return name
    # Small-edit-distance fallback using difflib (no extra deps).
    try:
        import difflib
        cand = difflib.get_close_matches(
            t, [n.lower() for n in _KNOWN_APPS], n=1, cutoff=0.75,
        )
        if cand:
            for name in _KNOWN_APPS:
                if name.lower() == cand[0]:
                    return name
    except Exception:
        pass
    return raw


def _open_app(target: str) -> tuple:
    name = _resolve_app(target)
    try:
        subprocess.Popen(["open", "-a", name])
        print(f"[viewer] open -a {name!r}", flush=True)
        return True, name
    except Exception as e:
        print(f"[viewer] open failed: {e}", flush=True)
        return False, name


# Server-side mirror of the JS matchCommand() so /transcribe can go
# straight from Whisper text → fired action without a second round trip.
_OPEN_RE  = re.compile(r"\b(?:open|launch)\s+([a-z0-9][a-z0-9 \-\.]{0,30})", re.I)
_CLOSE_RE = re.compile(r"\b(?:close|quit)\s+([a-z0-9][a-z0-9 \-\.]{0,30})",  re.I)
_INSTANT = [
    (re.compile(r"\bscroll\s+down\b", re.I),  ("scroll", "down")),
    (re.compile(r"\bscroll\s+up\b", re.I),    ("scroll", "up")),
    (re.compile(r"\bpage\s+down\b", re.I),    ("scroll", "down")),
    (re.compile(r"\bpage\s+up\b", re.I),      ("scroll", "up")),
    (re.compile(r"\b(?:click|tap)\b", re.I),  ("click",  "")),
    (re.compile(r"\bvolume\s+up\b", re.I),    ("volume", "up")),
    (re.compile(r"\bvolume\s+down\b", re.I),  ("volume", "down")),
    (re.compile(r"\b(?:mute|silence)\b", re.I), ("volume", "mute")),
    (re.compile(r"\bnext\s+desktop\b", re.I), ("desktop", "next")),
    (re.compile(r"\bprevious\s+desktop\b", re.I), ("desktop", "prev")),
]

def _clean_target(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^(the|up)\s+", "", s, flags=re.I)
    s = re.sub(r"\s+(please|now|app|application)$", "", s, flags=re.I)
    return s.strip()

def _match_command(text: str):
    t = (text or "").strip().lower()
    if not t:
        return None
    for pat, (action, target) in _INSTANT:
        if pat.search(t):
            return {"action": action, "target": target}
    m = _OPEN_RE.search(t)
    if m:
        return {"action": "open",  "target": _clean_target(m.group(1))}
    m = _CLOSE_RE.search(t)
    if m:
        return {"action": "close", "target": _clean_target(m.group(1))}
    # Bare app name: if the transcript (minus trailing punct / "please" etc)
    # matches a known installed app name, treat it as "open <app>".
    # Only fires for short utterances so we don't hijack real sentences.
    stripped = re.sub(r"[^a-z0-9 ]+", "", t).strip()
    stripped = re.sub(r"\s+(please|now|app|application)$", "", stripped)
    if stripped and len(stripped.split()) <= 4:
        if stripped in APP_ALIASES:
            return {"action": "open", "target": stripped}
        for app in INSTALLED_APPS:
            al = app.lower()
            if stripped == al or stripped == al.replace(".", ""):
                return {"action": "open", "target": app}
        # Also match the first word vs single-word app names for speed
        first = stripped.split()[0]
        for app in INSTALLED_APPS:
            al = app.lower()
            if " " not in al and al == first:
                return {"action": "open", "target": app}
    return None

def _dispatch_command(cmd: dict) -> dict:
    """Run a matched command (same switch as /command POST). Returns a
    small status dict suitable for JSON."""
    action = (cmd.get("action") or "").lower()
    target = cmd.get("target") or ""
    if action == "open" and target:
        ok, name = _open_app(target)
        return {"ok": ok, "action": action, "resolved": name}
    if action == "close" and target:
        ok, name = _quit_app(target)
        return {"ok": ok, "action": action, "resolved": name}
    if action == "scroll":
        _scroll(target)
        return {"ok": True, "action": action, "target": target}
    if action == "click":
        _click()
        return {"ok": True, "action": action}
    if action == "volume":
        _volume(target)
        return {"ok": True, "action": action, "target": target}
    if action == "desktop":
        _fire_swipe_action("right" if target == "next" else "left")
        return {"ok": True, "action": action, "target": target}
    return {"ok": False, "action": action, "error": "unknown action"}


def _quit_app(target: str) -> tuple:
    name = _resolve_app(target)
    try:
        subprocess.Popen(
            ["osascript", "-e", f'tell application "{name}" to quit']
        )
        print(f"[viewer] quit {name!r}", flush=True)
        return True, name
    except Exception as e:
        print(f"[viewer] quit failed: {e}", flush=True)
        return False, name


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a, **k) -> None:
        pass

    def _write_status(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        global _wispr_method, _scroll_sens, _scroll_mode
        global _right_click_method, _double_click_on, _cursor_sens
        global _scroll_gesture_enabled, _swipe_gesture_enabled
        global _scroll_sens_mult

        # Local speech-to-text + command dispatch. Browser POSTs the raw
        # audio blob (webm/opus from MediaRecorder) here; we transcribe
        # with faster-whisper, run _match_command, fire if matched.
        if self.path == "/transcribe":
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0:
                self._write_status(400, "text/plain", b"empty body")
                return
            blob = self.rfile.read(length)
            suffix = ".webm"
            ctype = (self.headers.get("Content-Type") or "").lower()
            if "wav" in ctype:
                suffix = ".wav"
            elif "ogg" in ctype:
                suffix = ".ogg"
            elif "mp4" in ctype or "m4a" in ctype:
                suffix = ".m4a"
            try:
                tmp = tempfile.NamedTemporaryFile(
                    suffix=suffix, delete=False
                )
                tmp.write(blob)
                tmp.close()
                model = _get_whisper()
                t0 = time.time()
                segments, _info = model.transcribe(
                    tmp.name, beam_size=1, vad_filter=True,
                    language="en",
                )
                text = " ".join(s.text.strip() for s in segments).strip()
                dt = time.time() - t0
                print(f"[viewer] whisper {dt*1000:.0f}ms: {text!r}",
                      flush=True)
                cmd = _match_command(text)
                fired = None
                if cmd:
                    fired = _dispatch_command(cmd)
                resp = {
                    "ok": True,
                    "text": text,
                    "ms": int(dt * 1000),
                    "command": cmd,
                    "fired": fired,
                }
                self._write_status(
                    200, "application/json",
                    json.dumps(resp).encode(),
                )
            except Exception as e:
                print(f"[viewer] transcribe err: {e}", flush=True)
                self._write_status(
                    500, "application/json",
                    json.dumps({"ok": False, "error": str(e)}).encode(),
                )
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass
            return

        if self.path == "/command":
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            try:
                data = json.loads(raw)
            except Exception:
                self._write_status(400, "text/plain", b"bad json")
                return
            action = (data.get("action") or "").lower()
            target = data.get("target") or ""
            if action == "open" and target:
                ok, name = _open_app(target)
                self._write_status(
                    200 if ok else 500,
                    "application/json",
                    json.dumps({"ok": ok, "resolved": name}).encode(),
                )
                return
            if action == "close" and target:
                ok, name = _quit_app(target)
                self._write_status(
                    200 if ok else 500,
                    "application/json",
                    json.dumps({"ok": ok, "resolved": name}).encode(),
                )
                return
            if action == "scroll":
                _scroll(target)
                self._write_status(200, "application/json", b'{"ok":true}')
                return
            if action == "click":
                _click()
                self._write_status(200, "application/json", b'{"ok":true}')
                return
            if action == "volume":
                _volume(target)
                self._write_status(200, "application/json", b'{"ok":true}')
                return
            if action == "desktop":
                _fire_swipe_action("right" if target == "next" else "left")
                self._write_status(200, "application/json", b'{"ok":true}')
                return
            if action == "jam":
                global _jam_mode
                _jam_mode = bool(data.get("on"))
                print(f"[viewer] jam mode = {_jam_mode}", flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({"ok": True, "jam": _jam_mode}).encode(),
                )
                return
            if action == "voice_daemon":
                want = bool(data.get("on"))
                if want:
                    _voice_start()
                else:
                    _voice_stop_fn()
                self._write_status(
                    200, "application/json",
                    json.dumps({
                        "ok": True, "voiceDaemon": _voice_enabled,
                        "voiceState": _voice_state,
                    }).encode(),
                )
                return
            if action == "cursor_enable":
                global _cursor_enabled, _cursor_calibrated, _cursor_calib_start
                global _finger_center, _gaze_center
                want = bool(data.get("on"))
                _cursor_enabled = want
                if want:
                    _cursor_calibrated = False
                    _cursor_calib_start = time.time()
                    _finger_center = None
                    _gaze_center = None
                print(f"[viewer] cursor {'ON' if want else 'OFF'} "
                      f"(via button)", flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({"ok": True, "cursor": _cursor_enabled}).encode(),
                )
                return
            if action == "cursor_method":
                global _pointing_method, _click_method
                pointing = data.get("pointing")
                click = data.get("click")
                changed = False
                if pointing in ("head", "finger", "gaze"):
                    if pointing != _pointing_method:
                        _pointing_method = pointing
                        changed = True
                if click in ("wink", "brow", "pinch", "blink",
                             "right_wink", "mouth"):
                    _click_method = click
                if changed:
                    # Swapping pointing method ⇒ re-calibrate on next frame.
                    _cursor_calibrated = False
                    _cursor_calib_start = time.time()
                    _finger_center = None
                    _gaze_center = None
                print(f"[viewer] cursor method = {_pointing_method}/"
                      f"{_click_method}", flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({
                        "ok": True,
                        "pointing": _pointing_method,
                        "click": _click_method,
                        "wispr": _wispr_method,
                        "scrollSens": _scroll_sens,
                        "scrollMode": _scroll_mode,
                        "scrollSpeed": round(_scroll_sens_mult, 2),
                        "rightClick": _right_click_method,
                        "doubleClick": _double_click_on,
                        "cursorSens": round(_cursor_sens, 2),
                    }).encode(),
                )
                return
            if action == "wispr_method":
                method = data.get("method")
                if method in ("applescript_fn", "cgevent_f19",
                              "cgevent_fn", "double_tap_f19",
                              "double_tap_fn", "triple_tap_fn",
                              "menu_click",
                              "all", "apple_dictation", "off"):
                    _wispr_method = method
                print(f"[viewer] wispr method = {_wispr_method}", flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({"ok": True, "wispr": _wispr_method}).encode(),
                )
                return
            if action == "right_click_method":
                method = data.get("method")
                if method in ("smile", "pucker", "furrow", "off"):
                    _right_click_method = method
                print(f"[viewer] right-click = {_right_click_method}", flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({"ok": True, "rightClick": _right_click_method}).encode(),
                )
                return
            if action == "double_click":
                _double_click_on = bool(data.get("on"))
                print(f"[viewer] double-click = {_double_click_on}", flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({"ok": True, "doubleClick": _double_click_on}).encode(),
                )
                return
            if action == "cursor_sens":
                try:
                    v = float(data.get("value", _cursor_sens))
                except (TypeError, ValueError):
                    v = _cursor_sens
                _cursor_sens = max(0.15, min(3.5, v))
                print(f"[viewer] cursor sens = {_cursor_sens:.2f}", flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({"ok": True, "cursorSens": _cursor_sens}).encode(),
                )
                return
            if action == "scroll_sens":
                sens = data.get("sens")
                if sens == "off":
                    _scroll_gesture_enabled = False
                elif sens in SCROLL_GAIN_MAP:
                    _scroll_gesture_enabled = True
                    _scroll_sens = sens
                print(f"[viewer] scroll gesture = "
                      f"{'ON:' + _scroll_sens if _scroll_gesture_enabled else 'OFF'}",
                      flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({
                        "ok": True,
                        "scrollSens": (_scroll_sens
                                       if _scroll_gesture_enabled else "off"),
                    }).encode(),
                )
                return
            if action == "scroll_speed":
                # Continuous multiplier on the base scroll gain. UI slider
                # in the control center sends a float here.
                try:
                    v = float(data.get("value", _scroll_sens_mult))
                except (TypeError, ValueError):
                    v = _scroll_sens_mult
                _scroll_sens_mult = max(0.3, min(8.0, v))
                print(f"[viewer] scroll speed = {_scroll_sens_mult:.2f}×",
                      flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({
                        "ok": True,
                        "scrollSpeed": round(_scroll_sens_mult, 2),
                    }).encode(),
                )
                return
            if action == "scroll_mode":
                mode = data.get("mode")
                if mode in ("fist", "two_hands", "off",
                            "head_lefthand", "head_mouth",
                            "head_always", "brow"):
                    _scroll_mode = mode
                    _scroll_gesture_enabled = (mode != "off")
                    print(f"[viewer] scroll mode = {_scroll_mode}", flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({
                        "ok": True,
                        "scrollMode": _scroll_mode,
                    }).encode(),
                )
                return
            if action == "tab_swipe":
                global _tab_swipe_enabled, _tab_swipe_action
                v = data.get("on")
                # Accept new action strings ("tabs"/"spaces"/"off") as well
                # as legacy bool / "on"/"off".
                if v in ("tabs", "spaces"):
                    _tab_swipe_action = v
                    _tab_swipe_enabled = True
                elif v == "off":
                    _tab_swipe_enabled = False
                elif isinstance(v, bool):
                    _tab_swipe_enabled = v
                elif v == "on":
                    _tab_swipe_enabled = True
                print(f"[viewer] tab swipe = "
                      f"{('ON:' + _tab_swipe_action) if _tab_swipe_enabled else 'OFF'}",
                      flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({
                        "ok": True,
                        "tabSwipe": _tab_swipe_enabled,
                        "tabSwipeAction": _tab_swipe_action,
                    }).encode(),
                )
                return
            if action == "swipe_gesture":
                v = data.get("on")
                if isinstance(v, bool):
                    _swipe_gesture_enabled = v
                elif v in ("on", "off"):
                    _swipe_gesture_enabled = (v == "on")
                print(f"[viewer] swipe gesture = "
                      f"{'ON' if _swipe_gesture_enabled else 'OFF'}", flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({
                        "ok": True,
                        "swipeGesture": _swipe_gesture_enabled,
                    }).encode(),
                )
                return
            if action == "wispr_test":
                _tap_wispr_hotkey()
                self._write_status(
                    200, "application/json",
                    json.dumps({"ok": True, "fired": _wispr_method}).encode(),
                )
                return
            if action == "wispr_hold_test":
                def _hold_then_release():
                    _press_wispr_key_down()
                    time.sleep(2.0)
                    _release_wispr_key_up()
                threading.Thread(target=_hold_then_release, daemon=True).start()
                self._write_status(
                    200, "application/json",
                    json.dumps({"ok": True, "method": _wispr_method}).encode(),
                )
                return
            if action == "master":
                _set_master(bool(data.get("on")), source="ui")
                self._write_status(
                    200, "application/json",
                    json.dumps({
                        "ok": True, "systemEnabled": _system_enabled,
                    }).encode(),
                )
                return
            if action == "t_timeout":
                global _t_timeout_enabled, _t_gesture_armed
                _t_timeout_enabled = bool(data.get("on"))
                _t_gesture_armed = True
                print(f"[viewer] t-timeout = {_t_timeout_enabled}", flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({"ok": True, "tTimeout": _t_timeout_enabled}).encode(),
                )
                return
            if action == "mouth_hold":
                global _mouth_hold_enabled
                _mouth_hold_enabled = bool(data.get("on"))
                if not _mouth_hold_enabled:
                    _mouth_hold_release()
                print(f"[viewer] mouth-hold = {_mouth_hold_enabled}", flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({"ok": True, "mouthHold": _mouth_hold_enabled}).encode(),
                )
                return
            if action == "peace_rclick":
                global _peace_rclick_enabled, _peace_armed
                _peace_rclick_enabled = bool(data.get("on"))
                _peace_armed = True
                if not _peace_rclick_enabled:
                    _peace_hold_force_release()
                print(f"[viewer] peace-hold = {_peace_rclick_enabled}", flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({"ok": True, "peaceRclick": _peace_rclick_enabled}).encode(),
                )
                return
            if action == "thumbs_dclick":
                global _thumbs_dclick_enabled, _thumbs_armed
                _thumbs_dclick_enabled = bool(data.get("on"))
                _thumbs_armed = True
                print(f"[viewer] thumbs-dclick = {_thumbs_dclick_enabled}", flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({"ok": True, "thumbsDclick": _thumbs_dclick_enabled}).encode(),
                )
                return
            if action == "head_copy":
                global _head_copy_enabled
                _head_copy_enabled = bool(data.get("on"))
                print(f"[viewer] head-up copy = {_head_copy_enabled}", flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({"ok": True, "headCopy": _head_copy_enabled}).encode(),
                )
                return
            if action == "fist_zoom":
                global _fist_zoom_enabled, _fist_zoom_baseline
                _fist_zoom_enabled = bool(data.get("on"))
                _fist_zoom_baseline = None
                print(f"[viewer] fist-zoom = {_fist_zoom_enabled}", flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({"ok": True, "fistZoom": _fist_zoom_enabled}).encode(),
                )
                return
            if action == "boxing":
                global _boxing_enabled, _left_wrist_prev, _right_wrist_prev
                _boxing_enabled = bool(data.get("on"))
                _left_wrist_prev = None
                _right_wrist_prev = None
                print(f"[viewer] 🥊 boxing = {_boxing_enabled}", flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({"ok": True, "boxing": _boxing_enabled}).encode(),
                )
                return
            if action == "atelier":
                # Toggle the feature ENABLE (gesture detection). The mode
                # itself is entered/exited by doing the A-pose.
                global _atelier_enabled, _atelier_armed
                _atelier_enabled = bool(data.get("on"))
                _atelier_armed = True
                print(f"[viewer] atelier feature = {_atelier_enabled}",
                      flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({"ok": True,
                                "atelierEnabled": _atelier_enabled}).encode(),
                )
                return
            if action == "atelier_force":
                # Manual mode toggle from UI (emergency exit without pose).
                global _atelier_mode
                _atelier_mode = bool(data.get("on"))
                print(f"[viewer] atelier mode (forced) = {_atelier_mode}",
                      flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({"ok": True,
                                "atelierMode": _atelier_mode}).encode(),
                )
                return
            if action == "release_extra_tap":
                global _wispr_release_extra_tap
                _wispr_release_extra_tap = bool(data.get("on"))
                print(f"[viewer] wispr release-extra-tap = "
                      f"{_wispr_release_extra_tap}", flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({"ok": True,
                                "releaseExtraTap": _wispr_release_extra_tap}).encode(),
                )
                return
            if action == "release_nuclear":
                global _wispr_release_nuclear
                _wispr_release_nuclear = bool(data.get("on"))
                print(f"[viewer] wispr release-nuclear = "
                      f"{_wispr_release_nuclear}", flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({"ok": True,
                                "releaseNuclear": _wispr_release_nuclear}).encode(),
                )
                return
            if action == "fn_diag":
                # Manual one-shot Fn-key probes for diagnosing which variant
                # Wispr (or anything else) actually listens to. keycode 63 = Fn.
                sub = (data.get("sub") or "").strip()
                kc = 63
                try:
                    if sub == "tap":
                        _tap_wispr_cgevent(kc, fn_flag=True)
                    elif sub == "double":
                        _double_tap_cgevent(kc, fn_flag=True)
                    elif sub == "triple":
                        _triple_tap_cgevent(kc, fn_flag=True)
                    elif sub == "hold_500ms":
                        threading.Thread(
                            target=_hold_cgevent_for,
                            args=(kc, True, 0.5), daemon=True).start()
                    elif sub == "hold_2s":
                        threading.Thread(
                            target=_hold_cgevent_for,
                            args=(kc, True, 2.0), daemon=True).start()
                    elif sub == "down":
                        if _QUARTZ_OK:
                            d = CGEventCreateKeyboardEvent(None, kc, True)
                            CGEventSetFlags(d, kCGEventFlagMaskSecondaryFn)
                            CGEventPost(kCGHIDEventTap, d)
                            print("[viewer] fn_diag: DOWN only", flush=True)
                    elif sub == "up":
                        if _QUARTZ_OK:
                            u = CGEventCreateKeyboardEvent(None, kc, False)
                            CGEventSetFlags(u, kCGEventFlagMaskSecondaryFn)
                            CGEventPost(kCGHIDEventTap, u)
                            print("[viewer] fn_diag: UP only", flush=True)
                    elif sub == "escape":
                        _tap_escape_cgevent()
                    elif sub == "menu":
                        _tap_wispr_menu_click()
                    elif sub == "nuclear":
                        _nuclear_stop_wispr()
                    print(f"[viewer] fn_diag {sub}", flush=True)
                except Exception as e:
                    print(f"[viewer] fn_diag failed: {e}", flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({"ok": True, "sub": sub}).encode(),
                )
                return
            if action == "clap_preset":
                global _clap_preset
                global CLAP_CLOSE_THRESHOLD, CLAP_FAR_THRESHOLD
                global CLAP_GAP_MIN_S, CLAP_GAP_MAX_S
                global CLAP_BOOT_COOLDOWN_S, CLAP_HAND_LOST_GRACE_S
                preset = data.get("preset")
                if preset == "off" or preset in CLAP_PRESETS:
                    _clap_preset = preset
                    if preset in CLAP_PRESETS:
                        (CLAP_CLOSE_THRESHOLD, CLAP_FAR_THRESHOLD,
                         CLAP_GAP_MIN_S, CLAP_GAP_MAX_S,
                         CLAP_BOOT_COOLDOWN_S,
                         CLAP_HAND_LOST_GRACE_S) = CLAP_PRESETS[preset]
                    print(f"[viewer] clap preset = {preset}", flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({"ok": True, "clapPreset": _clap_preset}).encode(),
                )
                return
            if action == "dict_gesture":
                global _dict_gesture
                g = data.get("gesture")
                if g in ("prayer", "fingertips", "fist", "off"):
                    _dict_gesture = g
                    # If switching while held, release so we don't get stuck.
                    if _prayer_active:
                        _update_prayer_like_release()
                        _release_wispr_key_up()
                    print(f"[viewer] dict gesture = {g}", flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({"ok": True, "dictGesture": _dict_gesture}).encode(),
                )
                return
            if action == "dict_mode":
                global _dict_mode
                m = data.get("mode")
                if m in ("hold", "latch"):
                    _dict_mode = m
                    # Don't leave a held key dangling when switching to latch.
                    _release_wispr_key_up()
                    print(f"[viewer] dict mode = {m}", flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({"ok": True, "dictMode": _dict_mode}).encode(),
                )
                return
            if action == "voice_hotkey":
                combo = (data.get("combo") or "").strip()
                phrase = (data.get("phrase") or "").strip()
                ok = _fire_hotkey(combo) if combo else False
                print(f"[viewer] voice '{phrase}' → {combo} ok={ok}", flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({"ok": ok}).encode(),
                )
                return
            if action == "quit":
                print("[viewer] quit requested from UI", flush=True)
                self._write_status(
                    200, "application/json", b'{"ok":true,"quitting":true}',
                )
                threading.Timer(
                    0.2, lambda: os.kill(os.getpid(), signal.SIGTERM),
                ).start()
                return
            self._write_status(400, "text/plain", b"unknown action")
            return
        self._write_status(404, "text/plain", b"not found")

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            self._write_status(200, "text/html; charset=utf-8",
                               HTML.encode("utf-8"))
            return

        if self.path == "/boxing":
            self._write_status(200, "text/html; charset=utf-8",
                               BOXING_HTML.encode("utf-8"))
            return

        if self.path == "/stream":
            self.send_response(200)
            self.send_header(
                "Content-Type",
                f"multipart/x-mixed-replace; boundary={BOUNDARY}",
            )
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            global _stream_clients
            with _stream_clients_lock:
                _stream_clients += 1
            try:
                while not _stop.is_set():
                    with _jpeg_lock:
                        jpg = _latest_jpeg
                    if jpg is None:
                        time.sleep(0.03)
                        continue
                    self.wfile.write(f"--{BOUNDARY}\r\n".encode())
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(jpg)}\r\n\r\n".encode()
                    )
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                    time.sleep(1 / 15.0)
            except (BrokenPipeError, ConnectionResetError):
                return
            finally:
                with _stream_clients_lock:
                    _stream_clients = max(0, _stream_clients - 1)
            return

        if self.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            try:
                while not _stop.is_set():
                    global _bob_pending, _blink_pending
                    global _prayer_start_pending, _prayer_end_pending
                    global _boot_pending, _swipe_pending, _dictation_pending
                    global _clap_tick_pending
                    global _atelier_toggle_pending, _atelier_action_pending
                    global _punch_pending
                    with _state_lock:
                        bob = _bob_pending
                        blink = _blink_pending
                        prayer_start = _prayer_start_pending
                        prayer_end = _prayer_end_pending
                        boot = _boot_pending
                        clap_tick = _clap_tick_pending
                        swipe = _swipe_pending
                        dictation = _dictation_pending
                        atelier_toggle = _atelier_toggle_pending
                        atelier_action = _atelier_action_pending
                        punches = list(_punch_pending)
                        _punch_pending.clear()
                        _bob_pending = False
                        _blink_pending = False
                        _prayer_start_pending = False
                        _prayer_end_pending = False
                        _boot_pending = False
                        _clap_tick_pending = False
                        _swipe_pending = None
                        _dictation_pending = False
                        _atelier_toggle_pending = False
                        _atelier_action_pending = None
                        smile = _smile_val
                        l_y = _left_hand_y
                        r_y = _right_hand_y
                        l_x = _left_hand_x
                        r_x = _right_hand_x
                        prayer_live = _prayer_active
                        jam_on = _jam_mode
                        cursor_on = _cursor_enabled
                        cursor_calibrated = _cursor_calibrated
                    payload = {
                        "smile": round(smile, 3),
                        "leftHandY":  None if l_y is None else round(l_y, 4),
                        "rightHandY": None if r_y is None else round(r_y, 4),
                        "leftHandX":  None if l_x is None else round(l_x, 4),
                        "rightHandX": None if r_x is None else round(r_x, 4),
                        "prayerLive": prayer_live,
                        "cursor": cursor_on,
                        "cursorCalibrated": cursor_calibrated,
                        "jam": jam_on,
                        "pointing": _pointing_method,
                        "click": _click_method,
                        "wispr": _wispr_method,
                        "scrollSens": _scroll_sens,
                        "scrollMode": _scroll_mode,
                        "scrollSpeed": round(_scroll_sens_mult, 2),
                        "rightClick": _right_click_method,
                        "doubleClick": _double_click_on,
                        "cursorSens": round(_cursor_sens, 2),
                        "systemEnabled": _system_enabled,
                        "tTimeout": _t_timeout_enabled,
                        "mouthHold": _mouth_hold_enabled,
                        "peaceRclick": _peace_rclick_enabled,
                        "thumbsDclick": _thumbs_dclick_enabled,
                        "headCopy": _head_copy_enabled,
                        "fistZoom": _fist_zoom_enabled,
                        "atelierEnabled": _atelier_enabled,
                        "atelierMode": _atelier_mode,
                        "releaseExtraTap": _wispr_release_extra_tap,
                        "releaseNuclear": _wispr_release_nuclear,
                        "scrollGesture": _scroll_gesture_enabled,
                        "swipeGesture": _swipe_gesture_enabled,
                        "tabSwipe": _tab_swipe_enabled,
                        "tabSwipeAction": _tab_swipe_action,
                        "clapPreset": _clap_preset,
                        "dictGesture": _dict_gesture,
                        "dictMode": _dict_mode,
                        "voiceDaemon": _voice_enabled,
                        "voiceState": _voice_state,
                        "voiceLastText": _voice_last_text,
                        "voiceLastResult": _voice_last_result,
                        "voiceErr": _voice_err,
                        "voiceTranscript": list(_voice_transcript),
                    }
                    if bob: payload["bob"] = True
                    if blink: payload["blink"] = True
                    if prayer_start: payload["prayerStart"] = True
                    if prayer_end: payload["prayerEnd"] = True
                    if boot: payload["boot"] = True
                    if clap_tick: payload["clapTick"] = True
                    if swipe: payload["swipe"] = swipe
                    if dictation: payload["dictation"] = True
                    if atelier_toggle: payload["atelierToggled"] = True
                    if atelier_action: payload["atelierAction"] = atelier_action
                    if punches: payload["punches"] = punches
                    line = f"data: {json.dumps(payload)}\n\n"
                    try:
                        self.wfile.write(line.encode("utf-8"))
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    time.sleep(1 / 25.0)
            except Exception as e:
                print(f"[viewer] events err: {e}", flush=True)
            return

        self._write_status(404, "text/plain", b"not found")


class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _kill_existing_instance() -> None:
    """If another viewer is already on PORT, kill it so we stay a singleton."""
    try:
        out = subprocess.check_output(
            ["lsof", "-ti", f"tcp:{PORT}"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return
    pids = [p for p in out.split() if p and p.isdigit()
            and int(p) != os.getpid()]
    if not pids:
        return
    print(f"[viewer] killing prior instance(s) on :{PORT}: {pids}", flush=True)
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    for _ in range(20):
        time.sleep(0.1)
        try:
            subprocess.check_output(
                ["lsof", "-ti", f"tcp:{PORT}"], stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            return
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(0.3)


def main() -> None:
    _kill_existing_instance()
    t = threading.Thread(target=_capture_loop, daemon=True)
    t.start()

    srv = ThreadingServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}/"
    print(f"[viewer] wispr hotkey = key code {WISPR_KEYCODE} "
          f"(fn flag={'on' if WISPR_USE_FN_FLAG else 'off'}). "
          "bind Wispr Flow to this key for reliable prayer-hold.", flush=True)
    print(f"[viewer] serving {url} — opening in your browser", flush=True)
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[viewer] shutting down", flush=True)
    finally:
        _stop.set()
        srv.server_close()


if __name__ == "__main__":
    main()
