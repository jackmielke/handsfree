#!/usr/bin/env python3
"""
reachy_voice.py — give Wonder a voice (ElevenLabs → the robot's own speaker).

Text goes to ElevenLabs TTS, the resulting MP3 is uploaded to the Reachy daemon,
and played on the robot's speaker — so the sound comes out of Wonder's body, not
your laptop. Stdlib only; no SDK/venv required.

CLI:
    python3 reachy_voice.py "Hey, I'm Vibey. Nice to meet you."

As a module:
    from reachy_voice import say
    say("Let's jam.")

Config comes from .env (see keys below) or the environment:
    ELEVENLABS_API_KEY   required
    ELEVEN_VOICE_ID      required — which ElevenLabs voice to speak in
    REACHY_URL           default http://192.168.1.120:8000
"""

from __future__ import annotations

import json
import mimetypes
import os
import sys
import time
import urllib.request
import uuid
from pathlib import Path


def load_env(path: str = ".env") -> None:
    """Minimal .env loader — populates os.environ for any KEY=VALUE lines
    that aren't already set. Avoids a python-dotenv dependency."""
    p = Path(__file__).parent / path
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


load_env()

REACHY_URL = os.environ.get("REACHY_URL", "http://192.168.1.120:8000").rstrip("/")
ELEVEN_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
VOICE_ID = os.environ.get("ELEVEN_VOICE_ID", "")
ELEVEN_MODEL = os.environ.get("ELEVEN_MODEL", "eleven_multilingual_v2")


def tts(text: str) -> bytes:
    """Return MP3 audio bytes for `text` from ElevenLabs."""
    if not ELEVEN_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY not set (see .env)")
    if not VOICE_ID:
        raise RuntimeError("ELEVEN_VOICE_ID not set (see .env)")
    url = (f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
           f"?output_format=mp3_44100_128")
    body = json.dumps({
        "text": text,
        "model_id": ELEVEN_MODEL,
        "voice_settings": {"stability": 0.4, "similarity_boost": 0.8},
    }).encode()
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "xi-api-key": ELEVEN_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def _multipart(field: str, filename: str, data: bytes, ctype: str):
    """Build a minimal multipart/form-data body (no external deps)."""
    boundary = f"----wonder{uuid.uuid4().hex}"
    pre = (f"--{boundary}\r\n"
           f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
           f"Content-Type: {ctype}\r\n\r\n").encode()
    post = f"\r\n--{boundary}--\r\n".encode()
    return pre + data + post, boundary


def upload_sound(mp3: bytes, name: str) -> str:
    """Upload MP3 to the daemon; returns the server-side filename to play."""
    payload, boundary = _multipart("file", name, mp3, "audio/mpeg")
    req = urllib.request.Request(
        f"{REACHY_URL}/api/media/sounds/upload", data=payload, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        json.loads(r.read() or b"null")  # {"status":"ok","path":...}
    return name


def play_sound(name: str) -> None:
    req = urllib.request.Request(
        f"{REACHY_URL}/api/media/play_sound",
        data=json.dumps({"file": name}).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        r.read()


def say(text: str, wait: bool = False) -> float:
    """Speak `text` on the robot. Returns the clip duration in seconds —
    accurate, from the MP3 byte length (128kbps CBR → 16000 bytes/sec), so
    callers can mute the mic for exactly as long as the robot is talking.
    If `wait`, also block for roughly that long."""
    mp3 = tts(text)
    duration = len(mp3) / 16000.0
    name = f"wonder_{uuid.uuid4().hex[:8]}.mp3"
    upload_sound(mp3, name)
    play_sound(name)
    if wait:
        time.sleep(max(1.0, duration))
    return duration


if __name__ == "__main__":
    msg = " ".join(sys.argv[1:]) or "Hey, I'm Vibey. Nice to meet you."
    print(f"[voice] speaking: {msg!r}", flush=True)
    say(msg)
    print("[voice] done", flush=True)
