#!/usr/bin/env python3
"""
reachy_chat.py — voice-to-voice conversation with Wonder.

You talk (laptop mic) → faster-whisper transcribes → Claude thinks up a short
reply in Wonder's persona → ElevenLabs renders it in Wonder's voice → it plays
from the robot's speaker. The mic is muted while Wonder speaks so it doesn't
hear itself.

Run in the handsfree venv (has sounddevice + faster-whisper + anthropic):

    .venv/bin/python3 reachy_chat.py

The brain resolves in this order:
  1. ANTHROPIC_API_KEY (from .env or env)  → Anthropic SDK
  2. `claude` CLI                          → works after `claude /login`
  3. echo mode                             → repeats what it heard (no brain)

Config (.env or environment):
    WONDER_MODEL     default claude-opus-4-8 (set claude-haiku-4-5 for snappier replies)
    WAKE_WORD        optional — e.g. "wonder"; if set, only reply when heard
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import sounddevice as sd

from reachy_voice import load_env, say

load_env()

MODEL = os.environ.get("WONDER_MODEL", "claude-opus-4-8")
WAKE_WORD = os.environ.get("WAKE_WORD", "").strip().lower()
CTRL_PORT = int(os.environ.get("CHAT_PORT", "8772"))
MEM_URL = os.environ.get("MEM_URL", "http://localhost:8773").rstrip("/")


def _current_person() -> str | None:
    """Name of whoever the face-memory service currently recognizes."""
    try:
        import urllib.request
        with urllib.request.urlopen(f"{MEM_URL}/current", timeout=1) as r:
            d = json.loads(r.read())
        if d.get("fresh") and d.get("name"):
            return d["name"]
    except Exception:
        pass
    return None

# Shared state the dashboard reads/writes over the control API.
STATE = {
    "mode": "starting",
    "model": MODEL,
    "muted": False,
    "speaking": False,
    "listening": False,
}
TRANSCRIPT: deque = deque(maxlen=40)   # {"who": "you"|"wonder", "text", "ts"}


def _log_turn(who: str, text: str) -> None:
    TRANSCRIPT.append({"who": who, "text": text, "ts": int(time.time() * 1000)})


class _CtrlHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/state"):
            self._json({**STATE, "transcript": list(TRANSCRIPT)})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path.startswith("/mute"):
            try:
                n = int(self.headers.get("Content-Length", 0))
                STATE["muted"] = bool(json.loads(self.rfile.read(n)).get("muted"))
                self._json({"ok": True, "muted": STATE["muted"]})
            except Exception as e:
                self._json({"error": str(e)}, 400)
        else:
            self._json({"error": "not found"}, 404)


def _start_ctrl_server():
    srv = ThreadingHTTPServer(("0.0.0.0", CTRL_PORT), _CtrlHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[chat] control API on http://localhost:{CTRL_PORT}", flush=True)

# --- VAD constants (proven values from browser_viewer.py) --------------------
SR = 16000
BLOCK_S = 0.1
BLOCK_N = int(SR * BLOCK_S)
SPEECH_RMS = 0.012
SILENCE_HANG_S = 0.8          # slightly longer than handsfree's 0.6 — full sentences
PRE_ROLL_S = 0.25
MIN_SEGMENT_S = 0.5           # ignore coughs/taps
MAX_SEGMENT_S = 20.0

PERSONA = (
    "You are Wonder, a small Reachy Mini robot with antennas, sitting on a desk "
    "in Jack's living room. You speak out loud through a speaker in a British-"
    "robot voice, so keep replies SHORT — one or two sentences, conversational, "
    "no lists, no markdown, no emoji. You are curious, playful, and a little "
    "cheeky. You can see through your camera, hear through your mics, remember "
    "faces, and dance when there's music. If asked to do something physical you "
    "can't do yet, be honest but game about it."
)


# --------------------------------------------------------------------------- #
# Brain
# --------------------------------------------------------------------------- #
class Brain:
    def __init__(self):
        self.history: list[dict] = []
        self.mode = "echo"
        self.client = None
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                import anthropic
                self.client = anthropic.Anthropic()
                self.mode = "api"
            except Exception as e:
                print(f"[brain] SDK unavailable: {e}", flush=True)
        if self.mode == "echo" and self._cli_works():
            self.mode = "cli"
        print(f"[brain] mode = {self.mode} ({MODEL})", flush=True)

    @staticmethod
    def _cli_works() -> bool:
        try:
            r = subprocess.run(
                ["claude", "-p", "--model", "haiku", "Say OK"],
                capture_output=True, text=True, timeout=30)
            return r.returncode == 0 and "OK" in r.stdout.upper()
        except Exception:
            return False

    def reply(self, text: str) -> str:
        self.history.append({"role": "user", "content": text})
        self.history = self.history[-12:]  # keep the last few turns
        try:
            if self.mode == "api":
                out = self._api_reply()
            elif self.mode == "cli":
                out = self._cli_reply()
            else:
                out = f"I heard you say: {text}. My brain isn't hooked up yet — run claude login, or give me an API key."
        except Exception as e:
            print(f"[brain] error: {e}", flush=True)
            out = "Hmm, my brain glitched for a second. Say that again?"
        self.history.append({"role": "assistant", "content": out})
        return out

    def _persona(self) -> str:
        who = _current_person()
        if who:
            return (f"{PERSONA}\n\nYour camera currently recognizes the person "
                    f"in front of you: it's {who}. Address them by name "
                    f"naturally (don't overdo it).")
        return PERSONA

    def _api_reply(self) -> str:
        resp = self.client.messages.create(
            model=MODEL,
            max_tokens=200,
            system=self._persona(),
            messages=self.history,
        )
        return next((b.text for b in resp.content if b.type == "text"), "").strip()

    def _cli_reply(self) -> str:
        # Bake persona + short history into one prompt; fresh CLI call per turn.
        convo = "\n".join(
            f"{'Human' if m['role'] == 'user' else 'Wonder'}: {m['content']}"
            for m in self.history)
        prompt = (f"{self._persona()}\n\nConversation so far:\n{convo}\n\n"
                  f"Reply as Wonder with ONLY the spoken sentence(s), nothing else.")
        r = subprocess.run(["claude", "-p", "--model", MODEL, prompt],
                           capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip()[:200])
        return r.stdout.strip()


# --------------------------------------------------------------------------- #
# STT
# --------------------------------------------------------------------------- #
_whisper = None

def transcribe(audio: np.ndarray) -> str:
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        print("[stt] loading whisper tiny.en …", flush=True)
        _whisper = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    segments, _ = _whisper.transcribe(audio, beam_size=1, vad_filter=False)
    return " ".join(s.text.strip() for s in segments).strip()


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main():
    _start_ctrl_server()
    brain = Brain()
    STATE["mode"] = brain.mode
    say("Voice chat is on. Talk to me!")
    muted_until = time.time() + 3.0   # let the greeting finish

    stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                            blocksize=BLOCK_N)
    stream.start()
    print("[chat] listening…", flush=True)

    pre_roll: deque = deque(maxlen=int(PRE_ROLL_S / BLOCK_S))
    buf: list = []
    in_speech = False
    silence_s = segment_s = 0.0

    while True:
        data, _ = stream.read(BLOCK_N)
        now = time.time()
        STATE["speaking"] = now < muted_until
        STATE["listening"] = not STATE["speaking"] and not STATE["muted"]
        if now < muted_until or STATE["muted"]:   # Wonder talking, or user muted
            pre_roll.clear(); buf = []; in_speech = False
            continue

        rms = float(np.sqrt(np.mean(data ** 2)))
        if rms > SPEECH_RMS:
            if not in_speech:
                in_speech = True
                buf = list(pre_roll)
                segment_s = len(buf) * BLOCK_S
            buf.append(data.copy()); segment_s += BLOCK_S; silence_s = 0.0
        else:
            pre_roll.append(data.copy())
            if in_speech:
                buf.append(data.copy()); segment_s += BLOCK_S; silence_s += BLOCK_S

        if in_speech and (silence_s >= SILENCE_HANG_S or segment_s >= MAX_SEGMENT_S):
            if segment_s >= MIN_SEGMENT_S and buf:
                audio = np.concatenate([b[:, 0] for b in buf]).astype(np.float32)
                t0 = time.time()
                text = transcribe(audio)
                print(f"[chat] heard ({time.time()-t0:.1f}s): {text!r}", flush=True)
                if text and len(text.split()) >= 2 and _passes_wake(text):
                    _log_turn("you", text)
                    t1 = time.time()
                    reply = brain.reply(text)
                    print(f"[chat] reply ({time.time()-t1:.1f}s): {reply!r}", flush=True)
                    if reply:
                        _log_turn("wonder", reply)
                        say(reply)
                        # mute mic for roughly the clip length + upload slack
                        muted_until = time.time() + max(2.0, len(reply.split()) / 2.4) + 1.5
            in_speech = False; buf = []; silence_s = segment_s = 0.0


def _passes_wake(text: str) -> bool:
    if not WAKE_WORD:
        return True
    return re.search(rf"\b{re.escape(WAKE_WORD)}\b", text.lower()) is not None


if __name__ == "__main__":
    main()
