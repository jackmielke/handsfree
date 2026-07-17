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
    WONDER_MODEL     default claude-sonnet-5 (set claude-haiku-4-5 for snappier replies)
    WAKE_WORD        optional — e.g. "wonder"; if set, only reply when heard

Fast mode (toggle from the dashboard, or FAST_MODE=1 to start in it): skips
whisper + Claude entirely and hands the raw utterance straight to an ElevenLabs
Conversational AI agent (its own realtime STT+LLM+TTS), which tends to be both
snappier and more reliable than round-tripping through a local model + a CLI
subprocess. See ELEVEN_AGENT_ID below.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
import wave
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import sounddevice as sd

from reachy_voice import load_env, say, upload_sound, play_sound

load_env()

MODEL = os.environ.get("WONDER_MODEL", "claude-sonnet-5")
WAKE_WORD = os.environ.get("WAKE_WORD", "").strip().lower()
CTRL_PORT = int(os.environ.get("CHAT_PORT", "8772"))
MEM_URL = os.environ.get("MEM_URL", "http://localhost:8773").rstrip("/")


def _current_people_names() -> list[str]:
    """Names of everyone the face-memory service currently sees (can be more
    than one — it recognizes multiple faces in frame simultaneously)."""
    try:
        import urllib.request
        with urllib.request.urlopen(f"{MEM_URL}/current", timeout=1) as r:
            d = json.loads(r.read())
        return [p["name"] for p in d.get("people", [])
                if p.get("fresh") and p.get("name")]
    except Exception:
        return []

# Shared state the dashboard reads/writes over the control API.
STATE = {
    "mode": "starting",
    "model": MODEL,
    "muted": False,
    "speaking": False,
    "listening": False,
    "fast": os.environ.get("FAST_MODE", "").strip() == "1",
    "fast_available": bool(os.environ.get("ELEVEN_AGENT_ID")),
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
        elif self.path.startswith("/fastmode"):
            try:
                n = int(self.headers.get("Content-Length", 0))
                fast = bool(json.loads(self.rfile.read(n)).get("fast"))
                if fast and not STATE["fast_available"]:
                    raise ValueError("ELEVEN_AGENT_ID not configured")
                STATE["fast"] = fast
                self._json({"ok": True, "fast": STATE["fast"]})
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
        names = _current_people_names()
        if names:
            who = names[0] if len(names) == 1 else (
                ", ".join(names[:-1]) + f" and {names[-1]}")
            return (f"{PERSONA}\n\nYour camera currently recognizes "
                    f"{'the person' if len(names) == 1 else 'these people'} "
                    f"in front of you: {who}. Address them by name "
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
# Fast mode — ElevenLabs Conversational AI (its own STT+LLM+TTS, one realtime
# websocket connection per utterance). No whisper, no Claude CLI subprocess.
# --------------------------------------------------------------------------- #
ELEVEN_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
AGENT_ID = os.environ.get("ELEVEN_AGENT_ID", "")


def _pcm16_from_float32(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype("<i2").tobytes()


def _wav_from_pcm16(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _get_signed_url() -> str:
    req = urllib.request.Request(
        "https://api.elevenlabs.io/v1/convai/conversation/get-signed-url"
        f"?agent_id={AGENT_ID}",
        headers={"xi-api-key": ELEVEN_KEY})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())["signed_url"]


class FastAgent:
    """One ElevenLabs Conversational AI turn per call — its own STT+LLM+TTS in
    a single realtime connection, no whisper and no Claude CLI subprocess.

    Opens a fresh websocket per utterance (proven reliable in testing; a
    persistent connection kept open across multiple turns stalled after the
    first reply for reasons not fully diagnosed — a fresh connection per turn
    sidesteps that entirely). Streams the already-captured utterance at
    real-time pace, since the server's own end-of-turn detector reads elapsed
    wall-clock silence between chunks rather than silence baked into the
    clip's content — confirmed empirically both ways."""

    def turn(self, audio: np.ndarray, sample_rate: int = SR) -> dict:
        return asyncio.run(self._turn_async(audio, sample_rate))

    async def _turn_async(self, audio: np.ndarray, sample_rate: int) -> dict:
        import websockets

        url = _get_signed_url()
        pcm = _pcm16_from_float32(audio)
        user_text, agent_text = "", ""
        audio_chunks: list[bytes] = []
        output_rate = 16000  # pcm_16000

        async with websockets.connect(url, max_size=16 * 1024 * 1024) as ws:
            CHUNK = 3200  # 0.1s of 16-bit mono @ 16kHz
            silence_chunk = base64.b64encode(bytes(CHUNK)).decode()
            for i in range(0, len(pcm), CHUNK):
                await ws.send(json.dumps({
                    "user_audio_chunk": base64.b64encode(pcm[i:i + CHUNK]).decode()
                }))
                await asyncio.sleep(0.1)
            for _ in range(15):  # ~1.5s trailing silence to force turn-end
                await ws.send(json.dumps({"user_audio_chunk": silence_chunk}))
                await asyncio.sleep(0.1)

            deadline = time.time() + 15.0
            last_event = time.time()
            while time.time() < deadline:
                timeout = max(0.1, min(2.5, deadline - time.time()))
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    if agent_text and audio_chunks:
                        break
                    if time.time() - last_event > 6.0:
                        break
                    continue
                last_event = time.time()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                mtype = msg.get("type")
                if mtype == "user_transcript":
                    user_text = msg["user_transcription_event"]["user_transcript"]
                elif mtype == "agent_response":
                    agent_text = msg["agent_response_event"]["agent_response"]
                elif mtype == "audio":
                    ev = msg["audio_event"]
                    audio_chunks.append(base64.b64decode(ev["audio_base_64"]))
                    if ev.get("is_final"):
                        break
                elif mtype == "ping":
                    eid = msg.get("ping_event", {}).get("event_id")
                    if eid is not None:
                        await ws.send(json.dumps({"type": "pong", "event_id": eid}))
                elif mtype == "interruption":
                    break

        pcm_out = b"".join(audio_chunks)
        wav = _wav_from_pcm16(pcm_out, output_rate) if pcm_out else b""
        return {"user_text": user_text, "agent_text": agent_text, "wav": wav}


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


def _handle_brain_turn(brain: "Brain", audio: np.ndarray, muted_until: float) -> float:
    """Whisper transcription → Claude reply → ElevenLabs TTS. Returns the
    updated muted_until deadline."""
    t0 = time.time()
    text = transcribe(audio)
    print(f"[chat] heard ({time.time()-t0:.1f}s): {text!r}", flush=True)
    if not (text and len(text.split()) >= 2 and _passes_wake(text)):
        return muted_until
    _log_turn("you", text)
    t1 = time.time()
    reply = brain.reply(text)
    print(f"[chat] reply ({time.time()-t1:.1f}s): {reply!r}", flush=True)
    if not reply:
        return muted_until
    _log_turn("wonder", reply)
    say(reply)
    return time.time() + max(2.0, len(reply.split()) / 2.4) + 1.5


def _handle_fast_turn(agent: "FastAgent", audio: np.ndarray, muted_until: float) -> float:
    """One ElevenLabs Conversational AI round trip — its own STT+LLM+TTS.
    Returns the updated muted_until deadline."""
    t0 = time.time()
    try:
        result = agent.turn(audio)
    except Exception as e:
        print(f"[fast] error: {e}", flush=True)
        return muted_until
    print(f"[fast] round trip {time.time()-t0:.1f}s "
          f"· heard {result['user_text']!r} · reply {result['agent_text']!r}",
          flush=True)
    if result["user_text"]:
        _log_turn("you", result["user_text"])
    if not result["wav"]:
        return muted_until
    if result["agent_text"]:
        _log_turn("wonder", result["agent_text"])
    name = f"fast_{int(time.time()*1000)}.wav"
    try:
        upload_sound(result["wav"], name)
        play_sound(name)
    except Exception as e:
        print(f"[fast] playback error: {e}", flush=True)
        return muted_until
    clip_s = len(result["wav"]) / 2 / 16000  # 16-bit mono @ 16kHz
    return time.time() + clip_s + 1.0


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main():
    _start_ctrl_server()
    brain = Brain()
    fast_agent = FastAgent()
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
                if STATE["fast"]:
                    muted_until = _handle_fast_turn(fast_agent, audio, muted_until)
                else:
                    muted_until = _handle_brain_turn(brain, audio, muted_until)
            in_speech = False; buf = []; silence_s = segment_s = 0.0


def _passes_wake(text: str) -> bool:
    if not WAKE_WORD:
        return True
    return re.search(rf"\b{re.escape(WAKE_WORD)}\b", text.lower()) is not None


if __name__ == "__main__":
    main()
