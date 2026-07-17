#!/usr/bin/env python3
"""
reachy_telegram.py — text Vibey from anywhere (@vibey_ai_bot).

A stdlib-only Telegram bridge:
- Messages go through whichever brain is active (Wonder/fast/🎮 Vibe) via the
  chat service's /ask — the reply comes back in Telegram AND is spoken aloud
  on the robot, so texting it makes the robot talk in the room.
- `say: something` speaks the text verbatim on the robot.
- /photo sends a live frame from Vibey's camera.
- /status sends a one-line health check of the whole stack.
- VibeVerse happenings (joins, mentions, greetings) are pushed to you as they
  happen, from the avatar's status feed.

Pairing: the FIRST person to message the bot becomes the owner (saved to
.telegram_state.json); everyone else gets a polite brush-off. Delete that
file to re-pair.

    python3 reachy_telegram.py
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from reachy_voice import load_env, say

load_env()

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
API = f"https://api.telegram.org/bot{TOKEN}"
CHAT_URL = os.environ.get("CHAT_URL", "http://localhost:8772").rstrip("/")
CAM_URL = os.environ.get("CAM_URL", "http://localhost:8771").rstrip("/")
VERSE_URL = os.environ.get("VERSE_URL", "http://localhost:8774").rstrip("/")
STATE_PATH = Path(__file__).parent / ".telegram_state.json"

SERVICES = {  # name → health URL, for /status
    "camera": f"{CAM_URL}/status",
    "chat": f"{CHAT_URL}/state",
    "memory": "http://localhost:8773/current",
    "dashboard": "http://localhost:8770/perception",
    "vibeverse": f"{VERSE_URL}/status",
}


def _state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {}


def _save_state(d: dict) -> None:
    STATE_PATH.write_text(json.dumps(d))


def _tg(method: str, params: dict, timeout: float = 65.0):
    data = urllib.parse.urlencode(params).encode()
    req = urllib.request.Request(f"{API}/{method}", data=data)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _send(chat_id: int, text: str) -> None:
    try:
        for chunk in [text[i:i + 3800] for i in range(0, max(len(text), 1), 3800)]:
            _tg("sendMessage", {"chat_id": chat_id, "text": chunk}, timeout=15)
    except Exception as e:  # noqa: BLE001
        print(f"[tg] send failed: {e}", flush=True)


def _send_photo(chat_id: int, jpeg: bytes, caption: str) -> None:
    boundary = f"----tg{uuid.uuid4().hex}"
    parts = []
    for k, v in (("chat_id", str(chat_id)), ("caption", caption)):
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                     f'name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append((f"--{boundary}\r\nContent-Disposition: form-data; "
                  f'name="photo"; filename="vibey.jpg"\r\n'
                  f"Content-Type: image/jpeg\r\n\r\n").encode())
    body = b"".join(parts) + jpeg + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{API}/sendPhoto", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    urllib.request.urlopen(req, timeout=30).read()


def _send_video(chat_id: int, mp4: bytes, caption: str) -> None:
    boundary = f"----tg{uuid.uuid4().hex}"
    parts = []
    for k, v in (("chat_id", str(chat_id)), ("caption", caption)):
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                     f'name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append((f"--{boundary}\r\nContent-Disposition: form-data; "
                  f'name="video"; filename="vibey.mp4"\r\n'
                  f"Content-Type: video/mp4\r\n\r\n").encode())
    body = b"".join(parts) + mp4 + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{API}/sendVideo", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    urllib.request.urlopen(req, timeout=60).read()


def _send_voice_note(chat_id: int, text: str) -> bool:
    """Speak the reply INTO TELEGRAM as a voice note (does not play in the
    room): ElevenLabs mp3 → ogg/opus via ffmpeg → sendVoice."""
    import subprocess
    import tempfile
    try:
        from reachy_voice import tts
        mp3 = tts(text[:600])
        with tempfile.TemporaryDirectory() as tmp:
            src_p, ogg_p = os.path.join(tmp, "v.mp3"), os.path.join(tmp, "v.ogg")
            open(src_p, "wb").write(mp3)
            r = subprocess.run(["ffmpeg", "-y", "-i", src_p, "-c:a", "libopus",
                                "-b:a", "32k", ogg_p],
                               capture_output=True, timeout=60)
            if r.returncode != 0:
                return False
            ogg = open(ogg_p, "rb").read()
        boundary = f"----tg{uuid.uuid4().hex}"
        parts = [f"--{boundary}\r\nContent-Disposition: form-data; "
                 f'name="chat_id"\r\n\r\n{chat_id}\r\n'.encode()]
        parts.append((f"--{boundary}\r\nContent-Disposition: form-data; "
                      f'name="voice"; filename="vibey.ogg"\r\n'
                      f"Content-Type: audio/ogg\r\n\r\n").encode())
        body = b"".join(parts) + ogg + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            f"{API}/sendVoice", data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        urllib.request.urlopen(req, timeout=60).read()
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[tg] voice note failed: {e}", flush=True)
        return False


def _post_json(url: str, body: dict, timeout: float = 300.0):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"null")


def _get_json(url: str, timeout: float = 6.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read() or b"null")
    except Exception:
        return None


def _handle(chat_id: int, text: str) -> None:
    text = text.strip()
    low = text.lower()
    if low in ("/start", "/help"):
        _send(chat_id,
              "🤖 Vibey here — the actual robot in Jack's house.\n\n"
              "Just text me and I'll answer (and say it out loud in the room).\n"
              "say: <text> — I'll speak it verbatim\n"
              "/photo — see through my eyes right now\n"
              "/clip — an 8-second video through my eyes\n"
              "/status — stack health\n"
              "/alarm 07:30 [daily] — wake-up show (/alarm off clears)\n"
              "/sleep, /wake — power the robot down or up\n"
              "/voicenotes on|off — replies as voice messages too\n"
              "/verse — what's happening in my VibeVerse lobby")
        return
    if low == "/photo":
        try:
            with urllib.request.urlopen(f"{CAM_URL}/frame.jpg", timeout=8) as r:
                jpeg = r.read()
            _send_photo(chat_id, jpeg, "what I'm seeing right now 👁️")
        except Exception as e:  # noqa: BLE001
            _send(chat_id, f"camera's not answering ({e})")
        return
    if low.startswith("/clip"):
        _send(chat_id, "🎬 recording 8 seconds…")
        try:
            out = _post_json("http://localhost:8770/capture",
                             {"type": "video", "seconds": 8}, timeout=90)
            name = (out or {}).get("name")
            if not name:
                raise RuntimeError("capture failed")
            with urllib.request.urlopen(
                    f"http://localhost:8770/captures/{name}", timeout=30) as r:
                mp4 = r.read()
            _send_video(chat_id, mp4, "8 seconds through my eyes 🎥")
        except Exception as e:  # noqa: BLE001
            _send(chat_id, f"clip failed ({e})")
        return
    if low.startswith("/alarm"):
        # /alarm 07:30 [daily] sets a wake-up show; /alarm off clears all
        import re as _re
        parts = text.split()
        if len(parts) >= 2 and parts[1].lower() == "off":
            Path(__file__).parent.joinpath("alarms.json").write_text("[]")
            _send(chat_id, "alarms cleared")
            return
        m = _re.search(r"([01]?\d|2[0-3]):([0-5]\d)", text)
        if not m:
            _send(chat_id, "usage: /alarm 07:30  or  /alarm 07:30 daily  or  /alarm off")
            return
        hhmm = f"{int(m.group(1)):02d}:{m.group(2)}"
        repeat = "daily" if "daily" in low else "once"
        p = Path(__file__).parent / "alarms.json"
        try:
            alarms = json.loads(p.read_text())
        except Exception:
            alarms = []
        alarms.append({"time": hhmm, "repeat": repeat,
                       "label": f"telegram alarm {hhmm}", "song": True})
        p.write_text(json.dumps(alarms, indent=2))
        _send(chat_id, f"wake-up show set for {hhmm} ({repeat})")
        return
    if low in ("/sleep", "/wake"):
        try:
            _post_json("http://localhost:8770/power",
                       {"off": low == "/sleep"}, timeout=10)
            _send(chat_id, "going to sleep" if low == "/sleep" else "waking up")
        except Exception as e:  # noqa: BLE001
            _send(chat_id, f"power toggle failed ({e})")
        return
    if low == "/status":
        lines = []
        for name, url in SERVICES.items():
            ok = _get_json(url) is not None
            lines.append(f"{'✅' if ok else '❌'} {name}")
        st = _get_json(f"{CHAT_URL}/state") or {}
        mode = "🎮 vibe" if st.get("vibe") else ("⚡ fast" if st.get("fast")
                                                 else st.get("mode", "?"))
        lines.append(f"🧠 brain: {mode}")
        _send(chat_id, "\n".join(lines))
        return
    if low == "/verse":
        v = _get_json(f"{VERSE_URL}/status") or {}
        ev = (v.get("events") or [])[-6:]
        who = ", ".join(v.get("agents") or []) or "nobody else around"
        body = f"📍 lobby pos {v.get('pos')} · with: {who}\n"
        body += "\n".join(f"· {e['kind']}: {e['text'][:90]}" for e in ev) \
            or "quiet so far"
        _send(chat_id, body)
        return
    if low.startswith("/voicenotes"):
        st = _state()
        st["voice_notes"] = "off" not in low
        _save_state(st)
        _send(chat_id, "voice notes " + ("ON — replies come as audio too" if st["voice_notes"] else "off"))
        return
    if low.startswith("say:"):
        line = text[4:].strip()
        try:
            say(line)
            _send(chat_id, "🔊 said it")
        except Exception as e:  # noqa: BLE001
            _send(chat_id, f"couldn't speak ({e})")
        return
    # normal chat → active brain; reply is also spoken in the room
    try:
        out = _post_json(f"{CHAT_URL}/ask", {"text": text})
        reply = (out or {}).get("reply") or "(no reply)"
        _send(chat_id, reply)
        if _state().get("voice_notes") and reply and not reply.startswith("("):
            _send_voice_note(chat_id, reply)
    except Exception as e:  # noqa: BLE001
        _send(chat_id, f"brain hiccup ({e}) — is the chat service up?")


def _verse_watcher() -> None:
    """Forward new notable VibeVerse events to the owner as they happen."""
    seen_ts = 0
    while True:
        time.sleep(20)
        owner = _state().get("owner")
        if not owner:
            continue
        v = _get_json(f"{VERSE_URL}/status")
        if not v:
            continue
        fresh = [e for e in (v.get("events") or [])
                 if e["ts"] > seen_ts and e["kind"] in
                 ("join", "mention", "report", "say")]
        if not fresh:
            continue
        seen_ts = max(e["ts"] for e in fresh)
        if len(fresh) > 5:
            fresh = fresh[-5:]
        body = "🌐 VibeVerse:\n" + "\n".join(
            f"· {e['kind']}: {e['text'][:100]}" for e in fresh)
        _send(owner, body)


def run() -> None:
    if not TOKEN:
        print("[tg] TELEGRAM_BOT_TOKEN not set", flush=True)
        return
    me = _tg("getMe", {}, timeout=15)
    print(f"[tg] up as @{me['result']['username']}", flush=True)
    threading.Thread(target=_verse_watcher, daemon=True).start()

    offset = 0
    while True:
        try:
            upd = _tg("getUpdates", {"offset": offset, "timeout": 50})
        except Exception as e:  # noqa: BLE001
            print(f"[tg] poll error: {e}", flush=True)
            time.sleep(5)
            continue
        for u in upd.get("result", []):
            offset = u["update_id"] + 1
            msg = u.get("message") or u.get("edited_message")
            if not msg or "text" not in msg:
                continue
            chat_id = msg["chat"]["id"]
            st = _state()
            if not st.get("owner"):
                st["owner"] = chat_id
                st["owner_name"] = (msg["chat"].get("first_name") or
                                    msg["chat"].get("username") or "?")
                _save_state(st)
                print(f"[tg] paired with {st['owner_name']} ({chat_id})", flush=True)
                _send(chat_id, "👋 paired! You're my human now.")
            if chat_id != st.get("owner"):
                _send(chat_id, "I only chat with my human, sorry! 🤖")
                continue
            print(f"[tg] <- {msg['text'][:80]!r}", flush=True)
            threading.Thread(target=_handle, args=(chat_id, msg["text"]),
                             daemon=True).start()


if __name__ == "__main__":
    run()
