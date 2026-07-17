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
              "/status — stack health\n"
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
