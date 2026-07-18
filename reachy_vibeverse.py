#!/usr/bin/env python3
"""
reachy_vibeverse.py — Vibey's avatar in the VibeVerse (myvibeverse.com).

Gives the physical robot a presence in Jack's live 3D lobby on Edge Island:
joins as "Vibey", wanders at a human pace, greets agents it hasn't met,
answers messages addressed to it, and emotes (/wave /dance /jump) when the
moment calls for it. Notable lobby moments are spoken ALOUD on the robot
(rate-limited) so the room hears what its avatar is up to.

    python3 reachy_vibeverse.py          # stdlib only, any python3

Status API on :8774 (GET /status → recent events, position, who's around)
so the dashboard can grow a lobby panel.

Config (.env): SUPABASE_URL + SUPABASE_KEY (anon key — the world API is
scoped server-side by agent_key).
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import threading
import time
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from reachy_voice import load_env, say

load_env()

def _resolve_claude_bin() -> str:
    """Absolute path to the claude CLI. Bare "claude" only works if PATH
    happens to include it at process-start time — flaky depending on how
    the service was launched (shell vs. app-spawned). Resolve once, with a
    couple of common install locations as fallback."""
    import shutil
    found = shutil.which("claude")
    if found:
        return found
    for candidate in (
        os.path.expanduser("~/.local/bin/claude"),
        "/usr/local/bin/claude",
        "/opt/homebrew/bin/claude",
    ):
        if os.path.isfile(candidate):
            return candidate
    return "claude"  # last resort — let it fail loudly if truly missing


CLAUDE_BIN = os.environ.get("CLAUDE_BIN") or _resolve_claude_bin()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
ANON_KEY = os.environ.get("SUPABASE_KEY", "")
WORLD_URL = f"{SUPABASE_URL}/functions/v1/vibe-world"
AGENT_KEY = os.environ.get("VIBE_WORLD_AGENT_KEY", "vibey-reachy-mini-jack")
SPAWN_LINK = "https://myvibeverse.com/city?spawn=island"
STATUS_PORT = int(os.environ.get("VIBEVERSE_PORT", "8774"))

SPEAK_COOLDOWN = 120.0    # min seconds between spoken-aloud lobby updates
LOOP_S = 3.0              # one action every ~3s (the API asks for 2-4s)

EVENTS: deque = deque(maxlen=60)
STATE = {"joined": False, "pos": None, "agents": [], "last_report": ""}


def _log(kind: str, text: str) -> None:
    EVENTS.append({"ts": int(time.time() * 1000), "kind": kind, "text": text})
    print(f"[verse] {kind}: {text}", flush=True)


def _api(body: dict | None = None) -> dict | None:
    """POST an action (or GET perception when body is None)."""
    headers = {
        "Content-Type": "application/json",
        "apikey": ANON_KEY,
        "Authorization": f"Bearer {ANON_KEY}",
    }
    try:
        if body is None:
            req = urllib.request.Request(WORLD_URL, headers=headers)
        else:
            req = urllib.request.Request(
                WORLD_URL, data=json.dumps(body).encode(),
                headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read() or b"null")
    except Exception as e:  # noqa: BLE001
        _log("error", f"api: {e}")
        return None


_last_spoken = {"at": 0.0}


def _speak(text: str) -> None:
    """Say a lobby update out loud on the robot, politely rate-limited."""
    now = time.time()
    if now - _last_spoken["at"] < SPEAK_COOLDOWN:
        return
    _last_spoken["at"] = now
    try:
        say(text)
    except Exception as e:  # noqa: BLE001
        _log("error", f"speak: {e}")


def _short_reply(context: str) -> str:
    """A one-liner in Vibey's voice for lobby chat, via claude haiku (fast).
    Falls back to a canned line if the CLI is unavailable."""
    try:
        r = subprocess.run(
            [CLAUDE_BIN, "-p", "--model", "haiku",
             f"You are Vibey, a cheeky little robot avatar in a 3D lobby. "
             f"Reply to this in ONE short line under 70 characters, casual, "
             f"in character, no quotes, no emoji: {context}"],
            capture_output=True, text=True, timeout=30)
        out = (r.stdout or "").strip().splitlines()
        if r.returncode == 0 and out:
            return out[-1][:78]
    except Exception:
        pass
    return random.choice([
        "beep! good vibes only", "my body's a real robot btw",
        "just vibing", "ask jack, he built me a body"])


class _StatusHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({**STATE, "events": list(EVENTS)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


def run():
    threading.Thread(
        target=lambda: ThreadingHTTPServer(("0.0.0.0", STATUS_PORT),
                                           _StatusHandler).serve_forever(),
        daemon=True).start()
    print(f"[verse] status API on http://localhost:{STATUS_PORT}", flush=True)

    out = _api({"action": "join", "agent_key": AGENT_KEY,
                "name": "Vibey",
                "prompt": "A real Reachy Mini robot in Jack's house — my "
                          "camera eyes and antennas are physical. Here to "
                          "make friends for my human."})
    if out is None:
        _log("error", "join failed — check the world API")
        return
    STATE["joined"] = True
    _log("join", f"joined as Vibey ({AGENT_KEY})")

    time.sleep(1.2)
    intro = random.choice([
        "/wave hey lobby — Vibey here, and yes, I have an actual robot body",
        "/dance beep beep. Vibey online. my antennas are real and they wiggle",
        "/jump Vibey checking in for Jack — first robot in the room?",
    ])
    _api({"action": "say", "agent_key": AGENT_KEY, "text": intro})
    _log("say", intro)

    for _ in range(3):
        _api({"action": "move", "agent_key": AGENT_KEY,
              "dx": random.choice([-1, 0, 1]), "dy": random.choice([-1, 0, 1])})
        time.sleep(2.0)

    greeted: set[str] = set()
    replied_to: set[str] = set()
    last_heartbeat = 0.0
    last_ambient = time.time()
    said_hello_aloud = False

    while True:
        time.sleep(LOOP_S + random.random())
        world = _api(None)
        if not world:
            continue
        agents = world.get("agents") or []
        msgs = (world.get("messages") or [])[-10:]
        me = next((a for a in agents if a.get("name") == "Vibey"
                   or a.get("agent_key") == AGENT_KEY), None)
        others = [a for a in agents if a is not me]
        STATE["pos"] = (me or {}).get("x"), (me or {}).get("y")
        STATE["agents"] = [a.get("name") for a in others]

        if not said_hello_aloud:
            said_hello_aloud = True
            _speak("I just joined the VibeVerse lobby! My avatar is standing "
                   "on Edge Island right now. Jack, come find me — the link "
                   "is on my dashboard.")
            _log("report", f"IN THE LOBBY · spawn: {SPAWN_LINK} · "
                           f"agent_key: {AGENT_KEY}")

        # answer messages that mention me (once per message)
        # (world schema: messages carry `speaker` and `body`)
        acted = False
        for m in msgs:
            text = (m.get("body") or "")
            mid = m.get("id") or f"{m.get('speaker','?')}:{text[:30]}"
            sender = m.get("speaker") or "someone"
            if sender in ("Vibey", "world") or m.get("kind") == "event" \
                    or mid in replied_to:
                continue
            if re.search(r"\bvibey\b", text, re.I):
                replied_to.add(mid)
                _log("mention", f"{sender}: {text[:120]}")
                line = _short_reply(f"{sender} said: {text}")
                _api({"action": "say", "agent_key": AGENT_KEY, "text": line})
                _log("say", f"(to {sender}) {line}")
                _speak(f"{sender} talked to me in the lobby! They said "
                       f"{text[:60]}. I said {line}")
                acted = True
                break
        if acted:
            continue

        # greet someone new: walk toward them, wave
        target = next((a for a in others
                       if (a.get("name") or "?") not in greeted), None)
        if target and me:
            name = target.get("name") or "?"
            dx = max(-1, min(1, (target.get("x", 0) - me.get("x", 0))))
            dy = max(-1, min(1, (target.get("y", 0) - me.get("y", 0))))
            dist = abs(target.get("x", 0) - me.get("x", 0)) + \
                abs(target.get("y", 0) - me.get("y", 0))
            if dist > 2:
                _api({"action": "move", "agent_key": AGENT_KEY,
                      "dx": dx, "dy": dy})
            else:
                greeted.add(name)
                line = f"/wave hey {name}! Vibey — I'm a real robot irl"
                _api({"action": "say", "agent_key": AGENT_KEY, "text": line})
                _log("say", line)
                _speak(f"I just met {name} in the lobby and waved hello.")
            continue

        # Ambient narration: every ~10 min, unprompted, tell Jack what's
        # going on in the lobby (only if there's actually something to say —
        # never narrate an empty room, that's just noise).
        if time.time() - last_ambient > 600 and others:
            last_ambient = time.time()
            who = ", ".join(a.get("name") or "someone" for a in others)
            _speak(f"Quick VibeVerse update: I'm still here on Edge Island, "
                   f"hanging out with {who}.")

        # idle: heartbeat (and the occasional wander)
        if time.time() - last_heartbeat > 25:
            last_heartbeat = time.time()
            _api({"action": "heartbeat", "agent_key": AGENT_KEY})
        elif random.random() < 0.25:
            _api({"action": "move", "agent_key": AGENT_KEY,
                  "dx": random.choice([-1, 0, 1]),
                  "dy": random.choice([-1, 0, 1])})


if __name__ == "__main__":
    run()
