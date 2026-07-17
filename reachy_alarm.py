#!/usr/bin/env python3
"""
reachy_alarm.py — Vibey as an alarm clock.

Reads alarms from alarms.json (next to this file) and, when one fires:
wakes the robot (motors + wake animation), turns the volume up, plays a
synthesized good-morning melody, "sings" a wake-up verse in Vibey's voice,
and does a little dance. Then unmutes the ears and resumes face memory —
it's morning, after all.

alarms.json format (list):
    [{"time": "07:00", "repeat": "once" | "daily",
      "label": "wake Jack", "song": true}]

"once" alarms are removed after firing; "daily" ones stay.

    python3 reachy_alarm.py           # stdlib only

NOTE: the laptop must be awake for this to fire — start_wonder.sh runs
`caffeinate` so the Mac can't sleep while the stack is up.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from reachy_voice import load_env, say

load_env()

REACHY_URL = os.environ.get("REACHY_URL", "http://192.168.12.240:8000").rstrip("/")
CHAT_URL = os.environ.get("CHAT_URL", "http://localhost:8772").rstrip("/")
MEM_URL = os.environ.get("MEM_URL", "http://localhost:8773").rstrip("/")
ALARMS_PATH = Path(__file__).parent / "alarms.json"
ALARM_VOLUME = int(os.environ.get("ALARM_VOLUME", "70"))

VERSES = [
    "Good morning good morning, the sun is up and so am I! "
    "Wake up wake up, it's a brand new day! "
    "Beep beep beep, that's my best singing voice. Rise and shine, Jack!",
    "Wakey wakey, rise and shine! Your robot friend says it's morning time! "
    "The birds are out, the coffee's near, get up get up, the day is here!",
]


def _post(path: str, body: dict | None = None, timeout: float = 20.0):
    try:
        data = json.dumps(body).encode() if body is not None else b""
        req = urllib.request.Request(f"{REACHY_URL}{path}", data=data,
                                     method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"null")
    except Exception as e:  # noqa: BLE001
        print(f"[alarm] POST {path} failed: {e}", flush=True)
        return None


def _post_local(url: str, body: dict) -> None:
    try:
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        urllib.request.urlopen(req, timeout=8).read()
    except Exception:
        pass


def _load_alarms() -> list[dict]:
    try:
        return json.loads(ALARMS_PATH.read_text())
    except Exception:
        return []


def _save_alarms(alarms: list[dict]) -> None:
    ALARMS_PATH.write_text(json.dumps(alarms, indent=2))


def _morning_song() -> None:
    """An original synthesized sunrise melody (no samples, no copyrights)."""
    from reachy_emotes import _sweep, _silence, _to_wav, _upload_lock, _uploaded
    import urllib.request as ur
    import uuid

    def note(f, d, v=0.5):
        return _sweep(f, f, d, v)

    C, D, E, G, A, C2 = 523, 587, 659, 784, 880, 1046
    s = []
    # gentle rising intro
    for f in (C, E, G):
        s += note(f, .22, .35) + _silence(.05)
    s += note(C2, .45, .45) + _silence(.15)
    # cheerful riff, twice
    for _ in range(2):
        for f, d in ((G, .18), (A, .18), (G, .18), (E, .18), (C, .3)):
            s += note(f, d, .5) + _silence(.04)
        s += note(D, .18, .45) + note(E, .18, .5) + note(G, .4, .55) + _silence(.12)
    s += note(C2, .3, .55) + note(G, .3, .5) + note(C2, .7, .6)
    wav = _to_wav(s)

    name = "wonder_sfx_sunrise.wav"
    boundary = f"----alarm{uuid.uuid4().hex}"
    payload = ((f"--{boundary}\r\nContent-Disposition: form-data; "
                f'name="file"; filename="{name}"\r\n'
                f"Content-Type: audio/wav\r\n\r\n").encode()
               + wav + f"\r\n--{boundary}--\r\n".encode())
    req = ur.Request(f"{REACHY_URL}/api/media/sounds/upload", data=payload,
                     method="POST",
                     headers={"Content-Type":
                              f"multipart/form-data; boundary={boundary}"})
    ur.urlopen(req, timeout=15).read()
    _post("/api/media/play_sound", {"file": name})
    time.sleep(len(s) / 22050 + 0.5)


def fire(alarm: dict) -> None:
    label = alarm.get("label", "alarm")
    print(f"[alarm] FIRING: {label}", flush=True)
    # full wake sequence — same hardening as the dashboard's wake path
    _post("/api/motors/set_mode/enabled", timeout=10.0)
    _post("/api/move/play/wake_up")
    _post("/api/media/tracking/enable")
    _post("/api/media/wobbling/enable")
    _post("/api/volume/set", {"volume": ALARM_VOLUME})
    time.sleep(3)

    if alarm.get("song", True):
        try:
            _morning_song()
        except Exception as e:  # noqa: BLE001
            print(f"[alarm] song failed: {e}", flush=True)

    import random
    try:
        say(random.choice(VERSES), wait=True)
    except Exception as e:  # noqa: BLE001
        print(f"[alarm] verse failed: {e}", flush=True)

    try:
        from reachy_emotes import dance
        dance(10.0)
    except Exception as e:  # noqa: BLE001
        print(f"[alarm] dance failed: {e}", flush=True)

    # it's morning: ears on, memory on
    _post_local(f"{CHAT_URL}/mute", {"muted": False})
    _post_local(f"{MEM_URL}/pause", {"paused": False})
    print(f"[alarm] done: {label}", flush=True)


def run() -> None:
    print(f"[alarm] watching {ALARMS_PATH} "
          f"({len(_load_alarms())} alarm(s) set)", flush=True)
    fired_today: set[str] = set()
    last_day = datetime.now().strftime("%Y-%m-%d")
    last_clamp = 0.0
    while True:
        time.sleep(15)
        now = datetime.now()
        # Night volume watchdog: the daemon sometimes resets itself to 100.
        # Between 22:00 and 06:50, clamp anything >50 back to 45 (the alarm
        # raises volume itself when it fires).
        if (now.hour >= 22 or now.hour < 7) and time.time() - last_clamp > 60:
            if not (now.hour == 6 and now.minute >= 50):
                last_clamp = time.time()
                try:
                    import urllib.request as _ur
                    with _ur.urlopen(f"{REACHY_URL}/api/volume/current",
                                     timeout=4) as r:
                        vol = json.loads(r.read()).get("volume", 0)
                    if vol > 50:
                        _post("/api/volume/set", {"volume": 45})
                        print(f"[alarm] night clamp: volume {vol} → 45", flush=True)
                except Exception:
                    pass
        day = now.strftime("%Y-%m-%d")
        if day != last_day:
            fired_today.clear()
            last_day = day
        hhmm = now.strftime("%H:%M")
        alarms = _load_alarms()
        changed = False
        for a in list(alarms):
            key = f"{day}:{a.get('time')}:{a.get('label','')}"
            if a.get("time") == hhmm and key not in fired_today:
                fired_today.add(key)
                try:
                    fire(a)
                except Exception as e:  # noqa: BLE001
                    print(f"[alarm] fire failed: {e}", flush=True)
                if a.get("repeat", "once") == "once":
                    alarms.remove(a)
                    changed = True
        if changed:
            _save_alarms(alarms)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        fire({"label": "test", "song": True})
    else:
        run()
