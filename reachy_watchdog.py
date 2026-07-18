#!/usr/bin/env python3
"""
reachy_watchdog.py — keep the Vibey stack alive without a human noticing.

Born from the 2026-07-18 incident: chat crashed at 2:44PM and the robot spent
the evening headless (daemon scanning the room = "crazy mode") because nothing
restarts dead services. This does.

Every CHECK_S seconds each service is health-checked (HTTP port, or process
presence for the portless ones). Two consecutive misses → restart it with the
exact interpreter start_wonder.sh uses, and DM Jack on Telegram. A service
that needs more than MAX_RESTARTS restarts in an hour is left down (crash
loop — a human should look) with one final Telegram note.

Run with system python (stdlib only):   python3 reachy_watchdog.py
Started automatically by start_wonder.sh; stop with the rest of the stack.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CHECK_S = 30
MISSES_TO_RESTART = 2          # ~60s dead before we act (rides out slow starts)
MAX_RESTARTS = 4               # per service per rolling hour, then give up
STARTUP_GRACE_S = 45           # leave freshly (re)started services alone


def _load_env() -> None:
    try:
        with open(os.path.join(HERE, ".env")) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    except FileNotFoundError:
        pass
    url = os.environ.get("REACHY_URL", "")
    m = re.match(r"https?://([^:/]+)", url)
    if m:
        os.environ.setdefault("REACHY_HOST", m.group(1))
    # stale HF token 401s public whisper downloads (see start_wonder.sh)
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"


_load_env()

# name → (health url or None, script, interpreter relative to repo, logfile)
SERVICES = {
    "camera":    ("http://localhost:8771/status",  "reachy_camera.py",    "reachy_env/bin/python3", "/tmp/reachy_camera.log"),
    "viewer":    ("http://localhost:8770/perception", "reachy_viewer.py", "python3",                "/tmp/reachy_viewer.log"),
    "chat":      ("http://localhost:8772/state",   "reachy_chat.py",      ".venv/bin/python3",      "/tmp/reachy_chat.log"),
    "robot_mic": ("http://localhost:8775/status",  "reachy_robot_mic.py", "reachy_env/bin/python3", "/tmp/reachy_robot_mic.log"),
    "memory":    ("http://localhost:8773/current", "reachy_memory.py",    "reachy_env/bin/python3", "/tmp/reachy_memory.log"),
    "vibeverse": ("http://localhost:8774/status",  "reachy_vibeverse.py", "python3",                "/tmp/vibeverse.log"),
    "telegram":  (None,                            "reachy_telegram.py",  "python3",                "/tmp/telegram.log"),
    "alarm":     (None,                            "reachy_alarm.py",     "python3",                "/tmp/reachy_alarm.log"),
}

_misses: dict[str, int] = {n: 0 for n in SERVICES}
_restarts: dict[str, list[float]] = {n: [] for n in SERVICES}
_gave_up: set[str] = set()
_started_at: dict[str, float] = {}


def _telegram(text: str) -> None:
    """Best-effort DM to the paired owner. Never raises."""
    try:
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        with open(os.path.join(HERE, ".telegram_state.json")) as f:
            chat_id = json.load(f).get("owner")
        if not (token and chat_id):
            return
        body = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
        urllib.request.urlopen(
            f"https://api.telegram.org/bot{token}/sendMessage", body, timeout=10
        ).read()
    except Exception as e:  # noqa: BLE001
        print(f"[watchdog] telegram notify failed: {e}", flush=True)


def _alive(name: str) -> bool:
    url, script = SERVICES[name][0], SERVICES[name][1]
    if url:
        try:
            urllib.request.urlopen(url, timeout=5).read()
            return True
        except Exception:  # noqa: BLE001
            return False
    # portless services: a live process counts
    r = subprocess.run(["pgrep", "-f", script], capture_output=True)
    return r.returncode == 0


def _restart(name: str) -> None:
    _, script, interp, log = SERVICES[name]
    subprocess.run(["pkill", "-f", script], capture_output=True)
    time.sleep(1)
    interp_path = interp if interp == "python3" else os.path.join(HERE, interp)
    with open(log, "a") as lf:
        lf.write(f"\n--- [watchdog] restart {time.strftime('%F %T')} ---\n")
        lf.flush()
        subprocess.Popen([interp_path, os.path.join(HERE, script)],
                         cwd=HERE, stdout=lf, stderr=lf,
                         start_new_session=True)
    _started_at[name] = time.time()
    print(f"[watchdog] restarted {name}", flush=True)


def main() -> None:
    print(f"[watchdog] guarding {', '.join(SERVICES)} every {CHECK_S}s", flush=True)
    # everything just booted with the stack — give it all a grace window
    now = time.time()
    for n in SERVICES:
        _started_at[n] = now

    while True:
        time.sleep(CHECK_S)
        for name in SERVICES:
            if name in _gave_up:
                continue
            if time.time() - _started_at.get(name, 0) < STARTUP_GRACE_S:
                continue
            if _alive(name):
                _misses[name] = 0
                continue
            _misses[name] += 1
            print(f"[watchdog] {name} unhealthy ({_misses[name]}/{MISSES_TO_RESTART})",
                  flush=True)
            if _misses[name] < MISSES_TO_RESTART:
                continue
            _misses[name] = 0
            cutoff = time.time() - 3600
            _restarts[name] = [t for t in _restarts[name] if t > cutoff]
            if len(_restarts[name]) >= MAX_RESTARTS:
                _gave_up.add(name)
                msg = (f"🚨 Vibey watchdog: {name} crashed {MAX_RESTARTS}x in an "
                       f"hour — giving up on it. Check {SERVICES[name][3]}")
                print(f"[watchdog] {msg}", flush=True)
                _telegram(msg)
                continue
            _restarts[name].append(time.time())
            _restart(name)
            _telegram(f"🩹 Vibey watchdog: {name} was down — restarted it "
                      f"({len(_restarts[name])}/{MAX_RESTARTS} this hour).")


if __name__ == "__main__":
    main()
