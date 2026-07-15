#!/usr/bin/env python3
"""
reachy_bridge.py — give the handsfree perception stream a physical body.

Subscribes to the handsfree Server-Sent Events stream (the same 25 Hz feed the
browser UI reads) and translates gesture/voice signals into motion on a Reachy
Mini over its REST API. No new dependencies — standard library only.

First three reactions (step 1):
  • "Hey Wonder" armed  → antennas perk up   (relax again on "bye wonder")
  • voice command fired → a nod
  • jam / party head-bob → the robot bobs its head on every beat

Run it alongside the handsfree daemon:

    python3 reachy_bridge.py

Env overrides:
    HANDSFREE_URL   default http://localhost:8765
    REACHY_URL      default http://reachy-mini.local:8000
    NO_WAKE=1       skip the hello wake-up on start
"""

from __future__ import annotations

import json
import os
import queue
import signal
import sys
import threading
import time
import urllib.error
import urllib.request

HANDSFREE_URL = os.environ.get("HANDSFREE_URL", "http://localhost:8765").rstrip("/")
REACHY_URL = os.environ.get("REACHY_URL", "http://192.168.1.120:8000").rstrip("/")
EVENTS_URL = f"{HANDSFREE_URL}/events"

# Neutral head pose the robot returns to between reactions.
NEUTRAL_HEAD = {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}


# --------------------------------------------------------------------------- #
# Reachy REST helpers                                                         #
# --------------------------------------------------------------------------- #
def _post(path: str, body: dict | None = None, timeout: float = 5.0):
    """POST JSON to the Reachy daemon. Errors are swallowed and logged — a
    flaky robot connection should never take the bridge down."""
    url = f"{REACHY_URL}{path}"
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read() or b"null")
    except Exception as e:  # noqa: BLE001 - deliberately broad; keep bridge alive
        print(f"[reachy] POST {path} failed: {e}", flush=True)
        return None


def robot_reachable() -> bool:
    try:
        with urllib.request.urlopen(f"{REACHY_URL}/api/daemon/status", timeout=4) as r:
            return r.status == 200
    except Exception:
        return False


def enable_motors():
    _post("/api/motors/set_mode/enabled")


def wake_up():
    _post("/api/move/play/wake_up", timeout=15)


def go_to_sleep():
    _post("/api/move/play/goto_sleep", timeout=15)


def goto(head: dict, antennas: list[float], duration: float):
    _post("/api/move/goto", {"head_pose": head, "antennas": antennas, "duration": duration})


# --------------------------------------------------------------------------- #
# Motion controller — a single worker so reactions never stack or fight, and  #
# the SSE read loop never blocks on a robot HTTP call.                         #
# --------------------------------------------------------------------------- #
class Motion:
    """Latest-intent motion worker.

    Reactions are pushed as callables onto a 1-slot queue. If the robot is
    mid-move, a newly arriving low-priority beat is dropped rather than queued,
    so the head never lags behind the music. The persistent antenna posture
    (perked vs relaxed) is remembered so idle moves keep it."""

    def __init__(self):
        self.q: queue.Queue = queue.Queue(maxsize=1)
        self.antenna_base = [0.0, 0.0]     # current resting antenna posture
        self.lock = threading.Lock()
        self.stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, fn, coalesce=True):
        """Queue a motion. If coalesce and the slot is full, drop this one."""
        if coalesce:
            try:
                self.q.put_nowait(fn)
            except queue.Full:
                pass
        else:
            self.q.put(fn)

    def _run(self):
        while not self.stop.is_set():
            try:
                fn = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                print(f"[motion] {e}", flush=True)

    # -- reactions ---------------------------------------------------------- #
    def set_antenna_base(self, base: list[float]):
        with self.lock:
            self.antenna_base = list(base)

    def perk(self):
        """Antennas spring up and forward — 'I'm listening'."""
        self.set_antenna_base([0.9, -0.9])
        def _do():
            goto(NEUTRAL_HEAD, [0.9, -0.9], 0.4)
        self.submit(_do, coalesce=False)

    def relax(self):
        """Antennas settle back to neutral — 'standby'."""
        self.set_antenna_base([0.0, 0.0])
        def _do():
            goto(NEUTRAL_HEAD, [0.0, 0.0], 0.5)
        self.submit(_do, coalesce=False)

    def nod(self):
        """A single deliberate nod — 'got it'."""
        base = list(self.antenna_base)
        def _do():
            down = dict(NEUTRAL_HEAD, pitch=0.30)
            goto(down, base, 0.28)
            time.sleep(0.30)
            goto(NEUTRAL_HEAD, base, 0.28)
        self.submit(_do, coalesce=False)

    def bob(self):
        """A quick head-bob dip on a music beat. Coalesced so fast beats don't
        pile up — better to skip a beat than to fall behind."""
        base = list(self.antenna_base)
        def _do():
            # antennas swing opposite the head for a livelier bounce
            a = [base[0] + 0.35, base[1] - 0.35]
            goto(dict(NEUTRAL_HEAD, pitch=0.22), a, 0.14)
            time.sleep(0.14)
            goto(NEUTRAL_HEAD, base, 0.16)
        self.submit(_do, coalesce=True)


# --------------------------------------------------------------------------- #
# SSE reader + edge detection                                                 #
# --------------------------------------------------------------------------- #
def stream_events(motion: Motion, stop: threading.Event):
    prev_armed = False
    prev_result = None

    while not stop.is_set():
        try:
            req = urllib.request.Request(EVENTS_URL, headers={"Accept": "text/event-stream"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                print(f"[bridge] connected to {EVENTS_URL}", flush=True)
                for raw in resp:
                    if stop.is_set():
                        return
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    try:
                        d = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue

                    # 1. "Hey Wonder" arm / disarm → perk / relax
                    armed = bool(d.get("v2Armed"))
                    if armed and not prev_armed:
                        print("[bridge] armed → perk", flush=True)
                        motion.perk()
                    elif not armed and prev_armed:
                        print("[bridge] standby → relax", flush=True)
                        motion.relax()
                    prev_armed = armed

                    # 2. Voice command fired → nod
                    #    voiceLastResult changes to a real action (not "(no match)")
                    result = d.get("voiceLastResult")
                    if (result and result != prev_result
                            and result != "(no match)"):
                        print(f"[bridge] command '{result}' → nod", flush=True)
                        motion.nod()
                    prev_result = result

                    # 3. Jam / party head-bob beat → bob
                    if d.get("bob") and d.get("jam"):
                        motion.bob()

        except urllib.error.URLError as e:
            print(f"[bridge] handsfree stream unavailable ({e}); retrying in 2s", flush=True)
            time.sleep(2)
        except Exception as e:  # noqa: BLE001
            print(f"[bridge] stream error: {e}; retrying in 2s", flush=True)
            time.sleep(2)


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    print(f"[bridge] handsfree = {HANDSFREE_URL}", flush=True)
    print(f"[bridge] reachy    = {REACHY_URL}", flush=True)

    if not robot_reachable():
        print(f"[bridge] ⚠ can't reach Reachy at {REACHY_URL} — is it awake and on "
              f"the same network? Set REACHY_URL to override.", flush=True)
        sys.exit(1)

    enable_motors()
    if not os.environ.get("NO_WAKE"):
        print("[bridge] waking up …", flush=True)
        wake_up()
        time.sleep(2.5)

    motion = Motion()
    motion.relax()  # start from a known neutral posture

    stop = threading.Event()

    def _shutdown(*_):
        print("\n[bridge] shutting down — sending robot to sleep", flush=True)
        stop.set()
        motion.stop.set()
        try:
            go_to_sleep()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    stream_events(motion, stop)


if __name__ == "__main__":
    main()
