#!/usr/bin/env python3
"""
reachy_robot_mic.py — stream Vibey's OWN microphone to the laptop, so voice
chat hears the room the robot is actually in instead of the laptop's mic.

Connects to the robot over WebRTC (same SDK path as reachy_camera.py),
downmixes the mic's stereo float32 samples to mono, and re-serves them as a
continuous raw PCM16 stream any client can read with a plain HTTP GET —
mirroring reachy_camera.py's MJPEG bridge, but for audio.

    http://localhost:8775/pcm     → raw s16le mono 16kHz, continuous
    http://localhost:8775/status  → {"connected": bool, "samplerate": int}

MUST run inside the SDK venv (needs reachy_mini + GStreamer):

    reachy_env/bin/python3 reachy_robot_mic.py

To make reachy_chat.py listen through this instead of the laptop mic, set
in .env:  MIC_SOURCE=robot

Env overrides:
    REACHY_HOST   derived from REACHY_URL in .env if unset
    ROBOT_MIC_PORT default 8775
"""

from __future__ import annotations

import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from reachy_mini import ReachyMini
from reachy_voice import load_env

load_env()

# This bridge only wants the robot's INCOMING mic audio. The SDK's WebRTC
# client also sets up a SEND chain (streaming this laptop's own mic to the
# robot's speaker) which we have no use for here — disable it, same fix as
# reachy_camera.py, to avoid wasted bandwidth and any barge-in interference.
try:
    from reachy_mini.media.webrtc_client_gstreamer import GstWebRTCClient

    def _no_audio_send(self):  # noqa: ANN001
        self.logger.info("audio send chain disabled (receive-only mic bridge)")

    GstWebRTCClient._setup_audio_send_chain = _no_audio_send
except Exception as _e:  # noqa: BLE001
    print(f"[robotmic] WARNING: could not disable audio send chain: {_e}", flush=True)


def _default_host() -> str:
    host = os.environ.get("REACHY_HOST")
    if host:
        return host
    url = os.environ.get("REACHY_URL", "")
    m = re.match(r"https?://([^:/]+)", url)
    return m.group(1) if m else "192.168.1.120"


REACHY_HOST = _default_host()
MIC_PORT = int(os.environ.get("ROBOT_MIC_PORT", "8775"))
TARGET_SR = 16000  # matches reachy_chat.py's SR — robot mic already 16kHz

_lock = threading.Condition()
_buf = bytearray()
_connected = False
_actual_sr = TARGET_SR


def _capture_loop():
    global _connected, _actual_sr
    while True:
        try:
            print(f"[robotmic] connecting to {REACHY_HOST} …", flush=True)
            mini = ReachyMini(host=REACHY_HOST, connection_mode="network")
            mini.media.start_recording()
            _actual_sr = mini.media.get_input_audio_samplerate() or TARGET_SR
            _connected = True
            print(f"[robotmic] connected — {_actual_sr}Hz, streaming", flush=True)
            misses = 0
            while True:
                sample = mini.media.get_audio_sample()
                if sample is None:
                    misses += 1
                    if misses > 200:
                        raise RuntimeError("mic stream stalled")
                    time.sleep(0.01)
                    continue
                misses = 0
                # (N, channels) float32 → mono int16
                mono = sample.mean(axis=1) if sample.ndim > 1 else sample
                pcm16 = np.clip(mono, -1.0, 1.0)
                pcm16 = (pcm16 * 32767).astype("<i2").tobytes()
                with _lock:
                    _buf.extend(pcm16)
                    # cap buffer so a slow/absent reader can't grow it forever
                    if len(_buf) > TARGET_SR * 2 * 10:  # 10s
                        del _buf[:len(_buf) - TARGET_SR * 2 * 2]
                    _lock.notify_all()
        except Exception as e:  # noqa: BLE001 - keep retrying forever
            _connected = False
            print(f"[robotmic] connection lost ({e}); retrying in 3s", flush=True)
            time.sleep(3)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        if self.path.startswith("/pcm"):
            self._stream()
        elif self.path.startswith("/status"):
            import json
            body = json.dumps({"connected": _connected,
                               "samplerate": _actual_sr}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        pos = 0
        try:
            while True:
                with _lock:
                    while len(_buf) <= pos:
                        _lock.wait(timeout=5)
                        pos = min(pos, len(_buf))  # buffer may have trimmed
                    chunk = bytes(_buf[pos:])
                    pos = len(_buf)
                self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    threading.Thread(target=_capture_loop, daemon=True).start()
    print(f"[robotmic] PCM stream  http://localhost:{MIC_PORT}/pcm", flush=True)
    ThreadingHTTPServer(("0.0.0.0", MIC_PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
