#!/usr/bin/env python3
"""
reachy_memory.py — Wonder remembers faces (Supabase-backed).

Watches the robot's face tracking. When a face holds steady in view, it grabs a
snapshot from the camera stream, computes a face embedding, and checks it
against everyone Wonder already knows (stored in Supabase). Returning people get
greeted by name in Wonder's voice; new people get added to memory.

Run in the SDK venv (it shares the camera + needs numpy):

    source reachy_env/bin/activate
    python3 reachy_memory.py

Requires (one-time):
    uv pip install face_recognition       # 128-d face encodings (pulls dlib)

Config (from .env / environment):
    REACHY_URL       default http://192.168.1.120:8000
    CAM_URL          default http://localhost:8771   (reachy_camera.py MJPEG server)
    SUPABASE_URL     required  e.g. https://xxxx.supabase.co
    SUPABASE_KEY     required  service-role key (server-side use)
    MATCH_TOLERANCE  default 0.55  (lower = stricter face match)

Name someone Wonder has met:
    python3 reachy_memory.py --name <face_id> "Jack"
"""

from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
import urllib.request
import warnings
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# face_recognition_models still imports the deprecated pkg_resources; quiet it.
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

from reachy_voice import load_env, say  # reuse .env loader + TTS

load_env()

REACHY_URL = os.environ.get("REACHY_URL", "http://192.168.1.120:8000").rstrip("/")
CAM_URL = os.environ.get("CAM_URL", "http://localhost:8771").rstrip("/")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
MATCH_TOLERANCE = float(os.environ.get("MATCH_TOLERANCE", "0.55"))

STABLE_SECONDS = 1.5      # face must hold this long before we act
GREET_COOLDOWN = 300.0    # don't re-greet a named person within this window
MEM_PORT = int(os.environ.get("MEM_PORT", "8773"))

# Live recognition state, shared with the dashboard via the control API.
CURRENT = {"face_id": None, "name": None, "ts": 0.0}
_current_lock = threading.Lock()


def _set_current(face_id, name):
    with _current_lock:
        CURRENT.update({"face_id": face_id, "name": name, "ts": time.time()})


# --------------------------------------------------------------------------- #
# Supabase REST (PostgREST) — no supabase-py dependency                        #
# --------------------------------------------------------------------------- #
def _sb_headers(extra: dict | None = None) -> dict:
    h = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if extra:
        h.update(extra)
    return h


def sb_get_faces() -> list[dict]:
    url = f"{SUPABASE_URL}/rest/v1/faces?select=id,name,embedding,times_seen"
    req = urllib.request.Request(url, headers=_sb_headers())
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read() or b"[]")


def sb_insert_face(embedding: list[float], snapshot: str) -> dict:
    url = f"{SUPABASE_URL}/rest/v1/faces"
    body = json.dumps({"embedding": embedding, "snapshot": snapshot}).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers=_sb_headers({"Prefer": "return=representation"}))
    with urllib.request.urlopen(req, timeout=10) as r:
        rows = json.loads(r.read() or b"[]")
        return rows[0] if rows else {}


def sb_touch_face(face_id: str, times_seen: int) -> None:
    url = f"{SUPABASE_URL}/rest/v1/faces?id=eq.{face_id}"
    body = json.dumps({
        "times_seen": times_seen + 1,
        "last_seen": datetime.now(timezone.utc).isoformat(),
    }).encode()
    req = urllib.request.Request(url, data=body, method="PATCH",
                                 headers=_sb_headers({"Prefer": "return=minimal"}))
    urllib.request.urlopen(req, timeout=10).read()


def sb_name_face(face_id: str, name: str) -> None:
    url = f"{SUPABASE_URL}/rest/v1/faces?id=eq.{face_id}"
    body = json.dumps({"name": name}).encode()
    req = urllib.request.Request(url, data=body, method="PATCH",
                                 headers=_sb_headers({"Prefer": "return=minimal"}))
    urllib.request.urlopen(req, timeout=10).read()


# --------------------------------------------------------------------------- #
# Perception helpers                                                           #
# --------------------------------------------------------------------------- #
def robot_face() -> dict:
    try:
        req = urllib.request.Request(f"{REACHY_URL}/api/media/tracking/face")
        with urllib.request.urlopen(req, timeout=3) as r:
            return json.loads(r.read()).get("face_target", {})
    except Exception:
        return {}


def snapshot_jpeg() -> bytes | None:
    try:
        with urllib.request.urlopen(f"{CAM_URL}/frame.jpg", timeout=5) as r:
            return r.read()
    except Exception:
        return None


def encode_face(jpeg: bytes):
    """Return (embedding_list, snapshot_data_uri) for the largest face, or None.
    Lazily imports face_recognition so the rest of the module loads without it."""
    try:
        import numpy as np
        import face_recognition
    except ImportError:
        print("[memory] face_recognition/numpy missing — "
              "run: uv pip install face_recognition", flush=True)
        return None
    import base64
    img = face_recognition.load_image_file(io.BytesIO(jpeg))
    boxes = face_recognition.face_locations(img)
    if not boxes:
        return None
    # pick the largest detected face
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[1] - b[3]), reverse=True)
    enc = face_recognition.face_encodings(img, [boxes[0]])
    if not enc:
        return None
    uri = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
    return enc[0].tolist(), uri


def best_match(embedding: list[float], known: list[dict]):
    """Nearest known face within tolerance, else None."""
    try:
        import numpy as np
    except ImportError:
        return None
    if not known:
        return None
    e = np.array(embedding)
    best, best_d = None, 1e9
    for row in known:
        emb = row.get("embedding")
        if not emb:
            continue
        d = float(np.linalg.norm(e - np.array(emb)))
        if d < best_d:
            best, best_d = row, d
    if best is not None and best_d <= MATCH_TOLERANCE:
        return best, best_d
    return None


# --------------------------------------------------------------------------- #
# Control API — lets the dashboard see who Wonder recognizes and name them.   #
# --------------------------------------------------------------------------- #
class _MemHandler(BaseHTTPRequestHandler):
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
        if self.path.startswith("/current"):
            with _current_lock:
                cur = dict(CURRENT)
            cur["fresh"] = (time.time() - cur["ts"]) < 15.0
            self._json(cur)
        elif self.path.startswith("/people"):
            try:
                rows = sb_get_faces()
                self._json([{"id": r["id"], "name": r.get("name"),
                             "times_seen": r.get("times_seen", 1)} for r in rows])
            except Exception as e:
                self._json({"error": str(e)}, 500)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if not self.path.startswith("/name"):
            self._json({"error": "not found"}, 404)
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n))
            name = (body.get("name") or "").strip()[:60]
            face_id = body.get("face_id") or CURRENT.get("face_id")
            if not (name and face_id):
                raise ValueError("need name (and a face in view)")
            sb_name_face(face_id, name)
            _set_current(face_id, name)
            print(f"[memory] named {face_id} -> {name}", flush=True)
            try:
                from reachy_voice import say as _say
                _say(f"Nice to meet you, {name}. I'll remember you.")
            except Exception:
                pass
            self._json({"ok": True, "face_id": face_id, "name": name})
        except Exception as e:
            self._json({"error": str(e)}, 400)


def _start_mem_server():
    srv = ThreadingHTTPServer(("0.0.0.0", MEM_PORT), _MemHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[memory] control API on http://localhost:{MEM_PORT}", flush=True)


# --------------------------------------------------------------------------- #
# Main loop                                                                    #
# --------------------------------------------------------------------------- #
def run():
    if not (SUPABASE_URL and SUPABASE_KEY):
        print("[memory] SUPABASE_URL / SUPABASE_KEY not set — fill them in .env "
              "and run reachy_sql/faces.sql on your project first.", flush=True)
        sys.exit(1)
    _start_mem_server()
    print(f"[memory] watching for faces · supabase={SUPABASE_URL}", flush=True)
    stable_since = None
    last_greet: dict[str, float] = {}
    asked_name: set[str] = set()   # unnamed faces we've already asked — once per run

    while True:
        f = robot_face()
        now = time.time()
        if not f.get("detected"):
            stable_since = None
            time.sleep(0.3)
            continue
        if stable_since is None:
            stable_since = now
        if now - stable_since < STABLE_SECONDS:
            time.sleep(0.2)
            continue

        # A face has been steady — identify it.
        jpg = snapshot_jpeg()
        stable_since = None  # reset so we don't spin
        if not jpg:
            continue
        enc = encode_face(jpg)
        if not enc:
            continue
        embedding, snap = enc

        try:
            known = sb_get_faces()
        except Exception as e:
            print(f"[memory] supabase read failed: {e}", flush=True)
            time.sleep(2)
            continue

        m = best_match(embedding, known)
        if m:
            row, dist = m
            fid = row["id"]
            name = row.get("name")
            _set_current(fid, name)
            if now - last_greet.get(fid, 0) > GREET_COOLDOWN:
                last_greet[fid] = now
                try:
                    sb_touch_face(fid, row.get("times_seen", 1))
                except Exception:
                    pass
                if name:
                    print(f"[memory] recognized {name} (d={dist:.2f})", flush=True)
                    say(f"Hey {name}, good to see you.")
                elif fid not in asked_name:
                    # Ask exactly once per run, then stay quiet about it —
                    # the dashboard shows a "who is this?" prompt instead.
                    asked_name.add(fid)
                    print(f"[memory] unnamed {fid} (d={dist:.2f}) — asking once", flush=True)
                    say("I recognize you! I don't know your name yet — "
                        "you can tell me on the dashboard.")
        else:
            try:
                row = sb_insert_face(embedding, snap)
                fid = row.get("id", "?")
                last_greet[fid] = now
                asked_name.add(fid)
                _set_current(fid, None)
                print(f"[memory] NEW face stored: {fid}", flush=True)
                say("Hi there, I don't think we've met. I'll remember your face.")
            except Exception as e:
                print(f"[memory] supabase insert failed: {e}", flush=True)
        time.sleep(1.0)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--name":
        sb_name_face(sys.argv[2], " ".join(sys.argv[3:]))
        print(f"named {sys.argv[2]} -> {' '.join(sys.argv[3:])}")
    else:
        run()
