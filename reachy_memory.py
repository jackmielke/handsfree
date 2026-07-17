#!/usr/bin/env python3
"""
reachy_memory.py — Wonder remembers faces (Supabase-backed).

Polls the camera on a fixed interval, detects *every* face in frame (not just
one), and checks each against everyone Wonder already knows — so two people
in view get recognized simultaneously. Returning people get greeted by name in
Wonder's voice; new people get added to memory.

Earlier versions gated capture behind the robot daemon's own single-target
face tracker (/api/media/tracking/face): wait for it to report "detected",
wait for that to hold steady, then look once. That tracker only follows one
face and drops lock on a pose change — which is why recognition would "lose"
someone who just turned their head. This version ignores it entirely and does
its own full-frame multi-face detection on a steady poll instead.

Each person is modelled as several face-embedding *samples* (table
face_samples), not one reference photo — a single photo is fragile to angle
and lighting, and in practice caused the same person to get re-enrolled as a
"new" face repeatedly (measured: same-person distance ~0.565, just over the
old 0.55 cutoff). Matching is nearest-sample-across-everyone, and every
confident recognition (well inside the match tolerance) automatically banks
a new sample of that person, so the model gets more accurate the more Wonder
sees you — no separate enrollment step needed.

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
    MATCH_TOLERANCE  default 0.58  (lower = stricter match; measured gap
                                    between same-person and different-person
                                    distances sits around 0.57–0.62)
    LEARN_TOLERANCE  default 0.45  (only auto-bank a new sample when this
                                    confident — keeps the model from drifting)
    MAX_SAMPLES      default 8     (oldest sample is dropped past this cap)
    POLL_INTERVAL    default 1.2   (seconds between full-frame face scans)
    MAX_FACES        default 4     (largest N faces considered per frame)

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
MATCH_TOLERANCE = float(os.environ.get("MATCH_TOLERANCE", "0.58"))
LEARN_TOLERANCE = float(os.environ.get("LEARN_TOLERANCE", "0.45"))
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "8"))
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "1.2"))
MAX_FACES = int(os.environ.get("MAX_FACES", "4"))

GREET_COOLDOWN = 300.0    # don't re-greet a named person within this window
LEARN_COOLDOWN = 45.0     # min time between auto-banked samples per person
PERSON_FRESH_S = 6.0      # how long a detection stays "currently visible"
MEM_PORT = int(os.environ.get("MEM_PORT", "8773"))

# Live recognition state, shared with the dashboard via the control API.
# A list because more than one face can be in frame at once — each entry is
# {face_id, name, x, y, ts}, x/y normalized to roughly [-1, 1] around center
# (same convention the robot's own tracker used, so the dashboard overlay
# math didn't need to change).
CURRENT_PEOPLE: list[dict] = []
_current_lock = threading.Lock()


def _set_current_people(people: list[dict]):
    with _current_lock:
        CURRENT_PEOPLE[:] = people


def _current_people_fresh() -> list[dict]:
    now = time.time()
    with _current_lock:
        return [dict(p, fresh=(now - p["ts"]) < PERSON_FRESH_S) for p in CURRENT_PEOPLE]


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
    """Everyone Wonder knows, id/name/times_seen only (no samples)."""
    url = f"{SUPABASE_URL}/rest/v1/faces?select=id,name,times_seen,snapshot"
    req = urllib.request.Request(url, headers=_sb_headers())
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read() or b"[]")


def sb_get_samples() -> list[dict]:
    """Every embedding sample, each tagged with its person's id/name — the
    full "face model" used for matching."""
    url = (f"{SUPABASE_URL}/rest/v1/face_samples"
           "?select=id,face_id,embedding,created_at,faces(name,times_seen)")
    req = urllib.request.Request(url, headers=_sb_headers())
    with urllib.request.urlopen(req, timeout=10) as r:
        rows = json.loads(r.read() or b"[]")
    out = []
    for row in rows:
        person = row.get("faces") or {}
        out.append({
            "sample_id": row["id"],
            "face_id": row["face_id"],
            "embedding": row.get("embedding"),
            "created_at": row.get("created_at"),
            "name": person.get("name"),
            "times_seen": person.get("times_seen", 1),
        })
    return out


def sb_insert_face(snapshot: str) -> dict:
    """Create a new person identity (no embedding — samples live separately)."""
    url = f"{SUPABASE_URL}/rest/v1/faces"
    body = json.dumps({"snapshot": snapshot}).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers=_sb_headers({"Prefer": "return=representation"}))
    with urllib.request.urlopen(req, timeout=10) as r:
        rows = json.loads(r.read() or b"[]")
        return rows[0] if rows else {}


def sb_add_sample(face_id: str, embedding: list[float], snapshot: str) -> None:
    """Bank a new embedding sample for a person, capped at MAX_SAMPLES (drops
    the oldest sample first if already at the cap)."""
    url = (f"{SUPABASE_URL}/rest/v1/face_samples"
           f"?face_id=eq.{face_id}&select=id,created_at&order=created_at.asc")
    req = urllib.request.Request(url, headers=_sb_headers())
    with urllib.request.urlopen(req, timeout=10) as r:
        existing = json.loads(r.read() or b"[]")
    if len(existing) >= MAX_SAMPLES:
        oldest_id = existing[0]["id"]
        del_req = urllib.request.Request(
            f"{SUPABASE_URL}/rest/v1/face_samples?id=eq.{oldest_id}",
            method="DELETE", headers=_sb_headers({"Prefer": "return=minimal"}))
        urllib.request.urlopen(del_req, timeout=10).read()

    body = json.dumps({
        "face_id": face_id, "embedding": embedding, "snapshot": snapshot,
    }).encode()
    ins_req = urllib.request.Request(
        f"{SUPABASE_URL}/rest/v1/face_samples", data=body, method="POST",
        headers=_sb_headers({"Prefer": "return=minimal"}))
    urllib.request.urlopen(ins_req, timeout=10).read()


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


def sb_delete_face(face_id: str) -> None:
    """Forget a person entirely — their face_samples go too (FK cascade)."""
    url = f"{SUPABASE_URL}/rest/v1/faces?id=eq.{face_id}"
    req = urllib.request.Request(url, method="DELETE",
                                 headers=_sb_headers({"Prefer": "return=minimal"}))
    urllib.request.urlopen(req, timeout=10).read()


# --------------------------------------------------------------------------- #
# Perception helpers                                                           #
# --------------------------------------------------------------------------- #
def snapshot_jpeg() -> bytes | None:
    try:
        with urllib.request.urlopen(f"{CAM_URL}/frame.jpg", timeout=5) as r:
            return r.read()
    except Exception:
        return None


def encode_faces(jpeg: bytes) -> list[dict]:
    """Every face in the frame (up to MAX_FACES, largest first): embedding for
    matching, a per-face crop as the snapshot to store, and a normalized
    (x, y) center in roughly [-1, 1] for the dashboard overlay. Lazily imports
    face_recognition so the rest of the module loads without it."""
    try:
        import numpy as np
        import face_recognition
        from PIL import Image
    except ImportError:
        print("[memory] face_recognition/numpy/Pillow missing — "
              "run: uv pip install face_recognition", flush=True)
        return []
    import base64
    img = face_recognition.load_image_file(io.BytesIO(jpeg))
    h, w = img.shape[0], img.shape[1]
    boxes = face_recognition.face_locations(img)
    if not boxes:
        return []
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[1] - b[3]), reverse=True)
    boxes = boxes[:MAX_FACES]
    encodings = face_recognition.face_encodings(img, boxes)

    out = []
    pil_img = Image.fromarray(img)
    for (top, right, bottom, left), enc in zip(boxes, encodings):
        cx, cy = (left + right) / 2, (top + bottom) / 2
        # pad the crop a bit so the stored photo isn't a tight, awkward box
        pad_x, pad_y = (right - left) * 0.4, (bottom - top) * 0.4
        crop = pil_img.crop((
            max(0, int(left - pad_x)), max(0, int(top - pad_y)),
            min(w, int(right + pad_x)), min(h, int(bottom + pad_y)),
        ))
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=85)
        uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        out.append({
            "embedding": enc.tolist(),
            "snapshot": uri,
            "x": (cx / w - 0.5) * 2,
            "y": (cy / h - 0.5) * 2,
        })
    return out


def best_match(embedding: list[float], samples: list[dict]):
    """Nearest sample across everyone Wonder knows, within tolerance.
    Matching against every sample (not one photo per person) is what makes
    this robust to angle/lighting — some sample will happen to be close."""
    try:
        import numpy as np
    except ImportError:
        return None
    if not samples:
        return None
    e = np.array(embedding)
    best, best_d = None, 1e9
    for row in samples:
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
            # List, not a single object — zero, one, or several faces can be
            # in frame at once. Kept the /current name for compatibility.
            self._json({"people": _current_people_fresh()})
        elif self.path.startswith("/names"):
            try:
                faces = sb_get_faces()
                names = sorted({r["name"] for r in faces if r.get("name")})
                self._json(names)
            except Exception as e:
                self._json({"error": str(e)}, 500)
        elif self.path.startswith("/people"):
            try:
                faces = sb_get_faces()
                samples = sb_get_samples()
                counts: dict[str, int] = {}
                for s in samples:
                    counts[s["face_id"]] = counts.get(s["face_id"], 0) + 1
                people = [{
                    "id": r["id"],
                    "name": r.get("name"),
                    "times_seen": r.get("times_seen", 1),
                    "snapshot": r.get("snapshot"),
                    "sample_count": counts.get(r["id"], 0),
                } for r in faces]
                people.sort(key=lambda p: p["times_seen"], reverse=True)
                self._json(people)
            except Exception as e:
                self._json({"error": str(e)}, 500)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path.startswith("/name"):
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n))
                name = (body.get("name") or "").strip()[:60]
                face_id = body.get("face_id")
                if not face_id:
                    with _current_lock:
                        if len(CURRENT_PEOPLE) == 1:
                            face_id = CURRENT_PEOPLE[0]["face_id"]
                if not (name and face_id):
                    raise ValueError("need name (and a face_id, or exactly "
                                      "one face currently in view)")
                sb_name_face(face_id, name)
                live = False
                with _current_lock:
                    for p in CURRENT_PEOPLE:
                        if p["face_id"] == face_id:
                            p["name"] = name
                            live = time.time() - p["ts"] < PERSON_FRESH_S
                print(f"[memory] named {face_id} -> {name}", flush=True)
                # Only greet out loud if this is someone actually standing in
                # front of the camera right now — renaming an old gallery
                # entry from across the room shouldn't make Wonder talk.
                if live:
                    try:
                        from reachy_voice import say as _say
                        _say(f"Nice to meet you, {name}. I'll remember you.")
                    except Exception:
                        pass
                self._json({"ok": True, "face_id": face_id, "name": name})
            except Exception as e:
                self._json({"error": str(e)}, 400)
        elif self.path.startswith("/deleteface"):
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n))
                face_id = body.get("face_id")
                if not face_id:
                    raise ValueError("face_id required")
                sb_delete_face(face_id)
                with _current_lock:
                    CURRENT_PEOPLE[:] = [p for p in CURRENT_PEOPLE
                                         if p["face_id"] != face_id]
                print(f"[memory] deleted {face_id}", flush=True)
                self._json({"ok": True, "face_id": face_id})
            except Exception as e:
                self._json({"error": str(e)}, 400)
        else:
            self._json({"error": "not found"}, 404)


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
    print(f"[memory] watching for faces (multi-face, {POLL_INTERVAL}s poll) "
          f"· supabase={SUPABASE_URL} "
          f"· match<={MATCH_TOLERANCE} learn<={LEARN_TOLERANCE}", flush=True)
    last_greet: dict[str, float] = {}
    last_learn: dict[str, float] = {}
    asked_name: set[str] = set()   # unnamed faces we've already asked — once per run

    while True:
        time.sleep(POLL_INTERVAL)
        jpg = snapshot_jpeg()
        if not jpg:
            continue
        faces = encode_faces(jpg)
        if not faces:
            _set_current_people([])
            continue

        try:
            samples = sb_get_samples()
        except Exception as e:
            print(f"[memory] supabase read failed: {e}", flush=True)
            time.sleep(2)
            continue

        now = time.time()
        seen_now: list[dict] = []
        to_greet: list[tuple[str, str | None]] = []  # (fid, name) — spoken after the loop

        for f in faces:
            embedding, snap, x, y = f["embedding"], f["snapshot"], f["x"], f["y"]
            m = best_match(embedding, samples)

            if m:
                row, dist = m
                fid, name = row["face_id"], row.get("name")
                seen_now.append({"face_id": fid, "name": name, "x": x, "y": y, "ts": now})

                # A confident match auto-banks this sighting as a new sample —
                # the model gets more accurate just from normal use.
                if dist <= LEARN_TOLERANCE and now - last_learn.get(fid, 0) > LEARN_COOLDOWN:
                    last_learn[fid] = now
                    try:
                        sb_add_sample(fid, embedding, snap)
                        print(f"[memory] learned a new sample for "
                              f"{name or fid} (d={dist:.2f})", flush=True)
                    except Exception as e:
                        print(f"[memory] learn failed: {e}", flush=True)

                if now - last_greet.get(fid, 0) > GREET_COOLDOWN:
                    last_greet[fid] = now
                    try:
                        sb_touch_face(fid, row.get("times_seen", 1))
                    except Exception:
                        pass
                    if name:
                        print(f"[memory] recognized {name} (d={dist:.2f})", flush=True)
                        to_greet.append((fid, name))
                    elif fid not in asked_name:
                        asked_name.add(fid)
                        print(f"[memory] unnamed {fid} (d={dist:.2f}) — asking once", flush=True)
                        to_greet.append((fid, None))
            else:
                try:
                    row = sb_insert_face(snap)
                    fid = row.get("id", "?")
                    sb_add_sample(fid, embedding, snap)
                    last_greet[fid] = now
                    asked_name.add(fid)
                    seen_now.append({"face_id": fid, "name": None, "x": x, "y": y, "ts": now})
                    print(f"[memory] NEW face stored: {fid}", flush=True)
                    to_greet.append((fid, "__new__"))
                except Exception as e:
                    print(f"[memory] supabase insert failed: {e}", flush=True)

        _set_current_people(seen_now)

        # Speak after processing every face this cycle, so two people walking
        # up together each get acknowledged instead of only the first.
        for fid, name in to_greet:
            if name == "__new__":
                say("Hi there, I don't think we've met. I'll remember your face.")
            elif name:
                say(f"Hey {name}, good to see you.")
            else:
                say("I recognize you! I don't know your name yet — "
                    "you can tell me on the dashboard.")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--name":
        sb_name_face(sys.argv[2], " ".join(sys.argv[3:]))
        print(f"named {sys.argv[2]} -> {' '.join(sys.argv[3:])}")
    else:
        run()
