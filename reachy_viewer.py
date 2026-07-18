#!/usr/bin/env python3
"""
reachy_viewer.py — "see what the robot sees" from your laptop.

A tiny zero-dependency web dashboard that shows what Vibey is perceiving in
real time: detected faces (position in frame), which direction it's hearing
sound from, its live head pose + antenna posture, and — folded in from the
handsfree daemon — whether voice commands are armed and the last one fired.

    python3 reachy_viewer.py      # then open http://localhost:8770

Why not raw camera video? The Reachy daemon's REST API does not expose camera
frames (they stream over WebRTC to on-robot apps only). What it *does* expose is
the derived perception: face target, sound direction-of-arrival, pose. That is
exactly "what the robot notices," which is what this view renders. A raw MJPEG
feed would need a small companion app running on the robot itself — a good
follow-up, but not required for this.

Env overrides:
    REACHY_URL      default http://192.168.1.120:8000
    HANDSFREE_URL   default http://localhost:8765
    PORT            default 8770
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from reachy_voice import say  # ElevenLabs → robot speaker (loads .env)

REACHY_URL = os.environ.get("REACHY_URL", "http://192.168.1.120:8000").rstrip("/")
HANDSFREE_URL = os.environ.get("HANDSFREE_URL", "http://localhost:8765").rstrip("/")
# Live camera MJPEG feed served by reachy_camera.py (runs in the SDK venv).
CAM_URL = os.environ.get("CAM_URL", "http://localhost:8771").rstrip("/")
# Voice-chat control API served by reachy_chat.py.
CHAT_URL = os.environ.get("CHAT_URL", "http://localhost:8772").rstrip("/")
# Face-memory control API served by reachy_memory.py.
MEM_URL = os.environ.get("MEM_URL", "http://localhost:8773").rstrip("/")
PORT = int(os.environ.get("PORT", "8770"))


def _get(url: str, timeout: float = 3.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read() or b"null")
    except Exception:
        return None


def _post(url: str, body: dict | None = None, timeout: float = 3.0):
    try:
        data = json.dumps(body).encode() if body is not None else b""
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"null")
    except Exception:
        return None


# Sleep switch — the dashboard power button. While asleep: chat is muted,
# memory is paused, the robot holds its sleep pose.
ASLEEP = {"on": False}


def _set_power(off: bool) -> None:
    ASLEEP["on"] = off
    _post(f"{CHAT_URL}/mute", {"muted": off})
    _post(f"{MEM_URL}/pause", {"paused": off})
    if off:
        _post(f"{REACHY_URL}/api/media/stop_sound")
        _post(f"{REACHY_URL}/api/move/play/goto_sleep", timeout=20.0)
    else:
        # goto_sleep leaves the motors disabled (that's what makes the sleep
        # pose limp) — they must be re-enabled or wake_up silently does
        # nothing and the robot stays face-down.
        _post(f"{REACHY_URL}/api/motors/set_mode/enabled", timeout=10.0)
        _post(f"{REACHY_URL}/api/move/play/wake_up", timeout=20.0)
        # face-following + speech wobble are core to feeling alive — they can
        # get dropped by daemon restarts, so re-assert on every wake.
        _post(f"{REACHY_URL}/api/media/tracking/enable")
        _post(f"{REACHY_URL}/api/media/wobbling/enable")


def _reboot_robot() -> None:
    """Full daemon restart on the robot — the fix for a stuck backend
    (symptoms: motions/sounds ignored, camera WebRTC won't connect). Takes
    ~20s; motors are re-enabled and the robot woken once it's back."""
    ASLEEP["on"] = False
    _post(f"{REACHY_URL}/api/daemon/restart", timeout=30.0)
    deadline = time.time() + 90
    while time.time() < deadline:
        time.sleep(5)
        st = _get(f"{REACHY_URL}/api/daemon/status", timeout=4.0)
        if st and st.get("state") == "running":
            break
    _post(f"{REACHY_URL}/api/motors/set_mode/enabled", timeout=10.0)
    _post(f"{REACHY_URL}/api/move/play/wake_up", timeout=20.0)
    _post(f"{REACHY_URL}/api/media/tracking/enable")
    _post(f"{REACHY_URL}/api/media/wobbling/enable")
    _post(f"{CHAT_URL}/mute", {"muted": False})
    _post(f"{MEM_URL}/pause", {"paused": False})
    print("[viewer] robot reboot sequence finished", flush=True)


CAPTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
os.makedirs(CAPTURES_DIR, exist_ok=True)
CAPTURING = {"video": False}


def _capture_photo() -> str | None:
    try:
        with urllib.request.urlopen(f"{CAM_URL}/frame.jpg", timeout=8) as r:
            jpeg = r.read()
        name = time.strftime("photo_%Y%m%d_%H%M%S.jpg")
        with open(os.path.join(CAPTURES_DIR, name), "wb") as f:
            f.write(jpeg)
        return name
    except Exception as e:
        print(f"[capture] photo failed: {e}", flush=True)
        return None


def _capture_video(seconds: float = 10.0, fps: int = 8) -> str | None:
    """Pull frames from the camera bridge and assemble an mp4 with ffmpeg."""
    if CAPTURING["video"]:
        return None
    CAPTURING["video"] = True
    try:
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            n = int(seconds * fps)
            interval = 1.0 / fps
            for i in range(n):
                t0 = time.time()
                try:
                    with urllib.request.urlopen(f"{CAM_URL}/frame.jpg", timeout=5) as r:
                        with open(os.path.join(tmp, f"{i:05d}.jpg"), "wb") as f:
                            f.write(r.read())
                except Exception:
                    pass
                time.sleep(max(0, interval - (time.time() - t0)))
            name = time.strftime("clip_%Y%m%d_%H%M%S.mp4")
            out = os.path.join(CAPTURES_DIR, name)
            r = subprocess.run(
                ["ffmpeg", "-y", "-framerate", str(fps),
                 "-pattern_type", "glob", "-i", os.path.join(tmp, "*.jpg"),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                 out], capture_output=True, timeout=120)
            if r.returncode != 0:
                print(f"[capture] ffmpeg: {r.stderr.decode()[-200:]}", flush=True)
                return None
            return name
    except Exception as e:
        print(f"[capture] video failed: {e}", flush=True)
        return None
    finally:
        CAPTURING["video"] = False


# Timelapse: one frame a minute into captures/timelapse_<date>/, assembled
# to mp4 on demand (POST /timelapse {"assemble": true}). Runs continuously —
# cheap (60 JPEGs/hour) and it means mornings come with a film of the night.
def _timelapse_loop():
    while True:
        try:
            day_dir = os.path.join(CAPTURES_DIR,
                                   time.strftime("timelapse_%Y%m%d"))
            os.makedirs(day_dir, exist_ok=True)
            with urllib.request.urlopen(f"{CAM_URL}/frame.jpg", timeout=8) as r:
                jpeg = r.read()
            with open(os.path.join(
                    day_dir, time.strftime("%H%M%S") + ".jpg"), "wb") as f:
                f.write(jpeg)
        except Exception:
            pass
        time.sleep(60)


def _timelapse_assemble(day: str | None = None) -> str | None:
    import subprocess
    day = day or time.strftime("%Y%m%d")
    day_dir = os.path.join(CAPTURES_DIR, f"timelapse_{day}")
    if not os.path.isdir(day_dir) or len(os.listdir(day_dir)) < 5:
        return None
    name = f"timelapse_{day}.mp4"
    out = os.path.join(CAPTURES_DIR, name)
    r = subprocess.run(
        ["ffmpeg", "-y", "-framerate", "12", "-pattern_type", "glob",
         "-i", os.path.join(day_dir, "*.jpg"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         out], capture_output=True, timeout=300)
    return name if r.returncode == 0 else None


def gather() -> dict:
    """One consolidated perception snapshot for the browser to render."""
    face = _get(f"{REACHY_URL}/api/media/tracking/face")
    state = _get(f"{REACHY_URL}/api/state/full?with_doa=true")
    current = _get(f"{MEM_URL}/current", timeout=1.0)
    online = face is not None or state is not None
    ft = (face or {}).get("face_target", {}) if face else {}
    doa = (state or {}).get("doa", {}) if state else {}
    # Everyone memory currently recognizes (0, 1, or several people at once).
    people = [p for p in (current or {}).get("people", []) if p.get("fresh")]
    return {
        "asleep": ASLEEP["on"],
        "people": people,
        "online": online,
        "face": {
            "detected": bool(ft.get("detected")),
            "x": ft.get("x"),          # normalized horizontal offset in frame
            "y": ft.get("y"),          # normalized vertical offset
            "roll": ft.get("roll"),
        },
        "pose": (state or {}).get("head_pose"),
        "antennas": (state or {}).get("antennas_position"),
        "body_yaw": (state or {}).get("body_yaw"),
        "doa": {
            "angle": doa.get("angle"),               # radians
            "speech": bool(doa.get("speech_detected")),
        },
    }


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Vibey — what the robot sees</title>
<style>
  :root{--bg:#0a0b10;--panel:#14161f;--line:#242838;--txt:#e6e9f2;--dim:#8b90a6;
        --accent:#5ac8fa;--good:#4ade80;--warn:#fbbf24;--bad:#f87171;}
  *{box-sizing:border-box}
  body{margin:0;font:14px/1.4 -apple-system,system-ui,sans-serif;background:var(--bg);
       color:var(--txt);padding:20px;}
  h1{font-size:18px;margin:0 0 2px;letter-spacing:.5px}
  .sub{color:var(--dim);font-size:12px;margin-bottom:18px}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;max-width:900px;margin:0 auto}
  @media (max-width:720px){
    .grid{grid-template-columns:1fr}
    .full{grid-column:1}
  }
  .hdr{max-width:900px;margin:0 auto}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px}
  .panel h2{font-size:11px;text-transform:uppercase;letter-spacing:1.2px;color:var(--dim);
            margin:0 0 12px;font-weight:600}
  .full{grid-column:1/3}
  .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;
       vertical-align:middle}
  .kv{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid var(--line)}
  .kv:last-child{border:0}
  .kv span:first-child{color:var(--dim)}
  .mono{font-family:ui-monospace,Menlo,monospace}
  .big{font-size:20px;font-weight:600}
  .fov{position:relative;border-radius:10px;overflow:hidden;background:#05060a;aspect-ratio:16/9}
  .fov img{width:100%;height:100%;object-fit:cover;display:block}
  .fov canvas{position:absolute;inset:0;width:100%;height:100%}
  .fov .noc{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
            color:#3a3f52;font-size:14px;text-align:center;padding:0 20px}
  .pill{display:inline-block;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600}
  .power{width:42px;height:42px;border-radius:50%;border:1px solid var(--line);
         background:var(--panel);color:var(--good);font-size:20px;cursor:pointer;
         transition:all .15s;max-width:840px}
  .power.off{background:#3b1219;border-color:#7f1d2d;color:#f87171}
  /* --- voice controls: two tidy rows of segmented, square icon buttons --- */
  .vc-controls{display:flex;flex-direction:column;gap:10px;margin-bottom:14px}
  .vc-row{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
  .seg{display:inline-flex;background:#0a0b10;border:1px solid var(--line);
       border-radius:11px;padding:3px;gap:2px}
  .icon-btn{width:34px;height:34px;border-radius:8px;border:1px solid transparent;
            background:transparent;font-size:15px;cursor:pointer;line-height:1;
            display:inline-flex;align-items:center;justify-content:center;
            transition:background .12s,border-color .12s;color:var(--txt)}
  .icon-btn:hover{background:#1b1f2e}
  .icon-btn:disabled{opacity:.3;cursor:not-allowed}
  .icon-btn.muted{background:#3b1219;border-color:#7f1d2d}
  .icon-btn.fast-on{background:#3a2e10;border-color:#7c5c1e}
  .icon-btn.vibe-on{background:#1e1233;border-color:#5b21b6}
  .micmeter{display:flex;align-items:center;gap:7px;font-size:11px;color:var(--dim);
            text-transform:uppercase;letter-spacing:.8px}
  .micbar{position:relative;width:72px;height:8px;border-radius:4px;background:#0a0b10;
          border:1px solid var(--line);overflow:hidden;display:inline-block}
  #miclevel{position:absolute;left:0;top:0;bottom:0;width:0%;background:var(--good);
            transition:width .15s}
  #micnotch{position:absolute;top:-1px;bottom:-1px;width:2px;background:var(--warn)}
  .vc-chip{font-size:12px;color:var(--dim);background:#0a0b10;border:1px solid var(--line);
           border-radius:20px;padding:6px 12px;white-space:nowrap}
  .vc-chip.live{color:var(--good);border-color:#1e3a2f}
  .vc-chip.talk{color:var(--accent);border-color:#14324a}
  .vc-vol{display:flex;align-items:center;gap:8px;color:var(--dim);font-size:11px;
          text-transform:uppercase;letter-spacing:.8px}
  .vc-vol input{width:120px;accent-color:var(--accent)}
  .sfxbar{display:grid;grid-template-columns:repeat(auto-fill,minmax(116px,1fr));gap:8px}
  .sfx-btn{height:42px;background:#0a0b10;border:1px solid var(--line);border-radius:10px;
           color:var(--txt);font:inherit;font-size:12px;cursor:pointer;padding:0 10px;
           display:flex;align-items:center;justify-content:space-between;gap:8px;
           transition:border-color .12s,background .12s,transform .12s}
  .sfx-btn:hover{border-color:var(--accent);background:#101521}
  .sfx-btn:active{transform:translateY(1px)}
  .sfx-btn.playing{border-color:#1e3a2f;background:#102018;color:var(--good)}
  .sfx-icon{font-family:ui-monospace,Menlo,monospace;font-size:10px;color:var(--accent);
            border:1px solid #14324a;border-radius:999px;padding:2px 6px;white-space:nowrap}
  .sfx-label{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .convo{max-height:260px;min-height:80px;overflow-y:auto;display:flex;flex-direction:column;
         gap:8px;padding:4px 2px;scroll-behavior:smooth}
  .convo-empty{color:#3a3f52;font-size:13px;text-align:center;padding:24px 0}
  .msg{max-width:78%;padding:9px 13px;border-radius:16px;font-size:14px;line-height:1.4}
  .msg.you{align-self:flex-end;background:#14324a;border-bottom-right-radius:4px}
  .msg.wonder{align-self:flex-start;background:#1b1f2e;border-bottom-left-radius:4px}
  .msg .who{display:block;font-size:10px;text-transform:uppercase;letter-spacing:1px;
            color:var(--dim);margin-bottom:2px}
  .gallery{display:flex;flex-wrap:wrap;gap:14px}
  .gallery-empty{color:#3a3f52;font-size:13px;padding:16px 0}
  .person{width:104px;text-align:center}
  .person .thumb{width:104px;height:104px;border-radius:12px;object-fit:cover;
                 background:#0a0b10;border:2px solid var(--line);display:block}
  .person.named .thumb{border-color:#14324a}
  .person-del{position:absolute;top:-6px;right:-6px;width:22px;height:22px;border-radius:50%;
              background:#3b1219;border:1px solid #7f1d2d;color:#fca5a5;font-size:14px;
              line-height:1;cursor:pointer;display:flex;align-items:center;justify-content:center}
  .person-del:hover{background:#5a1a25}
  .pname-input{width:104px;margin-top:6px;font-size:13px;font-weight:600;color:var(--txt);
               background:transparent;border:1px solid transparent;border-radius:6px;
               text-align:center;padding:2px 4px;font-family:inherit}
  .pname-input::placeholder{color:var(--dim);font-weight:400;font-style:italic}
  .pname-input:hover,.pname-input:focus{border-color:var(--line);background:#0a0b10;outline:none}
  .person .pmeta{font-size:11px;color:var(--dim);margin-top:1px}
  .pstack{display:flex;justify-content:center;align-items:center;margin-top:5px}
  .pstack img{width:28px;height:28px;border-radius:7px;object-fit:cover;
              border:2px solid var(--panel);margin-left:-9px}
  .pstack img:first-child{margin-left:0}
  .pstack-more{font-size:10px;color:var(--dim);margin-left:4px}
  /* --- Vibey's head: the OpenClaw thought stream --- */
  .brainlog{max-height:280px;overflow-y:auto;font-family:ui-monospace,Menlo,monospace;
            font-size:12px;line-height:1.55;display:flex;flex-direction:column;gap:6px;
            scroll-behavior:smooth;background:#05060a;border:1px solid var(--line);
            border-radius:10px;padding:12px}
  .brainlog-empty{color:#3a3f52;text-align:center;padding:18px 0;font-family:inherit}
  .bl{display:flex;gap:8px;align-items:baseline}
  .bl .tag{flex-shrink:0;font-size:10px;text-transform:uppercase;letter-spacing:.8px;
           width:64px;text-align:right}
  .bl.thinking .tag{color:#a78bfa} .bl.thinking .tx{color:#8b90a6;font-style:italic}
  .bl.tool .tag{color:var(--warn)}  .bl.tool .tx{color:#d0d4e2}
  .bl.result .tag{color:#4b5266}   .bl.result .tx{color:#4b5266}
  .bl.say .tag{color:var(--good)}  .bl.say .tx{color:var(--txt)}
  .bl.user .tag{color:var(--accent)} .bl.user .tx{color:var(--accent)}
  .bl .tx{white-space:pre-wrap;word-break:break-word}
  .peoplerows{display:flex;flex-direction:column;gap:8px;margin:8px 0}
  .prow{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
  .prow-known{background:#14324a;color:var(--accent);border-radius:20px;
              padding:4px 12px;font-size:13px;font-weight:600}
  /* --- custom name-picker dropdown --- */
  .dd{position:relative;display:inline-block}
  .dd-btn{background:#0a0b10;border:1px solid var(--line);border-radius:10px;height:34px;
          padding:0 12px;color:var(--txt);font:inherit;font-size:13px;cursor:pointer;
          display:inline-flex;align-items:center;gap:8px;transition:border-color .15s}
  .dd-btn:hover{border-color:var(--accent)}
  .dd-caret{color:var(--dim);font-size:10px}
  .dd-menu{display:none;position:absolute;top:calc(100% + 6px);left:0;z-index:30;
           min-width:190px;max-height:230px;overflow-y:auto;background:var(--panel);
           border:1px solid var(--line);border-radius:12px;padding:5px;
           box-shadow:0 12px 32px rgba(0,0,0,.55)}
  .dd-menu.open{display:block}
  .dd-item{padding:8px 12px;border-radius:8px;font-size:13px;cursor:pointer;
           white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .dd-item:hover{background:#1b1f2e}
  .dd-new{color:var(--accent)}
  .dd-sep{height:1px;background:var(--line);margin:5px 8px}
  .dd-inputwrap{padding:4px}
  .dd-input{width:100%;background:#0a0b10;border:1px solid var(--accent);border-radius:8px;
            color:var(--txt);font:inherit;font-size:13px;padding:7px 10px;outline:none}
  /* --- per-person photo manager modal --- */
  .modal-backdrop{display:none;position:fixed;inset:0;background:rgba(3,4,8,.72);
                  z-index:50;align-items:center;justify-content:center;padding:20px}
  .modal-backdrop.open{display:flex}
  .modal{background:var(--panel);border:1px solid var(--line);border-radius:16px;
         padding:18px;max-width:560px;width:100%;max-height:80vh;overflow-y:auto;
         box-shadow:0 24px 64px rgba(0,0,0,.6)}
  .modal-head{display:flex;align-items:center;gap:10px;margin-bottom:14px}
  .pm-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:12px}
  .pm-cell{position:relative}
  .pm-cell img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:10px;
               border:2px solid var(--line);display:block}
  .pm-del{position:absolute;top:-7px;right:-7px;width:24px;height:24px;border-radius:50%;
          background:#3b1219;border:1px solid #7f1d2d;color:#fca5a5;font-size:13px;
          cursor:pointer;display:flex;align-items:center;justify-content:center;line-height:1}
  .pm-del:hover{background:#5a1a25}
  .pm-del:disabled{opacity:.3;cursor:not-allowed}
  .pm-cell .when{font-size:10px;color:var(--dim);margin-top:3px;text-align:center}
  .cap-btn{background:#0a0b10;border:1px solid var(--line);border-radius:10px;height:36px;
           padding:0 16px;color:var(--txt);font:inherit;font-size:13px;cursor:pointer}
  .cap-btn:hover{border-color:var(--accent)}
  .cap-btn:disabled{opacity:.4}
  .capgrid{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}
  .capgrid a{display:block;position:relative}
  .capgrid img,.capgrid video{width:120px;height:68px;object-fit:cover;border-radius:8px;
             border:1px solid var(--line);display:block;background:#05060a}
  .capgrid .cap-tag{position:absolute;bottom:4px;right:4px;font-size:9px;background:rgba(0,0,0,.7);
             padding:1px 5px;border-radius:4px;color:var(--dim)}
</style></head><body>
<div class=hdr style="display:flex;align-items:center;gap:14px">
  <h1 style="flex:1">🤖 Vibey — what the robot sees</h1>
  <button id=alarmbtn class=power title="Wake-up show: sunrise song + singing + dance" style="color:var(--accent)">🌅</button>
  <button id=rebootbtn class=power title="Reboot the robot (fixes stuck motors/sounds/camera, ~30s)" style="color:var(--warn)">⟳</button>
  <button id=powerbtn class=power title="Put Vibey to sleep / wake it up">⏻</button>
</div>
<div class="sub hdr" id=status>connecting…</div>
<div class=grid>
  <div class="panel full">
    <h2>Field of view · live camera + face</h2>
    <div class=fov>
      <img id=cam alt="" />
      <canvas id=fovc width=800 height=450></canvas>
      <div class=noc id=noc style=display:none>
        camera feed offline<br><small>start it with:
        <code>source reachy_env/bin/activate &amp;&amp; python3 reachy_camera.py</code></small>
      </div>
    </div>
  </div>
  <div class="panel full">
    <h2>Voice · talk with Vibey</h2>
    <div class=vc-controls>
      <div class=vc-row>
        <span class=seg>
          <button id=mutebtn class=icon-btn title="Mute Vibey's ears">🎙️</button>
          <button id=fastbtn class=icon-btn title="Fast mode: ElevenLabs agent, skips Claude">⚡</button>
          <button id=vibebtn class=icon-btn title="Vibe mode: OpenClaw agent — can improve its own code">🎮</button>
          <button id=resaybtn class=icon-btn title="Re-say the last thing Vibey said">🔁</button>
        </span>
        <span id=vcstatus class=vc-chip>connecting…</span>
        <span style="flex:1"></span>
        <span class=micmeter title="Mic level — bar past the notch means Vibey can hear it">
          mic <span class=micbar><span id=miclevel></span><span id=micnotch></span></span>
        </span>
      </div>
      <div class=vc-row>
        <span id=emotes class=seg></span>
        <span style="flex:1"></span>
        <span class=vc-vol>vol
          <input id=vol type=range min=0 max=100 value=60>
          <b id=volval class=mono>–</b>
        </span>
      </div>
    </div>
    <div id=convo class=convo><div class=convo-empty>Say something — the conversation shows up here.</div></div>
    <div style="display:flex;gap:8px;margin-top:10px">
      <input id=saytext placeholder="…or message Vibey here (prefix with say: to speak text verbatim)"
        style="flex:1;background:#0a0b10;border:1px solid var(--line);border-radius:10px;
               padding:10px 12px;color:var(--txt);font:inherit;outline:none">
      <button id=saybtn
        style="background:var(--accent);color:#04121c;border:0;border-radius:10px;
               padding:10px 18px;font:inherit;font-weight:700;cursor:pointer">Send</button>
    </div>
    <div id=saystatus class=sub style="margin:8px 0 0"></div>
  </div>
  <div class="panel full">
    <h2>📸 Capture <span class=sub style="display:inline;margin-left:6px">saved to captures/</span></h2>
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
      <button id=snapbtn class=cap-btn>📷 Photo</button>
      <button id=clipbtn class=cap-btn>🎬 10s clip</button>
      <span id=capstatus class=sub style="margin:0"></span>
    </div>
    <div id=capgrid class=capgrid></div>
  </div>
  <div class=panel>
    <h2>Perception</h2>
    <div class=kv><span>Face detected</span><b id=facedet>—</b></div>
    <div class=kv><span>People in view</span><b id=personcount>—</b></div>
    <div id=peoplerows class=peoplerows></div>
    <div class=kv><span>Hearing sound at</span><b class=mono id=doa>—</b></div>
    <div class=kv><span>Speech now</span><b id=speech>—</b></div>
  </div>
  <div class="panel full">
    <h2>Sound effects <span id=sfxstatus class=sub style="display:inline;margin-left:6px"></span></h2>
    <div id=sfxbar class=sfxbar></div>
  </div>
  <div class="panel full">
    <h2>🧠 Vibey's head <span class=sub style="display:inline;margin-left:6px">OpenClaw agent — thinking, tool calls, code edits</span></h2>
    <div id=brainlog class=brainlog><div class=brainlog-empty>Turn on 🎮 Vibe mode and talk to it — its thought process streams here.</div></div>
  </div>
  <div class=panel>
    <h2>⏰ Alarms <span class=sub style="display:inline;margin-left:6px">wake-up shows</span></h2>
    <div id=alarmlist style="display:flex;flex-direction:column;gap:6px"></div>
    <div style="display:flex;gap:8px;margin-top:10px">
      <input id=alarmtime type=time value="07:00"
        style="background:#0a0b10;border:1px solid var(--line);border-radius:10px;
               padding:8px 10px;color:var(--txt);font:inherit;outline:none">
      <select id=alarmrepeat
        style="background:#0a0b10;border:1px solid var(--line);border-radius:10px;
               padding:8px 10px;color:var(--txt);font:inherit;outline:none">
        <option value=once>once</option><option value=daily>daily</option>
      </select>
      <button id=alarmadd class=cap-btn>＋ Add</button>
    </div>
  </div>
  <div class="panel full">
    <h2>🌐 VibeVerse <span class=sub style="display:inline;margin-left:6px">Vibey's avatar on Edge Island · <a href="https://myvibeverse.com/city?spawn=island" target=_blank style="color:var(--accent)">visit</a></span></h2>
    <div class=kv><span>In lobby with</span><b id=versewho>—</b></div>
    <div id=verselog class=brainlog style="max-height:170px;margin-top:10px"></div>
  </div>
  <div class=panel>
    <h2>Body · handsfree</h2>
    <div class=kv><span>Head pose (r/p/y)</span><b class=mono id=pose>—</b></div>
    <div class=kv><span>Antennas</span><b class=mono id=ant>—</b></div>
    <div class=kv><span>Voice</span><b id=voice>—</b></div>
    <div class=kv><span>Last command</span><b id=cmd>—</b></div>
  </div>
  <div class="panel full">
    <h2>Known faces <span id=peoplecount class=sub style="display:inline;margin-left:6px"></span></h2>
    <div id=gallery class=gallery>
      <div class=gallery-empty>Nobody learned yet — stand in front of Vibey and teach it a name above.</div>
    </div>
    <datalist id=knownNamesList></datalist>
  </div>
</div>
<div id=photomodal class=modal-backdrop>
  <div class=modal>
    <div class=modal-head>
      <b id=pm-title>Photos</b>
      <span id=pm-sub class=sub style="margin:0"></span>
      <span style="flex:1"></span>
      <button id=pm-close class=icon-btn title="Close">✕</button>
    </div>
    <div id=pm-grid class=pm-grid></div>
    <div id=pm-hint class=sub style="margin:10px 0 0"></div>
  </div>
</div>
<script>
const REACHY=%REACHY%, HANDSFREE=%HANDSFREE%, CAM=%CAM%;
const $=id=>document.getElementById(id);
const c=$('fovc'),g=c.getContext('2d');

// Camera MJPEG feed — <img> streams it; on error show the fallback note.
const cam=$('cam');
cam.onerror=()=>{$('noc').style.display='flex';cam.style.opacity=0;};
cam.onload =()=>{$('noc').style.display='none';cam.style.opacity=1;};
cam.src=CAM+'/stream';

// Names Vibey already knows, for the "who is this" dropdown — refreshed
// alongside the gallery so a newly-taught name shows up here too.
let knownNames=[];
async function fetchKnownNames(){
  try{
    knownNames=await(await fetch('/knownnames')).json();
    const dl=$('knownNamesList');
    dl.innerHTML=knownNames.map(n=>`<option value="${n.replace(/"/g,'&quot;')}">`).join('');
  }catch(_){}
}

function drawFOV(people){
  // Transparent overlay on top of the video — a ring + label per person in
  // frame (there can be more than one), plus a faint crosshair.
  const W=c.width,H=c.height;
  g.clearRect(0,0,W,H);
  g.strokeStyle='rgba(255,255,255,.10)';g.lineWidth=1;
  g.beginPath();g.moveTo(W/2,0);g.lineTo(W/2,H);g.moveTo(0,H/2);g.lineTo(W,H/2);g.stroke();
  for(const p of (people||[])){
    // x,y are normalized offsets ~[-1,1]; center them into the frame
    const x=W/2+(p.x||0)*W/2, y=H/2+(p.y||0)*H/2;
    const known=!!p.name;
    g.strokeStyle=known?'#5ac8fa':'#4ade80';g.lineWidth=3;
    g.beginPath();g.arc(x,y,44,0,7);g.stroke();
    const label=p.name||'unknown';
    g.font='bold 14px system-ui';
    const tw=g.measureText(label).width;
    g.fillStyle='rgba(5,6,10,.75)';
    g.fillRect(x-tw/2-8,y-72,tw+16,24);
    g.fillStyle=known?'#5ac8fa':'#4ade80';
    g.fillText(label,x-tw/2,y-55);
  }
}

// Renders one row per currently-visible person: a pill for known names, or a
// dropdown-of-known-names + free-text fallback + Teach button for unknowns.
let peopleRowIds=[];
// One open dropdown at a time; closed on any outside click.
document.addEventListener('click',e=>{
  if(!e.target.closest('.dd'))
    document.querySelectorAll('.dd-menu.open').forEach(m=>m.classList.remove('open'));
});

async function teachFace(faceId,name,after){
  name=(name||'').trim(); if(!name)return;
  try{
    await fetch('/nameface',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name,face_id:faceId})});
  }catch(_){}
  galleryLen=-1; fetchGallery(); fetchKnownNames();
  if(after)after();
}

// Custom name-picker: a pill button opening a floating menu of known names
// plus a "new name" row — replaces the clunky native <select>.
function nameDropdown(faceId){
  const dd=document.createElement('span');
  dd.className='dd';
  const btn=document.createElement('button');
  btn.className='dd-btn';
  btn.innerHTML='Who is this? <span class=dd-caret>▾</span>';
  const menu=document.createElement('div');
  menu.className='dd-menu';
  for(const n of knownNames){
    const it=document.createElement('div');
    it.className='dd-item'; it.textContent=n;
    it.onclick=()=>{menu.classList.remove('open');btn.textContent='Teaching…';teachFace(faceId,n);};
    menu.appendChild(it);
  }
  if(knownNames.length){
    const hr=document.createElement('div'); hr.className='dd-sep'; menu.appendChild(hr);
  }
  const newRow=document.createElement('div');
  newRow.className='dd-item dd-new'; newRow.textContent='＋ New name…';
  newRow.onclick=e=>{
    e.stopPropagation();
    newRow.replaceWith(makeNewNameInput());
  };
  const makeNewNameInput=()=>{
    const wrap=document.createElement('div'); wrap.className='dd-inputwrap';
    const inp=document.createElement('input');
    inp.className='dd-input'; inp.placeholder='type a name, Enter to save';
    inp.onclick=e=>e.stopPropagation();
    inp.addEventListener('keydown',ev=>{
      if(ev.key==='Enter'&&inp.value.trim()){
        menu.classList.remove('open');
        teachFace(faceId,inp.value);
      }
    });
    wrap.appendChild(inp);
    setTimeout(()=>inp.focus(),0);
    return wrap;
  };
  menu.appendChild(newRow);
  btn.onclick=e=>{
    e.stopPropagation();
    document.querySelectorAll('.dd-menu.open').forEach(m=>{if(m!==menu)m.classList.remove('open');});
    menu.classList.toggle('open');
  };
  dd.appendChild(btn); dd.appendChild(menu);
  return dd;
}

function renderPeopleRows(people){
  const ids=people.map(p=>p.face_id+'|'+(p.name||'')).join(',');
  if(ids===peopleRowIds.join(','))return;  // avoid nuking focus every 250ms
  peopleRowIds=people.map(p=>p.face_id+'|'+(p.name||''));
  const box=$('peoplerows');
  box.innerHTML='';
  for(const p of people){
    const row=document.createElement('div');
    row.className='prow';
    if(p.name){
      const pill=document.createElement('span');
      pill.className='prow-known';
      pill.textContent=p.name;
      row.appendChild(pill);
    }else{
      row.appendChild(nameDropdown(p.face_id));
    }
    box.appendChild(row);
  }
}

async function tick(){
  try{
    const r=await fetch('/perception');const d=await r.json();
    $('powerbtn').classList.toggle('off',!!d.asleep);
    $('status').innerHTML = d.asleep
      ? '<span class=dot style=background:var(--warn)></span>Vibey is asleep 😴 · '+REACHY
      : d.online
      ? '<span class=dot style=background:var(--good)></span>robot online · '+REACHY
      : '<span class=dot style=background:var(--bad)></span>robot offline · '+REACHY;
    const people=d.people||[];
    $('facedet').textContent=people.length?'yes ✅':'no';
    $('personcount').textContent=people.length
      ? people.length+' · '+people.map(p=>p.name||'unknown').join(', ')
      : '—';
    renderPeopleRows(people);
    const ang=d.doa&&d.doa.angle!=null?(d.doa.angle*180/Math.PI).toFixed(0)+'°':'—';
    $('doa').textContent=ang;
    $('speech').innerHTML=d.doa&&d.doa.speech
      ?'<span class=pill style="background:#1e3a2f;color:var(--good)">talking</span>':'quiet';
    const p=d.pose;
    $('pose').textContent=p?`${p.roll.toFixed(2)} ${p.pitch.toFixed(2)} ${p.yaw.toFixed(2)}`:'—';
    const a=d.antennas;
    $('ant').textContent=a?`${a[0].toFixed(2)}  ${a[1].toFixed(2)}`:'—';
    drawFOV(people);
  }catch(e){
    $('status').innerHTML='<span class=dot style=background:var(--bad)></span>viewer error';
  }
}
// handsfree voice state via its SSE stream
function connectHandsfree(){
  try{
    const es=new EventSource(HANDSFREE+'/events');
    es.onmessage=ev=>{
      try{const d=JSON.parse(ev.data);
        $('voice').innerHTML = d.v2Armed
          ? '<span class=pill style="background:#14324a;color:var(--accent)">armed 🟢</span>'
          : 'standby';
        $('cmd').textContent = d.voiceLastResult && d.voiceLastResult!=='(no match)'
          ? d.voiceLastResult : (d.voiceLastText||'—');
      }catch(_){}
    };
    es.onerror=()=>{$('voice').textContent='handsfree offline';};
  }catch(e){$('voice').textContent='handsfree offline';}
}
// ---- voice conversation panel ----
let vcMuted=false, vcFast=false, lastLen=-1;

$('mutebtn').onclick=async()=>{
  vcMuted=!vcMuted;
  $('mutebtn').classList.toggle('muted',vcMuted);
  $('mutebtn').textContent=vcMuted?'🔇':'🎙️';
  try{await fetch('/mute',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({muted:vcMuted})});}catch(_){}
};

$('fastbtn').onclick=async()=>{
  vcFast=!vcFast;
  $('fastbtn').classList.toggle('fast-on',vcFast);
  try{await fetch('/fastmode',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({fast:vcFast})});}catch(_){}
};

$('resaybtn').onclick=async()=>{
  try{
    const r=await fetch('/resay',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
    if(!r.ok)$('saystatus').textContent='nothing to re-say yet';
  }catch(_){}
};

let vcVibe=false;
$('vibebtn').onclick=async()=>{
  vcVibe=!vcVibe;
  $('vibebtn').classList.toggle('vibe-on',vcVibe);
  try{await fetch('/vibemode',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({vibe:vcVibe})});}catch(_){}
};

let volTimer=null;
$('vol').oninput=()=>{
  $('volval').textContent=$('vol').value;
  clearTimeout(volTimer);
  volTimer=setTimeout(async()=>{
    try{await fetch('/volume',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({volume:+$('vol').value})});}catch(_){}
  },250);
};
(async()=>{ // initial volume from robot
  try{const d=await(await fetch('/volume')).json();
    if(d&&d.volume!=null){$('vol').value=d.volume;$('volval').textContent=d.volume;}
  }catch(_){}
})();

function renderConvo(items){
  if(items.length===lastLen)return;
  lastLen=items.length;
  const c=$('convo');
  c.innerHTML='';
  if(!items.length){c.innerHTML='<div class=convo-empty>Say something — the conversation shows up here.</div>';return;}
  for(const m of items){
    const d=document.createElement('div');
    d.className='msg '+(m.who==='you'?'you':'wonder');
    d.innerHTML='<span class=who>'+(m.who==='you'?'You':'Vibey')+'</span>';
    d.appendChild(document.createTextNode(m.text));
    c.appendChild(d);
  }
  c.scrollTop=c.scrollHeight;
}

async function chatTick(){
  try{
    const d=await(await fetch('/chatstate')).json();
    const s=$('vcstatus');
    if(d.mode==='offline'){s.textContent='voice chat offline — start reachy_chat.py';s.className='vc-chip';}
    else if(d.speaking){s.textContent='🔊 Vibey is speaking…';s.className='vc-chip talk';}
    else if(d.muted){s.textContent='muted';s.className='vc-chip';}
    else if(d.vibe){s.textContent='👂 listening · 🎮 Vibe (OpenClaw — self-improving)';s.className='vc-chip live';}
    else if(d.fast){s.textContent='👂 listening · ⚡ fast mode (ElevenLabs agent)';s.className='vc-chip live';}
    else{s.textContent='👂 listening · brain: '+d.mode+' ('+(d.model||'').replace('claude-','')+')';s.className='vc-chip live';}
    if(d.muted!==undefined&&d.muted!==vcMuted){
      vcMuted=d.muted;
      $('mutebtn').classList.toggle('muted',vcMuted);
      $('mutebtn').textContent=vcMuted?'🔇':'🎙️';
    }
    if(d.fast!==undefined&&d.fast!==vcFast){
      vcFast=d.fast;
      $('fastbtn').classList.toggle('fast-on',vcFast);
    }
    if(d.vibe!==undefined&&d.vibe!==vcVibe){
      vcVibe=d.vibe;
      $('vibebtn').classList.toggle('vibe-on',vcVibe);
    }
    $('vibebtn').disabled = d.vibe_available===false;
    // mic meter: green fill vs the amber speech-threshold notch
    if(d.mic_level!==undefined){
      const scale=(d.mic_threshold||0.008)*3;   // notch lands at ~1/3 of the bar
      $('miclevel').style.width=Math.min(100,(d.mic_level/scale)*100)+'%';
      $('micnotch').style.left=Math.min(96,((d.mic_threshold||0.008)/scale)*100)+'%';
    }
    $('fastbtn').disabled = d.fast_available===false;
    $('fastbtn').title = d.fast_available===false
      ? 'Fast mode unavailable — ELEVEN_AGENT_ID not configured'
      : 'Fast mode: ElevenLabs agent, skips Claude';
    renderConvo(d.transcript||[]);
  }catch(_){}
}
setInterval(chatTick,800);chatTick();

async function sendText(text){
  text=(text||'').trim(); if(!text)return;
  // "say: something" speaks the text verbatim; anything else is a chat
  // message routed through whichever brain is active (Vibey/fast/OpenClaw).
  const verbatim=text.toLowerCase().startsWith('say:');
  try{
    if(verbatim){
      $('saystatus').textContent='speaking…';
      const r=await fetch('/say',{method:'POST',headers:{'Content-Type':'application/json'},
                                  body:JSON.stringify({text:text.slice(4).trim()})});
      $('saystatus').textContent=r.ok?'🔊 said it':'error — check viewer logs';
    }else{
      $('saystatus').textContent='💬 thinking… (reply appears above and out loud)';
      const r=await fetch('/chatmsg',{method:'POST',headers:{'Content-Type':'application/json'},
                                      body:JSON.stringify({text})});
      if(!r.ok)$('saystatus').textContent='error — is the chat service running?';
      else setTimeout(()=>{if($('saystatus').textContent.startsWith('💬'))$('saystatus').textContent='';},60000);
    }
  }catch(e){$('saystatus').textContent='error: '+e;}
}
$('saybtn').onclick=()=>{sendText($('saytext').value);$('saytext').value='';};
$('saytext').addEventListener('keydown',e=>{
  if(e.key==='Enter'){sendText($('saytext').value);$('saytext').value='';}});

// ---- known-faces gallery ----
let galleryLen=-1;
async function fetchGallery(){
  try{
    const people=await(await fetch('/peoplelist')).json();
    if(!Array.isArray(people))return;
    $('peoplecount').textContent=people.length?('· '+people.length):'';
    if(people.length===galleryLen)return;  // cheap no-op guard
    galleryLen=people.length;
    const g=$('gallery');
    if(!people.length){
      g.innerHTML='<div class=gallery-empty>Nobody learned yet — stand in front of Vibey and teach it a name above.</div>';
      return;
    }
    g.innerHTML='';
    for(const p of people){
      const el=document.createElement('div');
      el.className='person'+(p.name?' named':'');

      const thumbWrap=document.createElement('div');
      thumbWrap.style.cssText='position:relative';
      const img=document.createElement('img');
      img.className='thumb';
      img.src=p.snapshot||'';
      img.alt=p.name||'unnamed';
      img.style.cursor='pointer';
      img.title='See all photos of '+(p.name||'this person');
      img.onclick=()=>openPhotoModal(p);
      const delBtn=document.createElement('button');
      delBtn.textContent='×';
      delBtn.title='Forget '+(p.name||'this person');
      delBtn.className='person-del';
      delBtn.onclick=async()=>{
        if(!confirm(`Forget ${p.name||'this unnamed person'}? This deletes all ${p.sample_count} learned photo(s).`))return;
        delBtn.disabled=true;
        try{
          await fetch('/deleteface',{method:'POST',headers:{'Content-Type':'application/json'},
            body:JSON.stringify({face_id:p.id})});
        }catch(_){}
        galleryLen=-1; fetchGallery();
      };
      thumbWrap.appendChild(img); thumbWrap.appendChild(delBtn);

      const nameInput=document.createElement('input');
      nameInput.className='pname-input';
      nameInput.value=p.name||'';
      nameInput.placeholder='name…';
      nameInput.setAttribute('list','knownNamesList');
      const saveName=async()=>{
        const v=nameInput.value.trim();
        if(!v||v===p.name)return;
        try{
          await fetch('/nameface',{method:'POST',headers:{'Content-Type':'application/json'},
            body:JSON.stringify({name:v,face_id:p.id})});
        }catch(_){}
        galleryLen=-1; fetchGallery();
      };
      nameInput.addEventListener('blur',saveName);
      nameInput.addEventListener('keydown',e=>{if(e.key==='Enter')nameInput.blur();});
      // Select-all on focus so a click-to-edit never silently inserts text
      // mid-name instead of replacing it.
      nameInput.addEventListener('focus',()=>nameInput.select());

      const meta=document.createElement('div');
      meta.className='pmeta';
      meta.textContent=`seen ${p.times_seen}× · ${p.sample_count} photo${p.sample_count===1?'':'s'}`;

      el.appendChild(thumbWrap);
      // photo clump: the person's other learned angles, fanned under the main shot
      const extras=(p.photos||[]).slice(1);
      if(extras.length){
        const stack=document.createElement('div');
        stack.className='pstack';
        stack.style.cursor='pointer';
        stack.onclick=()=>openPhotoModal(p);
        for(const uri of extras){
          const s=document.createElement('img'); s.src=uri; stack.appendChild(s);
        }
        if(p.sample_count>3){
          const more=document.createElement('span');
          more.className='pstack-more'; more.textContent='+'+(p.sample_count-3);
          stack.appendChild(more);
        }
        el.appendChild(stack);
      }
      el.appendChild(nameInput); el.appendChild(meta);
      g.appendChild(el);
    }
  }catch(_){}
}

// ---- capture panel ----
async function refreshCaptures(){
  try{
    const files=await(await fetch('/captures')).json();
    const g=$('capgrid'); g.innerHTML='';
    for(const f of files.slice(0,12)){
      const a=document.createElement('a');
      a.href='/captures/'+f.name; a.target='_blank';
      if(f.name.endsWith('.mp4')){
        const v=document.createElement('video'); v.src='/captures/'+f.name; v.muted=true;
        v.onmouseover=()=>v.play(); v.onmouseout=()=>v.pause();
        a.appendChild(v);
      }else{
        const im=document.createElement('img'); im.src='/captures/'+f.name;
        a.appendChild(im);
      }
      const tag=document.createElement('span');
      tag.className='cap-tag'; tag.textContent=f.name.endsWith('.mp4')?'clip':'photo';
      a.appendChild(tag);
      g.appendChild(a);
    }
  }catch(_){}
}
$('snapbtn').onclick=async()=>{
  $('capstatus').textContent='snapping…';
  try{
    const r=await(await fetch('/capture',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({type:'photo'})})).json();
    $('capstatus').textContent=r.ok?'saved '+r.name:'failed';
  }catch(e){$('capstatus').textContent='error';}
  refreshCaptures();
};
$('clipbtn').onclick=async()=>{
  $('clipbtn').disabled=true;
  $('capstatus').textContent='recording 10s…';
  try{
    const r=await(await fetch('/capture',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({type:'video',seconds:10})})).json();
    $('capstatus').textContent=r.ok?'saved '+r.name:'failed';
  }catch(e){$('capstatus').textContent='error';}
  $('clipbtn').disabled=false;
  refreshCaptures();
};
setInterval(refreshCaptures,30000);refreshCaptures();

// ---- wake-up show ----
$('alarmbtn').onclick=async()=>{
  $('alarmbtn').disabled=true;$('alarmbtn').style.opacity=.4;
  try{await fetch('/alarmnow',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});}catch(_){}
  setTimeout(()=>{$('alarmbtn').disabled=false;$('alarmbtn').style.opacity=1;},45000);
};

// ---- reboot ----
$('rebootbtn').onclick=async()=>{
  if(!confirm('Reboot the robot? Takes ~30 seconds; it will wake up when done.'))return;
  $('rebootbtn').disabled=true;
  $('rebootbtn').style.opacity=.4;
  try{await fetch('/reboot',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});}catch(_){}
  setTimeout(()=>{$('rebootbtn').disabled=false;$('rebootbtn').style.opacity=1;},45000);
};

// ---- power (sleep/wake) ----
let asleep=false;
$('powerbtn').onclick=async()=>{
  asleep=!$('powerbtn').classList.contains('off');
  $('powerbtn').classList.toggle('off',asleep);
  try{await fetch('/power',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({off:asleep})});}catch(_){}
};

// ---- emote buttons ----
const EMOTES={happy:'😊',excited:'⚡',curious:'🤔',sad:'😢',smug:'😏',thinking:'💭',victory:'🏆'};
for(const [name,icon] of Object.entries(EMOTES)){
  const b=document.createElement('button');
  b.className='icon-btn'; b.textContent=icon; b.title=name;
  b.onclick=()=>fetch('/emote',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name})}).catch(()=>{});
  $('emotes').appendChild(b);
}

// ---- soundboard ----
async function loadSfx(){
  try{
    const sounds=await(await fetch('/sfx')).json();
    const box=$('sfxbar');
    box.innerHTML='';
    for(const s of sounds){
      const b=document.createElement('button');
      b.className='sfx-btn';
      b.title=s.label;
      b.innerHTML='<span class=sfx-label></span><span class=sfx-icon></span>';
      b.querySelector('.sfx-label').textContent=s.label;
      b.querySelector('.sfx-icon').textContent=s.icon||'SFX';
      b.onclick=async()=>{
        b.classList.add('playing');
        $('sfxstatus').textContent=s.label;
        try{
          const r=await fetch('/sfx',{method:'POST',headers:{'Content-Type':'application/json'},
            body:JSON.stringify({name:s.name})});
          if(!r.ok)$('sfxstatus').textContent='sound failed';
        }catch(_){$('sfxstatus').textContent='sound failed';}
        setTimeout(()=>b.classList.remove('playing'),450);
      };
      box.appendChild(b);
    }
  }catch(_){$('sfxstatus').textContent='soundboard offline';}
}
loadSfx();

connectHandsfree();
setInterval(tick,250);tick();
setInterval(fetchGallery,5000);fetchGallery();
setInterval(fetchKnownNames,5000);fetchKnownNames();

// ---- per-person photo manager ----
$('pm-close').onclick=()=>$('photomodal').classList.remove('open');
$('photomodal').addEventListener('click',e=>{
  if(e.target.id==='photomodal')$('photomodal').classList.remove('open');
});

async function openPhotoModal(p){
  $('pm-title').textContent=p.name||'Unnamed person';
  $('pm-sub').textContent='seen '+p.times_seen+'×';
  $('pm-hint').textContent='';
  $('pm-grid').innerHTML='<div class=sub>loading…</div>';
  $('photomodal').classList.add('open');
  await renderPhotoModal(p);
}

async function renderPhotoModal(p){
  let samples=[];
  try{samples=await(await fetch('/personsamples?face_id='+p.id)).json();}catch(_){}
  if(!Array.isArray(samples))samples=[];
  const g=$('pm-grid');
  g.innerHTML='';
  for(const s of samples){
    const cell=document.createElement('div');
    cell.className='pm-cell';
    const img=document.createElement('img');
    img.src=s.snapshot||'';
    const del=document.createElement('button');
    del.className='pm-del'; del.textContent='×';
    del.title='Forget this photo';
    del.disabled=samples.length<=1;
    del.onclick=async()=>{
      del.disabled=true;
      try{
        const r=await fetch('/deletesample',{method:'POST',
          headers:{'Content-Type':'application/json'},
          body:JSON.stringify({sample_id:s.id})});
        const out=await r.json();
        if(out&&out.error)$('pm-hint').textContent=out.error;
      }catch(_){}
      await renderPhotoModal(p);
      galleryLen=-1; fetchGallery();
    };
    const when=document.createElement('div');
    when.className='when';
    when.textContent=(s.created_at||'').slice(0,10);
    cell.appendChild(img); cell.appendChild(del);
    g.appendChild(cell);
    cell.appendChild(when);
  }
  $('pm-hint').textContent = samples.length<=1
    ? 'Last photo — deleting it would make this person unrecognizable; use the ⊗ on their card to forget them entirely.'
    : samples.length+' photos — × forgets just that one.';
}

// ---- Vibey's head: OpenClaw thought stream ----
const TAGS={thinking:'think',tool:'tool',result:'result',say:'say',user:'heard',error:'error'};
let brainLen=-1;
async function fetchBrain(){
  try{
    const d=await(await fetch('/vibelog')).json();
    const ev=d.events||[];
    if(ev.length===brainLen)return;
    brainLen=ev.length;
    const b=$('brainlog');
    const stick=b.scrollTop+b.clientHeight>=b.scrollHeight-40;
    b.innerHTML='';
    if(!ev.length){
      b.innerHTML='<div class=brainlog-empty>Turn on 🎮 Vibe mode and talk to it — its thought process streams here.</div>';
      return;
    }
    for(const e of ev){
      const row=document.createElement('div');
      row.className='bl '+e.kind;
      const tag=document.createElement('span');
      tag.className='tag'; tag.textContent=TAGS[e.kind]||e.kind;
      const tx=document.createElement('span');
      tx.className='tx'; tx.textContent=e.text;
      row.appendChild(tag); row.appendChild(tx);
      b.appendChild(row);
    }
    if(stick)b.scrollTop=b.scrollHeight;
  }catch(_){}
}
setInterval(fetchBrain,2500);fetchBrain();

// ---- VibeVerse lobby panel ----
let verseLen=-1;
async function fetchVerse(){
  try{
    const d=await(await fetch('/verselog')).json();
    $('versewho').textContent=(d.agents&&d.agents.length)?d.agents.join(', '):'nobody else right now';
    const ev=d.events||[];
    if(ev.length===verseLen)return;
    verseLen=ev.length;
    const b=$('verselog');
    const stick=b.scrollTop+b.clientHeight>=b.scrollHeight-40;
    b.innerHTML='';
    if(!ev.length){b.innerHTML='<div class=brainlog-empty>lobby is quiet…</div>';return;}
    for(const e of ev.slice(-30)){
      const row=document.createElement('div');
      row.className='bl '+(e.kind==='mention'?'user':e.kind==='say'?'say':'result');
      const tag=document.createElement('span');tag.className='tag';tag.textContent=e.kind;
      const tx=document.createElement('span');tx.className='tx';tx.textContent=e.text;
      row.appendChild(tag);row.appendChild(tx);b.appendChild(row);
    }
    if(stick)b.scrollTop=b.scrollHeight;
  }catch(_){}
}
setInterval(fetchVerse,5000);fetchVerse();

// ---- alarm editor ----
let ALARMS=[];
async function fetchAlarms(){
  try{
    ALARMS=await(await fetch('/alarms')).json();
    const box=$('alarmlist'); box.innerHTML='';
    if(!ALARMS.length){box.innerHTML='<div class=sub style=margin:0>no alarms set</div>';}
    ALARMS.forEach((a,i)=>{
      const row=document.createElement('div');
      row.style.cssText='display:flex;align-items:center;gap:8px';
      row.innerHTML='<b class=mono>'+a.time+'</b><span class=sub style=margin:0>'+(a.repeat||'once')+'</span><span style=flex:1></span>';
      const del=document.createElement('button');
      del.className='person-del'; del.style.position='static'; del.textContent='×';
      del.onclick=async()=>{
        ALARMS.splice(i,1);
        await fetch('/setalarms',{method:'POST',headers:{'Content-Type':'application/json'},
          body:JSON.stringify(ALARMS)});
        fetchAlarms();
      };
      row.appendChild(del); box.appendChild(row);
    });
  }catch(_){}
}
$('alarmadd').onclick=async()=>{
  const t=$('alarmtime').value; if(!t)return;
  ALARMS.push({time:t,repeat:$('alarmrepeat').value,label:'dashboard alarm '+t,song:true});
  await fetch('/setalarms',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(ALARMS)});
  fetchAlarms();
};
setInterval(fetchAlarms,20000);fetchAlarms();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/perception"):
            self._send(json.dumps(gather()).encode(), "application/json")
        elif self.path.startswith("/chatstate"):
            self._send(json.dumps(
                _get(f"{CHAT_URL}/state") or {"mode": "offline"}
            ).encode(), "application/json")
        elif self.path.startswith("/volume"):
            self._send(json.dumps(
                _get(f"{REACHY_URL}/api/volume/current") or {}
            ).encode(), "application/json")
        elif self.path.startswith("/sfx"):
            from reachy_sfx import catalog
            self._send(json.dumps(catalog()).encode(), "application/json")
        elif self.path.startswith("/peoplelist"):
            self._send(json.dumps(
                _get(f"{MEM_URL}/people", timeout=8.0) or []
            ).encode(), "application/json")
        elif self.path.startswith("/knownnames"):
            self._send(json.dumps(
                _get(f"{MEM_URL}/names", timeout=8.0) or []
            ).encode(), "application/json")
        elif self.path == "/alarms":
            try:
                alarms = json.loads(open(os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "alarms.json")).read())
            except Exception:
                alarms = []
            self._send(json.dumps(alarms).encode(), "application/json")
        elif self.path.startswith("/verselog"):
            self._send(json.dumps(
                _get("http://localhost:8774/status", timeout=4.0) or {}
            ).encode(), "application/json")
        elif self.path.startswith("/vibelog"):
            self._send(json.dumps(
                _get(f"{CHAT_URL}/vibelog", timeout=6.0) or {"events": []}
            ).encode(), "application/json")
        elif self.path == "/captures":
            files = sorted(os.listdir(CAPTURES_DIR), reverse=True)[:60]
            out = [{"name": f,
                    "size": os.path.getsize(os.path.join(CAPTURES_DIR, f))}
                   for f in files if not f.startswith(".")]
            self._send(json.dumps(out).encode(), "application/json")
        elif self.path.startswith("/captures/"):
            name = os.path.basename(urllib.parse.unquote(self.path.split("/captures/", 1)[1]))
            path = os.path.join(CAPTURES_DIR, name)
            if not os.path.isfile(path):
                self.send_response(404); self.end_headers(); return
            ctype = "video/mp4" if name.endswith(".mp4") else "image/jpeg"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(os.path.getsize(path)))
            self.end_headers()
            with open(path, "rb") as f:
                self.wfile.write(f.read())
        elif self.path.startswith("/personsamples"):
            qs = self.path.split("?", 1)[-1] if "?" in self.path else ""
            self._send(json.dumps(
                _get(f"{MEM_URL}/samples?{qs}", timeout=10.0) or []
            ).encode(), "application/json")
        elif self.path.startswith("/about"):
            # Who is Vibey? Rendered from its own identity file — the robot
            # maintains IDENTITY.md itself, so this page is self-describing.
            try:
                md = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "IDENTITY.md")).read()
            except Exception:
                md = "# Vibey\n(identity file missing)"
            import html as _html
            body = _html.escape(md)
            page = ("<!doctype html><html><head><meta charset=utf-8>"
                    "<meta name=viewport content='width=device-width,initial-scale=1'>"
                    "<title>About Vibey</title><style>"
                    "body{background:#0a0b10;color:#e6e9f2;font:15px/1.6 "
                    "-apple-system,system-ui,sans-serif;max-width:680px;"
                    "margin:0 auto;padding:40px 20px}"
                    "pre{white-space:pre-wrap;font:inherit}"
                    "a{color:#5ac8fa}</style></head><body>"
                    "<p><a href='/'>← dashboard</a></p>"
                    f"<pre>{body}</pre>"
                    "<p style='color:#8b90a6;font-size:12px'>This page renders "
                    "IDENTITY.md — a file the robot writes itself.</p>"
                    "</body></html>")
            self._send(page.encode(), "text/html; charset=utf-8")
        elif self.path == "/" or self.path.startswith("/index"):
            html = (PAGE
                    .replace("%REACHY%", json.dumps(REACHY_URL))
                    .replace("%HANDSFREE%", json.dumps(HANDSFREE_URL))
                    .replace("%CAM%", json.dumps(CAM_URL)))
            self._send(html.encode(), "text/html; charset=utf-8")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path.startswith("/power"):
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n))
                off = bool(body.get("off"))
                # goto_sleep/wake_up take seconds — run off-thread so the UI
                # gets an immediate response.
                threading.Thread(target=_set_power, args=(off,), daemon=True).start()
                self._send(json.dumps({"ok": True, "off": off}).encode(),
                           "application/json")
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return
        if self.path.startswith("/setalarms"):
            try:
                n = int(self.headers.get("Content-Length", 0))
                alarms = json.loads(self.rfile.read(n))
                assert isinstance(alarms, list)
                for a in alarms:
                    assert isinstance(a.get("time"), str) and ":" in a["time"]
                path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "alarms.json")
                with open(path, "w") as f:
                    json.dump(alarms, f, indent=2)
                self._send(json.dumps({"ok": True, "count": len(alarms)}).encode(),
                           "application/json")
            except Exception as e:
                self.send_response(400); self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return
        if self.path.startswith("/timelapse"):
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n)) if n else {}
                name = _timelapse_assemble(body.get("day"))
                self._send(json.dumps({"ok": bool(name), "name": name}).encode(),
                           "application/json")
            except Exception as e:
                self.send_response(400); self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return
        if self.path.startswith("/capture"):
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n)) if n else {}
                if body.get("type") == "video":
                    secs = min(30, max(2, float(body.get("seconds", 10))))
                    name = _capture_video(secs)
                else:
                    name = _capture_photo()
                self._send(json.dumps({"ok": bool(name), "name": name}).encode(),
                           "application/json")
            except Exception as e:
                self.send_response(400); self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return
        if self.path.startswith("/alarmnow"):
            def _show():
                try:
                    from reachy_alarm import fire
                    fire({"label": "on-demand wake-up show", "song": True})
                except Exception as e:
                    print(f"[viewer] alarm show failed: {e}", flush=True)
            threading.Thread(target=_show, daemon=True).start()
            self._send(json.dumps({"ok": True}).encode(), "application/json")
            return
        if self.path.startswith("/reboot"):
            threading.Thread(target=_reboot_robot, daemon=True).start()
            self._send(json.dumps({"ok": True, "rebooting": True}).encode(),
                       "application/json")
            return
        if self.path.startswith("/emote"):
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n))
                from reachy_emotes import play as play_emote
                ok = play_emote(body.get("name", ""), sound=True)
                self._send(json.dumps({"ok": ok}).encode(), "application/json")
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return
        if self.path.startswith("/sfx"):
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n))
                from reachy_sfx import play as play_sfx
                ok = play_sfx(body.get("name", ""))
                self._send(json.dumps({"ok": ok}).encode(), "application/json")
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return
        if (self.path.startswith("/mute") or self.path.startswith("/volume")
                or self.path.startswith("/nameface") or self.path.startswith("/fastmode")
                or self.path.startswith("/vibemode") or self.path.startswith("/chatmsg")
                or self.path.startswith("/resay") or self.path.startswith("/deletesample")
                or self.path.startswith("/deleteface")):
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n))
                if self.path.startswith("/mute"):
                    out = _post(f"{CHAT_URL}/mute", {"muted": bool(body.get("muted"))})
                elif self.path.startswith("/fastmode"):
                    out = _post(f"{CHAT_URL}/fastmode", {"fast": bool(body.get("fast"))})
                elif self.path.startswith("/vibemode"):
                    out = _post(f"{CHAT_URL}/vibemode", {"vibe": bool(body.get("vibe"))})
                elif self.path.startswith("/chatmsg"):
                    out = _post(f"{CHAT_URL}/message", {"text": body.get("text", "")},
                                timeout=10.0)
                elif self.path.startswith("/resay"):
                    out = _post(f"{CHAT_URL}/resay", {}, timeout=10.0)
                elif self.path.startswith("/nameface"):
                    out = _post(f"{MEM_URL}/name",
                                {"name": body.get("name", ""),
                                 "face_id": body.get("face_id")}, timeout=20.0)
                elif self.path.startswith("/deleteface"):
                    out = _post(f"{MEM_URL}/deleteface",
                                {"face_id": body.get("face_id")}, timeout=10.0)
                elif self.path.startswith("/deletesample"):
                    out = _post(f"{MEM_URL}/deletesample",
                                {"sample_id": body.get("sample_id")}, timeout=10.0)
                else:
                    vol = max(0, min(100, int(body.get("volume", 50))))
                    out = _post(f"{REACHY_URL}/api/volume/set", {"volume": vol})
                self._send(json.dumps(out or {}).encode(), "application/json")
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return
        if not self.path.startswith("/say"):
            self.send_response(404)
            self.end_headers()
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            text = json.loads(self.rfile.read(n)).get("text", "").strip()[:500]
            if not text:
                raise ValueError("empty text")
            # TTS+upload takes a few seconds — do it off-thread so the
            # dashboard's polling never stalls behind a speak request.
            threading.Thread(target=say, args=(text,), daemon=True).start()
            self._send(b'{"ok":true}', "application/json")
        except Exception as e:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())


def main():
    threading.Thread(target=_timelapse_loop, daemon=True).start()
    # Make sure face tracking is on so the view has data to show.
    _post(f"{REACHY_URL}/api/media/tracking/enable")
    _post(f"{REACHY_URL}/api/media/wobbling/enable")
    print(f"[viewer] reachy    = {REACHY_URL}")
    print(f"[viewer] handsfree = {HANDSFREE_URL}")
    print(f"[viewer] open       http://localhost:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
