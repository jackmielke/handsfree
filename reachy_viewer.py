#!/usr/bin/env python3
"""
reachy_viewer.py — "see what the robot sees" from your laptop.

A tiny zero-dependency web dashboard that shows what Wonder is perceiving in
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


def gather() -> dict:
    """One consolidated perception snapshot for the browser to render."""
    face = _get(f"{REACHY_URL}/api/media/tracking/face")
    state = _get(f"{REACHY_URL}/api/state/full?with_doa=true")
    person = _get(f"{MEM_URL}/current", timeout=1.0)
    online = face is not None or state is not None
    ft = (face or {}).get("face_target", {}) if face else {}
    doa = (state or {}).get("doa", {}) if state else {}
    if person and not person.get("fresh"):
        person = None
    return {
        "person": person,
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
<title>Wonder — what the robot sees</title>
<style>
  :root{--bg:#0a0b10;--panel:#14161f;--line:#242838;--txt:#e6e9f2;--dim:#8b90a6;
        --accent:#5ac8fa;--good:#4ade80;--warn:#fbbf24;--bad:#f87171;}
  *{box-sizing:border-box}
  body{margin:0;font:14px/1.4 -apple-system,system-ui,sans-serif;background:var(--bg);
       color:var(--txt);padding:20px;}
  h1{font-size:18px;margin:0 0 2px;letter-spacing:.5px}
  .sub{color:var(--dim);font-size:12px;margin-bottom:18px}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;max-width:840px}
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
  .vc-controls{display:flex;align-items:center;gap:12px;margin-bottom:12px}
  .vc-btn{width:42px;height:42px;border-radius:50%;border:1px solid var(--line);
          background:#0a0b10;font-size:18px;cursor:pointer;transition:all .15s}
  .vc-btn.muted{background:#3b1219;border-color:#7f1d2d}
  .vc-chip{font-size:12px;color:var(--dim);background:#0a0b10;border:1px solid var(--line);
           border-radius:20px;padding:5px 12px}
  .vc-chip.live{color:var(--good);border-color:#1e3a2f}
  .vc-chip.talk{color:var(--accent);border-color:#14324a}
  .vc-vol{display:flex;align-items:center;gap:8px;color:var(--dim);font-size:14px}
  .vc-vol input{width:130px;accent-color:var(--accent)}
  .convo{max-height:260px;min-height:80px;overflow-y:auto;display:flex;flex-direction:column;
         gap:8px;padding:4px 2px;scroll-behavior:smooth}
  .convo-empty{color:#3a3f52;font-size:13px;text-align:center;padding:24px 0}
  .msg{max-width:78%;padding:9px 13px;border-radius:16px;font-size:14px;line-height:1.4}
  .msg.you{align-self:flex-end;background:#14324a;border-bottom-right-radius:4px}
  .msg.wonder{align-self:flex-start;background:#1b1f2e;border-bottom-left-radius:4px}
  .msg .who{display:block;font-size:10px;text-transform:uppercase;letter-spacing:1px;
            color:var(--dim);margin-bottom:2px}
</style></head><body>
<h1>🤖 Wonder — what the robot sees</h1>
<div class=sub id=status>connecting…</div>
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
  <div class=panel>
    <h2>Perception</h2>
    <div class=kv><span>Face detected</span><b id=facedet>—</b></div>
    <div class=kv><span>Person</span><b id=person>—</b></div>
    <div id=namerow style="display:none;margin-top:10px;gap:8px" class=namerow>
      <input id=nameinput placeholder="Who is this? Type their name…"
        style="flex:1;background:#0a0b10;border:1px solid #7c5c1e;border-radius:10px;
               padding:8px 12px;color:var(--txt);font:inherit;outline:none">
      <button id=namebtn
        style="background:var(--warn);color:#1c1503;border:0;border-radius:10px;
               padding:8px 14px;font:inherit;font-weight:700;cursor:pointer">Teach</button>
    </div>
    <div class=kv><span>Face position</span><b class=mono id=facepos>—</b></div>
    <div class=kv><span>Hearing sound at</span><b class=mono id=doa>—</b></div>
    <div class=kv><span>Speech now</span><b id=speech>—</b></div>
  </div>
  <div class="panel full">
    <h2>Voice · talk with Wonder</h2>
    <div class=vc-controls>
      <button id=mutebtn class=vc-btn title="Mute Wonder's ears">🎙️</button>
      <span id=vcstatus class=vc-chip>connecting…</span>
      <span style="flex:1"></span>
      <span class=vc-vol>🔈
        <input id=vol type=range min=0 max=100 value=60>
        <b id=volval class=mono>–</b>
      </span>
    </div>
    <div id=convo class=convo><div class=convo-empty>Say something — the conversation shows up here.</div></div>
    <div style="display:flex;gap:8px;margin-top:10px">
      <input id=saytext placeholder="…or type something for Wonder to say"
        style="flex:1;background:#0a0b10;border:1px solid var(--line);border-radius:10px;
               padding:10px 12px;color:var(--txt);font:inherit;outline:none">
      <button id=saybtn
        style="background:var(--accent);color:#04121c;border:0;border-radius:10px;
               padding:10px 18px;font:inherit;font-weight:700;cursor:pointer">Speak</button>
    </div>
    <div id=saystatus class=sub style="margin:8px 0 0"></div>
  </div>
  <div class=panel>
    <h2>Body · handsfree</h2>
    <div class=kv><span>Head pose (r/p/y)</span><b class=mono id=pose>—</b></div>
    <div class=kv><span>Antennas</span><b class=mono id=ant>—</b></div>
    <div class=kv><span>Voice</span><b id=voice>—</b></div>
    <div class=kv><span>Last command</span><b id=cmd>—</b></div>
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

let personName=null;   // updated by tick(); drawn on the overlay

function drawFOV(f){
  // Transparent overlay on top of the video — only the face marker + a faint
  // crosshair; the live camera shows through everything else.
  const W=c.width,H=c.height;
  g.clearRect(0,0,W,H);
  g.strokeStyle='rgba(255,255,255,.10)';g.lineWidth=1;
  g.beginPath();g.moveTo(W/2,0);g.lineTo(W/2,H);g.moveTo(0,H/2);g.lineTo(W,H/2);g.stroke();
  if(f&&f.detected){
    // x,y are normalized offsets ~[-1,1]; center them into the frame
    const x=W/2+(f.x||0)*W/2, y=H/2+(f.y||0)*H/2;
    const known=!!personName;
    g.strokeStyle=known?'#5ac8fa':'#4ade80';g.lineWidth=3;
    g.beginPath();g.arc(x,y,44,0,7);g.stroke();
    const label=personName||'face';
    g.font='bold 14px system-ui';
    const tw=g.measureText(label).width;
    g.fillStyle='rgba(5,6,10,.75)';
    g.fillRect(x-tw/2-8,y-72,tw+16,24);
    g.fillStyle=known?'#5ac8fa':'#4ade80';
    g.fillText(label,x-tw/2,y-55);
  }
}

async function tick(){
  try{
    const r=await fetch('/perception');const d=await r.json();
    $('status').innerHTML = d.online
      ? '<span class=dot style=background:var(--good)></span>robot online · '+REACHY
      : '<span class=dot style=background:var(--bad)></span>robot offline · '+REACHY;
    const f=d.face||{};
    $('facedet').textContent=f.detected?'yes ✅':'no';
    $('facepos').textContent=f.detected?`x ${(+f.x).toFixed(2)}  y ${(+f.y).toFixed(2)}`:'—';
    // recognized person + name-teaching form
    const per=d.person;
    personName=per&&per.name?per.name:null;
    if(per&&per.name){
      $('person').innerHTML='<span class=pill style="background:#14324a;color:var(--accent)">'+per.name+'</span>';
      $('namerow').style.display='none';
    }else if(per&&per.face_id){
      $('person').innerHTML='<span class=pill style="background:#3a2e10;color:var(--warn)">unknown</span>';
      $('namerow').style.display='flex';
      window._faceId=per.face_id;
    }else{
      $('person').textContent='—';
      $('namerow').style.display='none';
    }
    const ang=d.doa&&d.doa.angle!=null?(d.doa.angle*180/Math.PI).toFixed(0)+'°':'—';
    $('doa').textContent=ang;
    $('speech').innerHTML=d.doa&&d.doa.speech
      ?'<span class=pill style="background:#1e3a2f;color:var(--good)">talking</span>':'quiet';
    const p=d.pose;
    $('pose').textContent=p?`${p.roll.toFixed(2)} ${p.pitch.toFixed(2)} ${p.yaw.toFixed(2)}`:'—';
    const a=d.antennas;
    $('ant').textContent=a?`${a[0].toFixed(2)}  ${a[1].toFixed(2)}`:'—';
    drawFOV(f);
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
let vcMuted=false, lastLen=-1;

$('mutebtn').onclick=async()=>{
  vcMuted=!vcMuted;
  $('mutebtn').classList.toggle('muted',vcMuted);
  $('mutebtn').textContent=vcMuted?'🔇':'🎙️';
  try{await fetch('/mute',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({muted:vcMuted})});}catch(_){}
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
    d.innerHTML='<span class=who>'+(m.who==='you'?'You':'Wonder')+'</span>';
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
    else if(d.speaking){s.textContent='🔊 Wonder is speaking…';s.className='vc-chip talk';}
    else if(d.muted){s.textContent='muted';s.className='vc-chip';}
    else{s.textContent='👂 listening · brain: '+d.mode+' ('+(d.model||'').replace('claude-','')+')';s.className='vc-chip live';}
    if(d.muted!==undefined&&d.muted!==vcMuted){
      vcMuted=d.muted;
      $('mutebtn').classList.toggle('muted',vcMuted);
      $('mutebtn').textContent=vcMuted?'🔇':'🎙️';
    }
    renderConvo(d.transcript||[]);
  }catch(_){}
}
setInterval(chatTick,800);chatTick();

// ---- teach Wonder a name ----
async function teachName(){
  const name=$('nameinput').value.trim();
  if(!name||!window._faceId)return;
  $('namebtn').textContent='…';
  try{
    const r=await fetch('/nameface',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name,face_id:window._faceId})});
    if(r.ok){$('nameinput').value='';$('namerow').style.display='none';}
  }catch(_){}
  $('namebtn').textContent='Teach';
}
$('namebtn').onclick=teachName;
$('nameinput').addEventListener('keydown',e=>{if(e.key==='Enter')teachName();});
async function speak(text){
  text=(text||'').trim(); if(!text)return;
  $('saystatus').textContent='speaking…';
  try{
    const r=await fetch('/say',{method:'POST',headers:{'Content-Type':'application/json'},
                                body:JSON.stringify({text})});
    $('saystatus').textContent=r.ok?'🔊 said: '+text:'error — check viewer logs';
  }catch(e){$('saystatus').textContent='error: '+e;}
}
$('saybtn').onclick=()=>{speak($('saytext').value);$('saytext').value='';};
$('saytext').addEventListener('keydown',e=>{
  if(e.key==='Enter'){speak($('saytext').value);$('saytext').value='';}});

connectHandsfree();
setInterval(tick,250);tick();
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
        if (self.path.startswith("/mute") or self.path.startswith("/volume")
                or self.path.startswith("/nameface")):
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n))
                if self.path.startswith("/mute"):
                    out = _post(f"{CHAT_URL}/mute", {"muted": bool(body.get("muted"))})
                elif self.path.startswith("/nameface"):
                    out = _post(f"{MEM_URL}/name",
                                {"name": body.get("name", ""),
                                 "face_id": body.get("face_id")}, timeout=20.0)
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
    # Make sure face tracking is on so the view has data to show.
    _post(f"{REACHY_URL}/api/media/tracking/enable")
    print(f"[viewer] reachy    = {REACHY_URL}")
    print(f"[viewer] handsfree = {HANDSFREE_URL}")
    print(f"[viewer] open       http://localhost:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
