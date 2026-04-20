"""handsfree viewer + gesture → music instrument.

Serves:
  /          HTML page with live MJPEG feed + Web Audio synth
  /stream    multipart MJPEG of the camera with landmarks drawn on
  /events    Server-Sent Events stream: motion + smile + clap triggers

Gesture wiring (v1):
  - clap (two hands close together)  → toggle beat on/off
  - body motion (landmark velocity)  → brightens the drum filter
  - smile (MediaPipe blendshape)     → fades in a warm major chord pad
"""
import http.server
import json
import math
import os
import socketserver
import subprocess
import threading
import time
import urllib.request
import webbrowser
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

try:
    from Quartz import (
        CGEventCreateKeyboardEvent,
        CGEventPost,
        CGEventSetFlags,
        kCGEventFlagMaskSecondaryFn,
        kCGHIDEventTap,
    )
    _QUARTZ_OK = True
except Exception as _e:  # pragma: no cover
    print(f"[viewer] Quartz import failed: {_e}", flush=True)
    _QUARTZ_OK = False

from avcamera import Camera

PORT = 8765
BOUNDARY = "handsfree-frame"
HERE = Path(__file__).parent
FACE_MODEL_PATH = HERE / "face_landmarker.task"
HAND_MODEL_PATH = HERE / "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# Hand skeleton: pairs of landmark indices to connect.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)
PALM_IDXS = (0, 5, 9, 13, 17)

FACE_DOT_COLOR_BGR = (183, 231, 110)   # #6ee7b7
HAND_DOT_COLOR_BGR = (236, 180, 112)
HAND_LINE_COLOR_BGR = (200, 140, 80)

MOTION_EMA_ALPHA = 0.25
SMILE_EMA_ALPHA = 0.30

# Head bob: look at nose Y over a short window; fire when the descent peaks.
BOB_WINDOW_FRAMES = 6         # rolling window size (~0.20s @ 30fps)
BOB_DESCENT_THRESHOLD = 0.006 # min drop across 3 frames to count as a bob
BOB_COOLDOWN_S = 0.22

# Blink: both eyelids closed together briefly.
BLINK_ON_THRESHOLD = 0.55
BLINK_OFF_THRESHOLD = 0.25
BLINK_COOLDOWN_S = 0.28

# Hand swipe across the frame — loose enough to fire on a real-feeling swipe.
SWIPE_WINDOW_S = 1.0
SWIPE_MIN_TRAVEL = 0.40
SWIPE_EDGE_START = 0.32
SWIPE_EDGE_END = 0.68
SWIPE_COOLDOWN_S = 1.2

# Prayer hands: hold two palms close → hold Wispr hotkey until released.
PRAYER_CLOSE_THRESHOLD = 0.14          # a bit easier to enter
PRAYER_OPEN_THRESHOLD = 0.22           # clearly apart before releasing
PRAYER_ENTER_HOLD_S = 0.20
PRAYER_MAX_HOLD_S = 30.0               # safety: auto-release after 30s
PRAYER_LOST_HANDS_GRACE_S = 1.2        # tolerate long tracker flicker

# Hands-above-head toggles cursor tracking.
HANDS_UP_Y_MAX = 0.22          # wrist Y (top=0) must be above this
HANDS_UP_HOLD_S = 0.35
HANDS_UP_COOLDOWN_S = 1.2

# Double-clap to boot.
CLAP_CLOSE_THRESHOLD = 0.09
CLAP_FAR_THRESHOLD = 0.24
CLAP_GAP_MIN_S = 0.08
CLAP_GAP_MAX_S = 0.7
CLAP_BOOT_COOLDOWN_S = 2.0

# Head-pose cursor control. Jaw-open toggles tracking so it never hijacks
# the mouse until you explicitly open your mouth.
CURSOR_SENSITIVITY_X = 55.0   # pixels per degree of yaw
CURSOR_SENSITIVITY_Y = 55.0   # pixels per degree of pitch
CURSOR_SMOOTHING = 0.28
CURSOR_DEAD_ZONE_DEG = 1.8
CURSOR_CALIB_S = 1.2

# Wispr hotkey. Fn (key code 63) doesn't reliably press/release via CGEvent,
# so the default is F19 (key code 80) — an unused function key that any app
# can be bound to. Bind Wispr Flow to F19 in its settings.
WISPR_KEYCODE = int(os.environ.get("HANDSFREE_WISPR_KEYCODE", "80"))
WISPR_USE_FN_FLAG = os.environ.get("HANDSFREE_WISPR_FN_FLAG", "0") == "1"

# Theremin gating: hand must be above this normalized Y (0=top) to sound.
HAND_PLAY_THRESHOLD_Y = 0.55


HTML = """<!doctype html>
<html><head>
<meta charset="utf-8">
<title>handsfree — play</title>
<style>
  :root { color-scheme: dark;
    --bg:#0c0c10; --panel:#14141a; --ink:#ececf1; --dim:#7a7a88;
    --accent:#6ee7b7; --warm:#fbbf24; --cool:#70b4ec; }
  * { box-sizing: border-box; }
  body { background: radial-gradient(1200px 800px at 50% -10%, #1b1f2a 0%, var(--bg) 60%); color: var(--ink);
    margin:0; min-height:100vh; font-family: ui-monospace, Menlo, monospace;
    display:flex; flex-direction:column; align-items:center; padding: 28px 24px 40px; gap:18px; }
  h1 { margin:0; font-size:12px; letter-spacing:0.16em; text-transform:uppercase; color: var(--dim); font-weight:500; }
  .stage { display:grid; grid-template-columns: auto 300px; gap:20px; align-items:start;
    max-width: 1200px; width: 100%; }
  img { width: 100%; max-width: 860px; border-radius: 14px; background:#000;
    box-shadow: 0 20px 60px rgba(0,0,0,0.55); }
  .side { display:flex; flex-direction:column; gap:18px; padding:20px;
    background: var(--panel); border-radius:14px; min-height:280px; }
  .status { font-size: 22px; font-weight:600; letter-spacing:0.04em; }
  .status.off  { color: var(--dim); }
  .status.on   { color: var(--accent); }
  .row { display:flex; flex-direction:column; gap:6px; }
  .row .label { font-size:10px; letter-spacing:0.14em; text-transform:uppercase; color: var(--dim); }
  .bar { height:10px; background:#0a0a0f; border-radius:6px; overflow:hidden; position:relative; }
  .bar .fill { height:100%; width:0%; border-radius:6px; transition: width 80ms linear; }
  .bar .fill.motion { background: linear-gradient(90deg, var(--cool), var(--accent)); }
  .bar .fill.smile  { background: linear-gradient(90deg, var(--warm), #f59e0b); }
  button { appearance:none; border:none; background: var(--accent); color:#05170f; font-weight:700;
    font-size:13px; letter-spacing:0.08em; text-transform:uppercase; padding:12px 16px;
    border-radius:10px; cursor:pointer; font-family: inherit; }
  button:disabled { background:#2a2a33; color: var(--dim); cursor: default; }
  .hint { color: var(--dim); font-size: 12px; max-width: 860px; text-align:center; line-height:1.6; }
  .clap-flash { position: fixed; inset: 0; pointer-events:none; background: radial-gradient(circle at center, rgba(110,231,183,0.18), transparent 60%);
    opacity:0; transition: opacity 200ms ease-out; }
  .clap-flash.on { opacity: 1; }
  #heard .final   { color: var(--accent); }
  #heard .interim { color: var(--dim); font-style: italic; }
  #jam-btn, #cc-btn { cursor:pointer; font-size:10px; letter-spacing:0.16em;
    text-transform:uppercase; font-weight:700; padding:6px 14px;
    border-radius:999px; border:1px solid #2a2a38; background:#15151c;
    color:var(--dim); }
  #jam-btn.on { background:#b48cff; color:#140a24; border-color:#b48cff; }
  #cc-btn.on  { background:#6ee7b7; color:#05170f; border-color:#6ee7b7; }
  #cc-panel { display:none; margin-top:10px; padding:14px;
    background:#0a0a0f; border:1px solid #262634; border-radius:10px; }
  #cc-panel.open { display:block; }
  .cc-row { display:flex; align-items:center; gap:10px;
    margin-bottom:10px; flex-wrap:wrap; }
  .cc-row:last-child { margin-bottom:0; }
  .cc-label { font-size:10px; letter-spacing:0.16em; text-transform:uppercase;
    color:var(--dim); width:72px; }
  .cc-opts { display:flex; gap:6px; flex-wrap:wrap; }
  .cc-opt { cursor:pointer; font-size:11px; padding:5px 12px;
    border-radius:999px; border:1px solid #2a2a38; background:#15151c;
    color:var(--ink); }
  .cc-opt.on { background:#6ee7b7; color:#05170f; border-color:#6ee7b7; }
  .cc-hint { font-size:10px; color:var(--dim); margin-top:4px; line-height:1.5; }
  body.jam #listening-pill, body.jam #cursor-pill, body.jam #voice-pill,
  body.jam #heard, body.jam .row:has(#notes), body.jam .hint,
  body.jam #status { display:none !important; }
  body.jam { background:radial-gradient(ellipse at top,#1a0f2a 0%,#07070b 70%); }
  #stars { position:fixed; inset:0; width:100%; height:100%;
    pointer-events:none; z-index:0; }
  body > *:not(#stars) { position:relative; z-index:1; }
  details.sec { background:#0f0f16; border:1px solid #23232f;
    border-radius:10px; padding:0; }
  details.sec > summary { list-style:none; cursor:pointer;
    padding:10px 14px; font-size:10px; letter-spacing:0.16em;
    text-transform:uppercase; color:var(--dim); display:flex;
    align-items:center; gap:8px; user-select:none; }
  details.sec > summary::-webkit-details-marker { display:none; }
  details.sec > summary::before { content:'▸'; transition:transform 120ms;
    color:var(--dim); font-size:9px; }
  details.sec[open] > summary::before { transform:rotate(90deg); }
  details.sec[open] > summary { color:var(--ink); }
  details.sec > .sec-body { padding:0 14px 14px; display:flex;
    flex-direction:column; gap:12px; }
  @media (max-width: 900px) { .stage { grid-template-columns: 1fr; } .side { min-height:auto; } }
</style>
</head>
<body>
  <canvas id="stars"></canvas>
  <div style="display:flex; gap:14px; align-items:center;">
    <h1>handsfree</h1>
    <div style="font-size:10px; color:var(--dim); letter-spacing:0.12em;">
      swipe L↔R → switch desktop • prayer hands → wispr
    </div>
    <div style="flex:1"></div>
    <button id="cc-btn"  type="button">control center</button>
    <button id="jam-btn" type="button">jam mode</button>
  </div>
  <div id="cc-panel">
    <div class="cc-row">
      <div class="cc-label">cursor</div>
      <div class="cc-opts">
        <button class="cc-opt" id="cc-cursor-on">turn on</button>
        <button class="cc-opt" id="cc-cursor-off">turn off</button>
        <span id="cc-cursor-state" style="font-size:10px; color:var(--dim);
              letter-spacing:0.14em; text-transform:uppercase;">off</span>
      </div>
    </div>
    <div class="cc-row">
      <div class="cc-label">pointing</div>
      <div class="cc-opts" id="cc-point-opts">
        <button class="cc-opt" data-v="finger">finger</button>
        <button class="cc-opt" data-v="gaze">gaze</button>
        <button class="cc-opt" data-v="head">head</button>
      </div>
    </div>
    <div class="cc-row">
      <div class="cc-label">click</div>
      <div class="cc-opts" id="cc-click-opts">
        <button class="cc-opt" data-v="brow">brow raise</button>
        <button class="cc-opt" data-v="blink">blink</button>
        <button class="cc-opt" data-v="right_wink">right wink</button>
        <button class="cc-opt" data-v="wink">either wink</button>
        <button class="cc-opt" data-v="mouth">mouth open</button>
        <button class="cc-opt" data-v="pinch">pinch</button>
      </div>
    </div>
    <div class="cc-row">
      <div class="cc-label">wispr</div>
      <div class="cc-opts" id="cc-wispr-opts">
        <button class="cc-opt" data-v="all">test all</button>
        <button class="cc-opt" data-v="cgevent_f19">cgevent f19</button>
        <button class="cc-opt" data-v="cgevent_fn">cgevent fn</button>
        <button class="cc-opt" data-v="applescript_fn">applescript fn</button>
        <button class="cc-opt" data-v="apple_dictation">apple dictation</button>
        <button class="cc-opt" data-v="off">off</button>
        <button class="cc-opt" id="cc-wispr-fire">fire now</button>
      </div>
    </div>
    <div class="cc-row">
      <div class="cc-label">right click</div>
      <div class="cc-opts" id="cc-rclick-opts">
        <button class="cc-opt" data-v="off">off</button>
        <button class="cc-opt" data-v="smile">smile (hard)</button>
        <button class="cc-opt" data-v="pucker">pucker</button>
        <button class="cc-opt" data-v="furrow">furrow brow</button>
      </div>
    </div>
    <div class="cc-row">
      <div class="cc-label">double click</div>
      <div class="cc-opts" id="cc-dclick-opts">
        <button class="cc-opt" data-v="off">off</button>
        <button class="cc-opt" data-v="on">double-tap primary</button>
      </div>
    </div>
    <div class="cc-row">
      <div class="cc-label">scroll</div>
      <div class="cc-opts" id="cc-scroll-opts">
        <button class="cc-opt" data-v="gentle">gentle</button>
        <button class="cc-opt" data-v="normal">normal</button>
        <button class="cc-opt" data-v="zippy">zippy</button>
      </div>
    </div>
    <div class="cc-hint">
      hands overhead → cursor on/off (re-calibrates each time).<br>
      head: hold still 1.2s facing center. finger: point right index at
      center. gaze: look straight at camera.<br>
      clicks: brow raise, deliberate blink, right-eye wink, either-eye wink,
      open mouth, or pinch — all edge-triggered.<br>
      <b>scroll:</b> raise both hands and move them up/down together. one
      hand = cursor, two hands = scroll.<br>
      <b>dictation:</b> <i>wispr</i> uses Wispr Flow hotkeys; <i>apple
      dictation</i> double-taps Control (set this in System Settings →
      Keyboard → Dictation → Shortcut = "Press Control key twice").<br>
      "test all" fires cgevent-F19, cgevent-Fn, and applescript-Fn
      in sequence to help you discover which one wakes Wispr.
    </div>
  </div>
  <div class="stage">
    <img src="/stream" alt="camera">
    <div class="side">
      <div id="status" class="status off">click page → allow mic → say "open arc"</div>
      <div style="display:flex; gap:8px; flex-wrap:wrap;">
        <div id="listening-pill" style="display:none;
             font-size:10px; letter-spacing:0.16em; text-transform:uppercase;
             color:#05170f; background:var(--warm); padding:4px 10px;
             border-radius:999px; font-weight:700;">listening…</div>
        <div id="cursor-pill" style="display:none;
             font-size:10px; letter-spacing:0.16em; text-transform:uppercase;
             color:#05170f; background:var(--cool); padding:4px 10px;
             border-radius:999px; font-weight:700;">cursor</div>
        <div id="voice-pill"
             style="font-size:10px; letter-spacing:0.16em; text-transform:uppercase;
             color:var(--dim); background:#1a1a24; padding:4px 10px;
             border-radius:999px; font-weight:700;">voice: off</div>
      </div>

      <details class="sec">
        <summary>voice & heard</summary>
        <div class="sec-body">
          <div id="heard" style="font-size:14px; line-height:1.5; min-height:64px;
               max-height:120px; overflow:hidden; padding:10px 12px;
               background:#0a0a0f; border:1px solid #262634; border-radius:10px;
               color:var(--ink);">
            <span class="interim" style="color:var(--dim); font-style:italic;">
              say something…
            </span>
          </div>
        </div>
      </details>

      <details class="sec">
        <summary>activity</summary>
        <div class="sec-body">
          <div class="row"><div class="label">smile</div><div class="bar"><div id="smile-fill" class="fill smile"></div></div></div>
          <div class="row"><div class="label">left hand</div><div class="bar"><div id="left-fill" class="fill motion"></div></div></div>
          <div class="row"><div class="label">right hand</div><div class="bar"><div id="right-fill" class="fill motion"></div></div></div>
        </div>
      </details>

      <details class="sec">
        <summary>dictation notes</summary>
        <div class="sec-body">
          <textarea id="notes" placeholder="prayer hands → dictated text lands here"
            style="min-height:120px; resize:vertical; background:#0a0a0f;
                   color:var(--ink); border:1px solid #262634; border-radius:10px;
                   padding:10px; font-family:inherit; font-size:13px;
                   line-height:1.5;"></textarea>
        </div>
      </details>

      <details class="sec">
        <summary>cheat sheet</summary>
        <div class="sec-body">
          <div style="color:var(--dim); font-size:11px; line-height:1.7;">
            one hand → cursor<br>
            two hands → scroll<br>
            prayer (hold) → dictate<br>
            double clap → voice on/off<br>
            swipe → next/prev desktop<br>
            bob → drum · smile → pad
          </div>
        </div>
      </details>
    </div>
  </div>
  <div id="bob-flash" class="clap-flash"></div>

<script>
(function() {
  const statusEl  = document.getElementById('status');
  const smileFill = document.getElementById('smile-fill');
  const leftFill  = document.getElementById('left-fill');
  const rightFill = document.getElementById('right-fill');
  const flashEl   = document.getElementById('bob-flash');
  const listenPill= document.getElementById('listening-pill');
  const cursorPill= document.getElementById('cursor-pill');
  const voicePill = document.getElementById('voice-pill');
  const heardEl   = document.getElementById('heard');
  const notesEl   = document.getElementById('notes');

  // ---- Voice commands (Web Speech API) ----------------------------------
  // Freeform: "open <anything>" / "close <anything>". The backend hands the
  // raw app name to `open -a`, which fuzzy-matches installed apps.
  const OPEN_RE  = /\b(?:open|launch)\s+([a-z0-9][a-z0-9 \-\.]{0,30})/i;
  const CLOSE_RE = /\b(?:close|quit)\s+([a-z0-9][a-z0-9 \-\.]{0,30})/i;
  const INSTANT = [
    [/\bscroll\s+down\b/i,       { action: 'scroll', target: 'down' }],
    [/\bscroll\s+up\b/i,         { action: 'scroll', target: 'up' }],
    [/\bpage\s+down\b/i,         { action: 'scroll', target: 'down' }],
    [/\bpage\s+up\b/i,           { action: 'scroll', target: 'up' }],
    [/\b(?:click|tap)\b/i,       { action: 'click',  target: '' }],
    [/\bvolume\s+up\b/i,         { action: 'volume', target: 'up' }],
    [/\bvolume\s+down\b/i,       { action: 'volume', target: 'down' }],
    [/\b(?:mute|silence)\b/i,    { action: 'volume', target: 'mute' }],
    [/\bnext\s+desktop\b/i,      { action: 'desktop', target: 'next' }],
    [/\bprevious\s+desktop\b/i,  { action: 'desktop', target: 'prev' }],
  ];

  function cleanTarget(s) {
    return (s || '').trim()
      .replace(/^(the|up)\s+/i, '')
      .replace(/\s+(please|now|app|application)$/i, '')
      .trim();
  }

  function matchCommand(text) {
    const t = (text || '').toLowerCase().trim();
    if (!t) return null;
    for (const [re, cmd] of INSTANT) {
      if (re.test(t)) return cmd;
    }
    let m = t.match(OPEN_RE);
    if (m) return { action: 'open', target: cleanTarget(m[1]) };
    m = t.match(CLOSE_RE);
    if (m) return { action: 'close', target: cleanTarget(m[1]) };
    return null;
  }

  function sendCommand(cmd) {
    fetch('/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cmd),
    }).catch(() => {});
  }

  function setVoicePill(state, text) {
    voicePill.textContent = 'voice: ' + text;
    if (state === 'on')   { voicePill.style.background = '#6ee7b7'; voicePill.style.color = '#05170f'; }
    else if (state === 'err') { voicePill.style.background = '#f87171'; voicePill.style.color = '#1a0707'; }
    else                      { voicePill.style.background = '#1a1a24'; voicePill.style.color = 'var(--dim)'; }
  }

  // Small ack sound (two sines) so you know listening flipped on/off.
  function ackTone(up) {
    if (!ctx) return;
    const t = ctx.currentTime;
    const freqs = up ? [659.25, 987.77] : [987.77, 659.25];
    freqs.forEach((f, i) => {
      const o = ctx.createOscillator();
      o.type = 'sine'; o.frequency.value = f;
      const g = ctx.createGain();
      const s = t + i * 0.08;
      g.gain.setValueAtTime(0.0001, s);
      g.gain.exponentialRampToValueAtTime(0.18, s + 0.01);
      g.gain.exponentialRampToValueAtTime(0.001, s + 0.25);
      o.connect(g).connect(ctx.destination);
      o.start(s); o.stop(s + 0.3);
    });
  }

  let rec = null;
  let voiceWanted = false;  // user intent: should we be listening?
  let lastFired = 0;

  async function ensureMic() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach(t => t.stop());  // release — SR will reopen
      return true;
    } catch (e) {
      console.warn('mic denied', e);
      setVoicePill('err', 'mic denied');
      return false;
    }
  }

  function createRecognizer() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { setVoicePill('err', 'unsupported'); return null; }
    const r = new SR();
    r.continuous = true;
    r.interimResults = true;
    r.lang = 'en-US';
    r.onstart = () => setVoicePill('on', 'listening');
    r.onerror = (e) => {
      // Silent retry for transient errors; only show real auth blockers.
      if (e.error === 'not-allowed' || e.error === 'service-not-allowed') {
        voiceWanted = false;
        setVoicePill('err', 'mic blocked');
      }
      // network / aborted / no-speech → let onend restart. Do not flicker pill.
    };
    r.onend = () => {
      if (voiceWanted) {
        setTimeout(() => { try { r.start(); } catch {} }, 500);
      } else {
        setVoicePill('off', 'off');
      }
    };
    r.onresult = (ev) => {
      let interim = '', finalText = '';
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const res = ev.results[i];
        if (res.isFinal) finalText += res[0].transcript + ' ';
        else interim += res[0].transcript + ' ';
      }
      const text = (finalText + interim).trim();
      if (text) {
        heardEl.innerHTML = finalText
          ? `<span class="final">${escapeHtml(finalText)}</span>`
            + `<span class="interim">${escapeHtml(interim)}</span>`
          : `<span class="interim">${escapeHtml(interim)}</span>`;
      }
      const cmd = matchCommand((finalText || interim).trim());
      const now = Date.now();
      if (cmd && cmd.target !== undefined && now - lastFired > 1500) {
        lastFired = now;
        flashBob();
        sendCommand(cmd);
      }
    };
    return r;
  }

  function escapeHtml(s) {
    return (s || '').replace(/[&<>\"']/g, c => ({
      '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
    }[c]));
  }

  async function startVoice() {
    if (voiceWanted) return;
    const ok = await ensureMic();
    if (!ok) return;
    if (!rec) rec = createRecognizer();
    if (!rec) return;
    voiceWanted = true;
    setVoicePill('on', 'starting…');
    try { rec.start(); } catch {}
    ackTone(true);
  }

  function stopVoice() {
    if (!voiceWanted) return;
    voiceWanted = false;
    if (rec) { try { rec.stop(); } catch {} }
    setVoicePill('off', 'off');
    ackTone(false);
  }

  function toggleVoice() {
    if (voiceWanted) stopVoice(); else startVoice();
  }

  // Clickable voice pill doubles as on/off button.
  voicePill.style.cursor = 'pointer';
  voicePill.addEventListener('click', toggleVoice);

  // Jam Mode button: hides voice UI, disables swipe, routes hands to synths.
  const jamBtn = document.getElementById('jam-btn');
  function setJamMode(on) {
    jamMode = on;
    document.body.classList.toggle('jam', on);
    jamBtn.classList.toggle('on', on);
    jamBtn.textContent = on ? 'jam mode · on' : 'jam mode';
    if (on && voiceWanted) stopVoice();
    // Immediately silence whichever synth bank is now idle.
    if (on) {
      if (leftTh) leftTh.setHandY(null);
      if (rightTh) rightTh.setHandY(null);
    } else {
      if (jamBass) jamBass.setXY(null, null);
      if (jamLead) jamLead.setXY(null, null);
    }
    fetch('/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'jam', on }),
    }).catch(() => {});
  }
  jamBtn.addEventListener('click', () => {
    unlockAudio();
    setJamMode(!jamMode);
  });

  // Control Center: pointing/click method picker.
  const ccBtn   = document.getElementById('cc-btn');
  const ccPanel = document.getElementById('cc-panel');
  const ccPoint = document.getElementById('cc-point-opts');
  const ccClick = document.getElementById('cc-click-opts');
  let ccOpen = false;
  function setCCOpen(on) {
    ccOpen = on;
    ccPanel.classList.toggle('open', on);
    ccBtn.classList.toggle('on', on);
  }
  ccBtn.addEventListener('click', () => setCCOpen(!ccOpen));

  function paintActive(container, value) {
    container.querySelectorAll('.cc-opt').forEach(b => {
      b.classList.toggle('on', b.dataset.v === value);
    });
  }
  function postMethod(payload) {
    fetch('/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(Object.assign({ action: 'cursor_method' }, payload)),
    }).catch(() => {});
  }
  const ccCursorOn   = document.getElementById('cc-cursor-on');
  const ccCursorOff  = document.getElementById('cc-cursor-off');
  const ccCursorState= document.getElementById('cc-cursor-state');
  function postCursor(on) {
    fetch('/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'cursor_enable', on }),
    }).catch(() => {});
  }
  ccCursorOn.addEventListener('click',  () => postCursor(true));
  ccCursorOff.addEventListener('click', () => postCursor(false));

  ccPoint.addEventListener('click', (e) => {
    const b = e.target.closest('.cc-opt'); if (!b) return;
    paintActive(ccPoint, b.dataset.v);
    postMethod({ pointing: b.dataset.v });
  });
  ccClick.addEventListener('click', (e) => {
    const b = e.target.closest('.cc-opt'); if (!b) return;
    paintActive(ccClick, b.dataset.v);
    postMethod({ click: b.dataset.v });
  });

  const ccWispr = document.getElementById('cc-wispr-opts');
  function postWispr(method) {
    fetch('/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'wispr_method', method }),
    }).catch(() => {});
  }
  function fireWispr() {
    fetch('/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'wispr_test' }),
    }).catch(() => {});
  }
  ccWispr.addEventListener('click', (e) => {
    const b = e.target.closest('.cc-opt'); if (!b) return;
    if (b.id === 'cc-wispr-fire') { fireWispr(); return; }
    ccWispr.querySelectorAll('.cc-opt').forEach(x => {
      if (x.id !== 'cc-wispr-fire')
        x.classList.toggle('on', x.dataset.v === b.dataset.v);
    });
    postWispr(b.dataset.v);
  });

  const ccRClick = document.getElementById('cc-rclick-opts');
  ccRClick.addEventListener('click', (e) => {
    const b = e.target.closest('.cc-opt'); if (!b) return;
    paintActive(ccRClick, b.dataset.v);
    fetch('/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'right_click_method', method: b.dataset.v }),
    }).catch(() => {});
  });

  const ccDClick = document.getElementById('cc-dclick-opts');
  ccDClick.addEventListener('click', (e) => {
    const b = e.target.closest('.cc-opt'); if (!b) return;
    paintActive(ccDClick, b.dataset.v);
    fetch('/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'double_click', on: b.dataset.v === 'on' }),
    }).catch(() => {});
  });

  const ccScroll = document.getElementById('cc-scroll-opts');
  function postScroll(sens) {
    fetch('/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'scroll_sens', sens }),
    }).catch(() => {});
  }
  ccScroll.addEventListener('click', (e) => {
    const b = e.target.closest('.cc-opt'); if (!b) return;
    paintActive(ccScroll, b.dataset.v);
    postScroll(b.dataset.v);
  });

  let ctx = null, drums = null, pad = null;
  let leftTh = null, rightTh = null;
  let jamBass = null, jamLead = null;
  let jamMode = false;
  let audioReady = false;

  function noiseBuffer(ctx, seconds = 2) {
    const buf = ctx.createBuffer(1, seconds * ctx.sampleRate, ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    return buf;
  }

  // BobDrums: one kick/snare (alternating) + hat on each call to hit().
  // No scheduler — the rhythm *is* your head bobbing.
  class BobDrums {
    constructor(ctx) {
      this.ctx = ctx;
      this.count = 0;

      this.filter = ctx.createBiquadFilter();
      this.filter.type = 'lowpass';
      this.filter.frequency.value = 2800;
      this.filter.Q.value = 0.4;

      this.master = ctx.createGain();
      this.master.gain.value = 0.8;
      this.filter.connect(this.master).connect(ctx.destination);

      this.noiseBuf = noiseBuffer(ctx, 1);
    }

    hit() {
      const t = this.ctx.currentTime;
      this.count++;
      if (this.count % 2 === 1) this._kick(t);
      else this._snare(t);
      this._hat(t, 0.4);
    }

    _kick(t) {
      const osc = this.ctx.createOscillator();
      const env = this.ctx.createGain();
      osc.frequency.setValueAtTime(140, t);
      osc.frequency.exponentialRampToValueAtTime(40, t + 0.12);
      env.gain.setValueAtTime(0.0001, t);
      env.gain.exponentialRampToValueAtTime(0.9, t + 0.003);
      env.gain.exponentialRampToValueAtTime(0.001, t + 0.28);
      osc.connect(env).connect(this.filter);
      osc.start(t); osc.stop(t + 0.32);
    }

    _snare(t) {
      const n = this.ctx.createBufferSource();
      n.buffer = this.noiseBuf;
      const bp = this.ctx.createBiquadFilter();
      bp.type = 'bandpass'; bp.frequency.value = 1700; bp.Q.value = 1.1;
      const env = this.ctx.createGain();
      env.gain.setValueAtTime(0.0001, t);
      env.gain.exponentialRampToValueAtTime(0.55, t + 0.002);
      env.gain.exponentialRampToValueAtTime(0.001, t + 0.14);
      n.connect(bp).connect(env).connect(this.filter);
      n.start(t); n.stop(t + 0.18);
    }

    _hat(t, vel) {
      const n = this.ctx.createBufferSource();
      n.buffer = this.noiseBuf;
      const hp = this.ctx.createBiquadFilter();
      hp.type = 'highpass'; hp.frequency.value = 6500;
      const env = this.ctx.createGain();
      env.gain.setValueAtTime(0.0001, t);
      env.gain.exponentialRampToValueAtTime(vel * 0.35, t + 0.001);
      env.gain.exponentialRampToValueAtTime(0.001, t + 0.04);
      n.connect(hp).connect(env).connect(this.filter);
      n.start(t); n.stop(t + 0.06);
    }

    // Rim tick — short resonant click, played on blink.
    rim() {
      const t = this.ctx.currentTime;
      const n = this.ctx.createBufferSource();
      n.buffer = this.noiseBuf;
      const bp = this.ctx.createBiquadFilter();
      bp.type = 'bandpass'; bp.frequency.value = 3800; bp.Q.value = 3.0;
      const env = this.ctx.createGain();
      env.gain.setValueAtTime(0.0001, t);
      env.gain.exponentialRampToValueAtTime(0.32, t + 0.001);
      env.gain.exponentialRampToValueAtTime(0.001, t + 0.06);
      n.connect(bp).connect(env).connect(this.filter);
      n.start(t); n.stop(t + 0.08);
    }

    // Gentle chime for acknowledgements (prayer, mode switch).
    chime() {
      const t = this.ctx.currentTime;
      [880, 1318.51].forEach((f, i) => {
        const o = this.ctx.createOscillator();
        o.type = 'sine';
        o.frequency.value = f;
        const env = this.ctx.createGain();
        env.gain.setValueAtTime(0.0001, t);
        env.gain.exponentialRampToValueAtTime(0.12, t + 0.01 + i * 0.03);
        env.gain.exponentialRampToValueAtTime(0.001, t + 0.6);
        o.connect(env).connect(this.master);
        o.start(t); o.stop(t + 0.65);
      });
    }
  }

  // Theremin: continuous pitched tone gated on hand being raised.
  class Theremin {
    constructor(ctx, notes, opts) {
      opts = opts || {};
      this.ctx = ctx;
      this.notes = notes;
      this.gainLevel = opts.gainLevel != null ? opts.gainLevel : 0.11;
      this.oscType  = opts.oscType  || 'sine';

      this.lpf = ctx.createBiquadFilter();
      this.lpf.type = 'lowpass';
      this.lpf.frequency.value = opts.lpfHz != null ? opts.lpfHz : 2600;

      this.gain = ctx.createGain(); this.gain.gain.value = 0;
      this.lpf.connect(this.gain).connect(ctx.destination);

      this.osc = ctx.createOscillator();
      this.osc.type = this.oscType;
      this.osc.frequency.value = notes[0];
      this.osc.connect(this.lpf);
      this.osc.start();

      // octave-up shimmer (skip if octaveMix = 0)
      const mix = opts.octaveMix != null ? opts.octaveMix : 0.18;
      if (mix > 0) {
        this.osc2 = ctx.createOscillator();
        this.osc2.type = 'triangle';
        this.osc2.frequency.value = notes[0] * 2;
        const g2 = ctx.createGain(); g2.gain.value = mix;
        this.osc2.connect(g2).connect(this.lpf);
        this.osc2.start();
      }
    }

    setHandY(y /* null or number in [0, HAND_THRESHOLD=0.55] */) {
      const now = this.ctx.currentTime;
      if (y == null || y < 0) {
        this.gain.gain.setTargetAtTime(0, now, 0.14);
        return;
      }
      const t = 1.0 - Math.max(0, Math.min(1, y / 0.55));
      const idx = Math.round(t * (this.notes.length - 1));
      const freq = this.notes[idx];
      this.osc.frequency.setTargetAtTime(freq, now, 0.05);
      if (this.osc2) this.osc2.frequency.setTargetAtTime(freq * 2, now, 0.05);
      this.gain.gain.setTargetAtTime(this.gainLevel, now, 0.08);
    }
  }

  // C major pentatonic in two ranges.
  const BASS_NOTES = [
    65.41, 73.42, 82.41, 98.00, 110.00,       // C2 D2 E2 G2 A2
    130.81, 146.83, 164.81, 196.00, 220.00,   // C3 D3 E3 G3 A3
  ];
  const LEAD_NOTES = [
    523.25, 587.33, 659.25, 783.99, 880.00,        // C5 D5 E5 G5 A5
    1046.50, 1174.66, 1318.51, 1567.98, 1760.00,   // C6 D6 E6 G6 A6
  ];

  class WarmPad {
    constructor(ctx) {
      this.ctx = ctx;
      // C major 7 (ish): C3, E3, G3, B3 — spacious, Rhodes-y
      const freqs = [130.81, 164.81, 196.00, 246.94];
      const lpf = ctx.createBiquadFilter();
      lpf.type = 'lowpass'; lpf.frequency.value = 1600; lpf.Q.value = 0.6;

      const reverbDelay = ctx.createDelay();
      reverbDelay.delayTime.value = 0.11;
      const feedback = ctx.createGain(); feedback.gain.value = 0.32;
      reverbDelay.connect(feedback).connect(reverbDelay);

      this.gain = ctx.createGain();
      this.gain.gain.value = 0;

      lpf.connect(this.gain);
      lpf.connect(reverbDelay);
      reverbDelay.connect(this.gain);
      this.gain.connect(ctx.destination);

      this.oscs = [];
      freqs.forEach(f => {
        for (const d of [-6, 6]) {
          const o = ctx.createOscillator();
          o.type = 'triangle';
          o.frequency.value = f;
          o.detune.value = d;
          o.connect(lpf);
          o.start();
          this.oscs.push(o);
        }
      });
    }
    setAmount(v) {
      // v in [0,1]; target peak gain ~0.22
      const g = Math.max(0, Math.min(1, v)) * 0.22;
      this.gain.gain.setTargetAtTime(g, this.ctx.currentTime, 0.18);
    }
  }

  // A minor pentatonic scales (darker, Tame Impala-ish).
  const JAM_BASS_NOTES = [
    55.00, 65.41, 73.42, 82.41, 98.00,         // A1 C2 D2 E2 G2
    110.00, 130.81, 146.83, 164.81, 196.00,    // A2 C3 D3 E3 G3
  ];
  const JAM_LEAD_NOTES = [
    220.00, 261.63, 293.66, 329.63, 392.00,        // A3 C4 D4 E4 G4
    440.00, 523.25, 587.33, 659.25, 783.99,        // A4 C5 D5 E5 G5
    880.00,                                         // A5
  ];

  // Left hand = warm sawtooth bass through a resonant lowpass.
  // Y → pitch (up = higher). X → filter cutoff (left = muted, right = open).
  class JamBass {
    constructor(ctx) {
      this.ctx = ctx;
      this.notes = JAM_BASS_NOTES;

      this.lpf = ctx.createBiquadFilter();
      this.lpf.type = 'lowpass';
      this.lpf.frequency.value = 500;
      this.lpf.Q.value = 4.5;

      this.drive = ctx.createWaveShaper();
      const curve = new Float32Array(512);
      for (let i = 0; i < 512; i++) {
        const x = i / 256 - 1;
        curve[i] = Math.tanh(x * 2.2);
      }
      this.drive.curve = curve;

      this.gain = ctx.createGain(); this.gain.gain.value = 0;

      this.osc = ctx.createOscillator();
      this.osc.type = 'sawtooth';
      this.osc.frequency.value = this.notes[0];

      // Sub osc an octave down for heft.
      this.sub = ctx.createOscillator();
      this.sub.type = 'sine';
      this.sub.frequency.value = this.notes[0] / 2;
      const subGain = ctx.createGain(); subGain.gain.value = 0.35;

      this.osc.connect(this.lpf);
      this.sub.connect(subGain).connect(this.lpf);
      this.lpf.connect(this.drive).connect(this.gain).connect(ctx.destination);
      this.osc.start(); this.sub.start();
    }
    setXY(x, y) {
      const now = this.ctx.currentTime;
      if (x == null || y == null) {
        this.gain.gain.setTargetAtTime(0, now, 0.18);
        return;
      }
      const t = 1.0 - Math.max(0, Math.min(1, y / 0.55));
      const idx = Math.round(t * (this.notes.length - 1));
      const f = this.notes[idx];
      this.osc.frequency.setTargetAtTime(f, now, 0.06);
      this.sub.frequency.setTargetAtTime(f / 2, now, 0.06);
      // X: left edge = 250Hz, right edge = 2500Hz (exponential feel)
      const xc = Math.max(0, Math.min(1, x));
      const cutoff = 250 * Math.pow(10, xc * 1.0);
      this.lpf.frequency.setTargetAtTime(cutoff, now, 0.06);
      this.gain.gain.setTargetAtTime(0.18, now, 0.12);
    }
  }

  // Right hand = phased sawtooth pad through 4-stage allpass phaser + delay.
  // Y → note in A-minor pent. X → phaser LFO rate + wet mix.
  class JamLead {
    constructor(ctx) {
      this.ctx = ctx;
      this.notes = JAM_LEAD_NOTES;

      // source: sawtooth + detuned copy for thickness
      this.osc = ctx.createOscillator();
      this.osc.type = 'sawtooth';
      this.osc.frequency.value = this.notes[0];

      this.osc2 = ctx.createOscillator();
      this.osc2.type = 'sawtooth';
      this.osc2.detune.value = 8;
      this.osc2.frequency.value = this.notes[0];

      const mixer = ctx.createGain(); mixer.gain.value = 0.5;
      this.osc.connect(mixer);
      this.osc2.connect(mixer);

      // gentle lowpass to tame highs
      const lpf = ctx.createBiquadFilter();
      lpf.type = 'lowpass'; lpf.frequency.value = 2800; lpf.Q.value = 0.5;
      mixer.connect(lpf);

      // 4-stage allpass phaser
      this.apf = [];
      const lfo = ctx.createOscillator();
      lfo.type = 'sine';
      lfo.frequency.value = 0.35;
      const lfoGain = ctx.createGain(); lfoGain.gain.value = 500;
      lfo.connect(lfoGain);
      let node = lpf;
      for (let i = 0; i < 4; i++) {
        const a = ctx.createBiquadFilter();
        a.type = 'allpass';
        a.frequency.value = 400 + i * 350;
        a.Q.value = 6;
        lfoGain.connect(a.frequency);
        node.connect(a);
        this.apf.push(a);
        node = a;
      }
      lfo.start();
      this.lfo = lfo;
      this.lfoGain = lfoGain;

      // dry + wet
      this.dryGain = ctx.createGain(); this.dryGain.gain.value = 0.6;
      this.wetGain = ctx.createGain(); this.wetGain.gain.value = 0.4;
      lpf.connect(this.dryGain);
      node.connect(this.wetGain);

      // delay reverb for depth
      const delay = ctx.createDelay();
      delay.delayTime.value = 0.22;
      const fb = ctx.createGain(); fb.gain.value = 0.38;
      delay.connect(fb).connect(delay);
      const wetRev = ctx.createGain(); wetRev.gain.value = 0.35;

      this.gain = ctx.createGain(); this.gain.gain.value = 0;
      this.dryGain.connect(this.gain);
      this.wetGain.connect(this.gain);
      this.wetGain.connect(delay);
      delay.connect(wetRev).connect(this.gain);
      this.gain.connect(ctx.destination);

      this.osc.start(); this.osc2.start();
    }
    setXY(x, y) {
      const now = this.ctx.currentTime;
      if (x == null || y == null) {
        this.gain.gain.setTargetAtTime(0, now, 0.2);
        return;
      }
      const t = 1.0 - Math.max(0, Math.min(1, y / 0.55));
      const idx = Math.round(t * (this.notes.length - 1));
      const f = this.notes[idx];
      this.osc.frequency.setTargetAtTime(f, now, 0.05);
      this.osc2.frequency.setTargetAtTime(f, now, 0.05);
      // X: left = slow dreamy phaser, right = fast swirling + more wet
      const xc = Math.max(0, Math.min(1, x));
      const rate = 0.15 + xc * 3.5;
      this.lfo.frequency.setTargetAtTime(rate, now, 0.1);
      this.wetGain.gain.setTargetAtTime(0.2 + xc * 0.7, now, 0.1);
      this.dryGain.gain.setTargetAtTime(0.7 - xc * 0.4, now, 0.1);
      this.gain.gain.setTargetAtTime(0.13, now, 0.12);
    }
  }

  function unlockAudio() {
    if (audioReady) return;
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    drums = new BobDrums(ctx);
    pad = new WarmPad(ctx);
    leftTh  = new Theremin(ctx, BASS_NOTES, {
      gainLevel: 0.14, oscType: 'triangle', octaveMix: 0.10, lpfHz: 1800,
    });
    rightTh = new Theremin(ctx, LEAD_NOTES, {
      gainLevel: 0.10, oscType: 'sine',     octaveMix: 0.20, lpfHz: 3200,
    });
    jamBass = new JamBass(ctx);
    jamLead = new JamLead(ctx);
    audioReady = true;
    statusEl.textContent = 'audio ready';
    statusEl.className = 'status on';
  }

  // Cheerful ascending chime — plays on double-clap boot.
  function bootChime() {
    if (!ctx) return;
    const t = ctx.currentTime;
    const notes = [523.25, 659.25, 783.99, 1046.50]; // C5 E5 G5 C6
    notes.forEach((f, i) => {
      const o = ctx.createOscillator();
      o.type = 'triangle';
      o.frequency.value = f;
      const g = ctx.createGain();
      const start = t + i * 0.08;
      g.gain.setValueAtTime(0.0001, start);
      g.gain.exponentialRampToValueAtTime(0.18, start + 0.01);
      g.gain.exponentialRampToValueAtTime(0.001, start + 0.5);
      o.connect(g).connect(ctx.destination);
      o.start(start); o.stop(start + 0.55);
    });
  }

  function flashBob() {
    flashEl.classList.add('on');
    setTimeout(() => flashEl.classList.remove('on'), 160);
  }

  function handBarPct(y) {
    if (y == null) return 0;
    return Math.round(Math.max(0, Math.min(1, 1 - y / 0.55)) * 100);
  }

  function connectEvents() {
    const es = new EventSource('/events');
    es.onmessage = (e) => {
      let msg; try { msg = JSON.parse(e.data); } catch { return; }

      if (msg.swipe) flashBob();

      if (typeof msg.smile === 'number') {
        smileFill.style.width = Math.round(msg.smile * 100) + '%';
        if (pad) pad.setAmount(msg.smile);
      }
      if ('leftHandY' in msg) {
        leftFill.style.width = handBarPct(msg.leftHandY) + '%';
        if (jamMode) {
          if (jamBass) jamBass.setXY(msg.leftHandX, msg.leftHandY);
          if (leftTh) leftTh.setHandY(null);
        } else {
          if (leftTh) leftTh.setHandY(msg.leftHandY);
          if (jamBass) jamBass.setXY(null, null);
        }
      }
      if ('rightHandY' in msg) {
        rightFill.style.width = handBarPct(msg.rightHandY) + '%';
        if (jamMode) {
          if (jamLead) jamLead.setXY(msg.rightHandX, msg.rightHandY);
          if (rightTh) rightTh.setHandY(null);
        } else {
          if (rightTh) rightTh.setHandY(msg.rightHandY);
          if (jamLead) jamLead.setXY(null, null);
        }
      }
      if (msg.bob) {
        if (drums) drums.hit();
        flashBob();
      }
      if (msg.blink && drums) drums.rim();

      if (msg.prayerStart) {
        listenPill.style.display = 'inline-block';
        notesEl.focus();
        if (drums) drums.chime();
        flashBob();
      }
      if (msg.prayerEnd) {
        listenPill.style.display = 'none';
      }
      if (msg.boot) {
        // Double-clap: toggle voice listening.
        toggleVoice();
        flashBob();
      }
      if ('pointing' in msg) paintActive(ccPoint, msg.pointing);
      if ('click'    in msg) paintActive(ccClick, msg.click);
      if ('wispr'    in msg) {
        ccWispr.querySelectorAll('.cc-opt').forEach(x => {
          if (x.id !== 'cc-wispr-fire')
            x.classList.toggle('on', x.dataset.v === msg.wispr);
        });
      }
      if ('scrollSens' in msg) paintActive(ccScroll, msg.scrollSens);
      if ('rightClick' in msg) paintActive(ccRClick, msg.rightClick);
      if ('doubleClick' in msg) paintActive(ccDClick, msg.doubleClick ? 'on' : 'off');
      if (msg.dictation) {
        const notesEl = document.getElementById('notes');
        if (notesEl) {
          // Expand the details section if collapsed, then focus.
          const parent = notesEl.closest('details.sec');
          if (parent && !parent.open) parent.open = true;
          try { notesEl.focus(); } catch (e) {}
        }
      }
      if ('cursor' in msg) {
        const state = !msg.cursor ? 'off'
          : (msg.cursorCalibrated ? 'live' : 'calibrating…');
        if (ccCursorState) ccCursorState.textContent = state;
        if (ccCursorOn)  ccCursorOn.classList.toggle('on', !!msg.cursor);
        if (ccCursorOff) ccCursorOff.classList.toggle('on', !msg.cursor);
        if (msg.cursor) {
          cursorPill.style.display = 'inline-block';
          const suffix = msg.pointing && msg.click
            ? ' · ' + msg.pointing + '/' + msg.click : '';
          cursorPill.textContent = (msg.cursorCalibrated
            ? 'cursor live' : 'cursor calibrating…') + suffix;
        } else {
          cursorPill.style.display = 'none';
        }
      }
    };
  }

  // Auto-unlock on first page interaction (click/keydown anywhere satisfies
  // the browser's autoplay policy). After that, double-clap really does "boot".
  function firstInteraction() {
    unlockAudio();
    startVoice();  // prompts mic permission; user must allow.
    window.removeEventListener('pointerdown', firstInteraction);
    window.removeEventListener('keydown', firstInteraction);
  }
  window.addEventListener('pointerdown', firstInteraction);
  window.addEventListener('keydown', firstInteraction);

  connectEvents();

  // Shooting stars — dim by default, brighter when jam mode is on.
  (function stars() {
    const canvas = document.getElementById('stars');
    const ctx = canvas.getContext('2d');
    let w = 0, h = 0, dpr = window.devicePixelRatio || 1;
    function resize() {
      w = window.innerWidth; h = window.innerHeight;
      canvas.width  = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width  = w + 'px';
      canvas.style.height = h + 'px';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    window.addEventListener('resize', resize);
    resize();
    const shooters = [];
    function spawn() {
      const fromLeft = Math.random() < 0.5;
      shooters.push({
        x: fromLeft ? -40 : w + 40,
        y: Math.random() * h * 0.7,
        vx: (fromLeft ? 1 : -1) * (4 + Math.random() * 4),
        vy: 1.2 + Math.random() * 1.8,
        life: 1.0,
        hue: 180 + Math.random() * 120,
      });
    }
    function tick() {
      ctx.clearRect(0, 0, w, h);
      // ambient tiny twinkle dots
      const jam = document.body.classList.contains('jam');
      const baseAlpha = jam ? 0.5 : 0.28;
      for (let i = 0; i < 60; i++) {
        const sx = (i * 97 + (performance.now() * 0.02)) % w;
        const sy = (i * 53) % h;
        const tw = 0.4 + 0.6 * Math.abs(Math.sin(performance.now()*0.001 + i));
        ctx.fillStyle = `rgba(200,220,255,${baseAlpha * tw * 0.35})`;
        ctx.fillRect(sx, sy, 1.2, 1.2);
      }
      if (Math.random() < (jam ? 0.035 : 0.012)) spawn();
      for (let i = shooters.length - 1; i >= 0; i--) {
        const s = shooters[i];
        // trail
        const grad = ctx.createLinearGradient(
          s.x, s.y, s.x - s.vx * 12, s.y - s.vy * 12);
        grad.addColorStop(0, `hsla(${s.hue}, 90%, 75%, ${0.9*s.life})`);
        grad.addColorStop(1, `hsla(${s.hue}, 90%, 70%, 0)`);
        ctx.strokeStyle = grad;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(s.x, s.y);
        ctx.lineTo(s.x - s.vx * 12, s.y - s.vy * 12);
        ctx.stroke();
        // head
        ctx.fillStyle = `hsla(${s.hue}, 100%, 92%, ${s.life})`;
        ctx.beginPath();
        ctx.arc(s.x, s.y, 1.8, 0, Math.PI * 2);
        ctx.fill();
        s.x += s.vx; s.y += s.vy;
        s.life -= 0.01;
        if (s.life <= 0 || s.x < -80 || s.x > w + 80 || s.y > h + 80) {
          shooters.splice(i, 1);
        }
      }
      requestAnimationFrame(tick);
    }
    tick();
  })();
})();
</script>
</body></html>
"""


# --- shared state ---------------------------------------------------------
_latest_jpeg: Optional[bytes] = None
_jpeg_lock = threading.Lock()
_stop = threading.Event()

_state_lock = threading.Lock()
_motion_val = 0.0   # still computed; not sent
_smile_val = 0.0
_bob_pending = False
_blink_pending = False
_prayer_start_pending = False
_prayer_end_pending = False
_prayer_active = False
_boot_pending = False
_swipe_pending: Optional[str] = None  # "left" or "right"
_dictation_pending = False
_left_hand_y: Optional[float] = None
_right_hand_y: Optional[float] = None
_left_hand_x: Optional[float] = None
_right_hand_x: Optional[float] = None
_jam_mode: bool = False

# Cursor prototyping: pointing method + click method, hot-swappable from UI.
_pointing_method: str = "finger"  # "head" | "finger" | "gaze"
_click_method: str = "brow"       # primary (left) click gesture
_right_click_method: str = "off"  # "smile" | "pucker" | "furrow" | "off"
_double_click_on: bool = False    # double-tap primary within DBL window → double-click
_wispr_method: str = "all"        # "applescript_fn"|"cgevent_f19"|"cgevent_fn"|"all"|"apple_dictation"|"off"

# Calibration centers (captured on cursor enable, per method).
_finger_center: Optional[tuple] = None   # (fx, fy) in [0..1]
_gaze_center: Optional[tuple] = None     # (gx, gy) in [0..1]

# Click detector state
_last_brow_click_at: float = 0.0
_pinch_was_closed: bool = False
_last_pinch_click_at: float = 0.0
_last_wink_click_at: float = 0.0
_wink_armed: bool = True    # edge-trigger gate so one wink = one click
_last_blink_click_at: float = 0.0
_blink_click_armed: bool = True
_last_right_wink_click_at: float = 0.0
_right_wink_armed: bool = True
_last_mouth_click_at: float = 0.0
_mouth_armed: bool = True
_last_smile_click_at: float = 0.0
_smile_click_armed: bool = True
_last_pucker_click_at: float = 0.0
_pucker_click_armed: bool = True
_last_furrow_click_at: float = 0.0
_furrow_click_armed: bool = True
# Pending-click mechanism so two primaries within DOUBLE_WINDOW_S merge into
# one double-click instead of firing two singles.
_pending_click_at: float = 0.0
_pending_click_waiting: bool = False
DOUBLE_WINDOW_S: float = 0.35
# Thresholds for new right-click gestures
SMILE_CLICK_THRESHOLD = 0.75
SMILE_CLICK_OPEN_THR  = 0.45
SMILE_CLICK_COOLDOWN_S = 0.5
PUCKER_CLICK_THRESHOLD = 0.55
PUCKER_CLICK_OPEN_THR  = 0.25
PUCKER_CLICK_COOLDOWN_S = 0.5
FURROW_CLICK_THRESHOLD = 0.55
FURROW_CLICK_OPEN_THR  = 0.25
FURROW_CLICK_COOLDOWN_S = 0.5

# Two-hand vertical scroll: both hands in frame → scroll mode.
# While in scroll mode the cursor is frozen and avg-y velocity scrolls.
_scroll_prev_y: Optional[float] = None
_scroll_active: bool = False
_scroll_accum: float = 0.0
SCROLL_ZONE_Y_MAX  = 0.85   # both wrists above waist
SCROLL_FRAME_DEAD  = 0.0015 # ignore per-frame jitter below this
SCROLL_GAIN_MAP = {"gentle": 80.0, "normal": 180.0, "zippy": 360.0}
_scroll_sens: str = "normal"
BROW_CLICK_THRESHOLD = 0.55
BROW_CLICK_COOLDOWN_S = 0.7
WINK_CLOSED_THRESHOLD = 0.55   # one eye must be this closed…
WINK_OPEN_THRESHOLD   = 0.3    # …while the other stays below this (loose rearm)
WINK_COOLDOWN_S       = 0.5
BLINK_CLICK_THRESHOLD = 0.55   # both eyes closed: score on both
BLINK_CLICK_OPEN_THR  = 0.3
BLINK_CLICK_COOLDOWN_S = 0.45
MOUTH_CLICK_THRESHOLD = 0.45
MOUTH_CLICK_OPEN_THR  = 0.2
MOUTH_CLICK_COOLDOWN_S = 0.4
PINCH_CLOSE_THRESHOLD = 0.055   # normalized (frame coords)
PINCH_OPEN_THRESHOLD = 0.085
PINCH_COOLDOWN_S = 0.35

# Finger pointing: remap a smaller box to full screen so he doesn't have to
# reach frame edges.
FINGER_GAIN_X = 2.2
FINGER_GAIN_Y = 2.0

# Gaze pointing: iris deltas are tiny → high gain.
GAZE_GAIN_X = 12.0
GAZE_GAIN_Y = 10.0

_prev_face_nose: Optional[tuple] = None
_prev_hand_wrists: list = []
_nose_y_history: Deque[float] = deque(maxlen=BOB_WINDOW_FRAMES)
_last_bob_at = 0.0

_blink_armed = True
_last_blink_at = 0.0

_swipe_xs: Deque[tuple] = deque(maxlen=40)
_last_swipe_at = 0.0

_prayer_hold_start: Optional[float] = None
_prayer_active_since: Optional[float] = None
_prayer_hands_lost_since: Optional[float] = None

_hands_up_start: Optional[float] = None
_last_hands_up_at = 0.0

_clap_state = "far"  # or "close"
_clap_times: List[float] = []
_last_clap_boot_at = 0.0

_fn_pressed = False

_screen_w, _screen_h = pyautogui.size()

# Virtual-desktop bounding box across ALL monitors.  On macOS the cursor
# lives in a coord space where (0,0) is the top-left of the primary display
# and secondary displays can have negative or > _screen_w coordinates.
# We compute the union via AppKit so gesture-driven cursor can traverse
# every monitor the same way the trackpad does.
def _virtual_screen_bounds() -> tuple:
    try:
        from AppKit import NSScreen
        screens = NSScreen.screens()
        if not screens:
            return 0.0, 0.0, float(_screen_w), float(_screen_h)
        # NSScreen frames use a bottom-left origin where y grows upward.
        # Mouse coords use a top-left origin where y grows downward.
        # Flip y using the primary (index 0) screen height.
        primary_h = float(screens[0].frame().size.height)
        xs_min = []
        ys_min = []
        xs_max = []
        ys_max = []
        for s in screens:
            f = s.frame()
            x = float(f.origin.x)
            y_bl = float(f.origin.y)
            w = float(f.size.width)
            h = float(f.size.height)
            # Convert bottom-left-origin y to top-left-origin y.
            y_tl = primary_h - (y_bl + h)
            xs_min.append(x)
            ys_min.append(y_tl)
            xs_max.append(x + w)
            ys_max.append(y_tl + h)
        return min(xs_min), min(ys_min), max(xs_max), max(ys_max)
    except Exception as e:
        print(f"[viewer] virtual-screen probe failed: {e}", flush=True)
        return 0.0, 0.0, float(_screen_w), float(_screen_h)


_virt_x0, _virt_y0, _virt_x1, _virt_y1 = _virtual_screen_bounds()
_virt_w = _virt_x1 - _virt_x0
_virt_h = _virt_y1 - _virt_y0
print(f"[viewer] virtual screen: ({_virt_x0:.0f},{_virt_y0:.0f}) → "
      f"({_virt_x1:.0f},{_virt_y1:.0f})  size {_virt_w:.0f}x{_virt_h:.0f}  "
      f"(primary {_screen_w}x{_screen_h})", flush=True)

_cursor_enabled = False
_cursor_calibrated = False
_cursor_calib_start: Optional[float] = None
_yaw_center = 0.0
_pitch_center = 0.0
_cur_x = (_virt_x0 + _virt_x1) * 0.5
_cur_y = (_virt_y0 + _virt_y1) * 0.5


def _matrix_to_yaw_pitch(matrix) -> tuple:
    r = np.array(matrix).reshape(4, 4)[:3, :3]
    sy = float(np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2))
    if sy > 1e-6:
        yaw = float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))
        pitch = float(np.degrees(np.arctan2(-r[2, 0], sy)))
    else:
        yaw = 0.0
        pitch = float(np.degrees(np.arctan2(-r[2, 0], sy)))
    return yaw, pitch


def _draw_face_landmarks(frame_bgr: np.ndarray, landmarks) -> None:
    h, w = frame_bgr.shape[:2]
    for lm in landmarks[::4]:
        cv2.circle(frame_bgr, (int(lm.x * w), int(lm.y * h)), 1,
                   FACE_DOT_COLOR_BGR, -1)


def _draw_hand(frame_bgr: np.ndarray, hand_landmarks) -> None:
    h, w = frame_bgr.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame_bgr, pts[a], pts[b], HAND_LINE_COLOR_BGR, 1,
                 cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        r = 4 if i in (4, 8, 12, 16, 20) else 2
        cv2.circle(frame_bgr, (x, y), r, HAND_DOT_COLOR_BGR, -1, cv2.LINE_AA)


def _palm_center(hand):
    xs = [hand[i].x for i in PALM_IDXS]
    ys = [hand[i].y for i in PALM_IDXS]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _ensure_model(path: Path, url: str) -> bool:
    if path.exists():
        return True
    try:
        print(f"[viewer] downloading {path.name}...", flush=True)
        urllib.request.urlretrieve(url, path)
        return True
    except Exception as e:
        print(f"[viewer] download failed: {e}", flush=True)
        return False


def _detect_bob(nose_y: Optional[float], now: float) -> bool:
    """Fire at the bottom of a head bob.

    Uses a rolling window of recent nose_y values. A bob has two halves:
    descent (nose_y rises — head moves down) then return (nose_y falls).
    We fire when the earlier half has non-trivial descent and the latest
    two frames are either flat or rising back up (head reached bottom).
    """
    global _last_bob_at
    if nose_y is None:
        return False
    _nose_y_history.append(nose_y)
    if len(_nose_y_history) < 5:
        return False
    hist = list(_nose_y_history)
    # Descent of the earlier 3 frames (positive = nose moved down).
    earlier_descent = hist[-3] - hist[-5]
    # Recent 2 frames velocity (positive = still falling down).
    recent_vel = hist[-1] - hist[-3]
    if (earlier_descent > BOB_DESCENT_THRESHOLD
            and recent_vel <= 0
            and (now - _last_bob_at) > BOB_COOLDOWN_S):
        _last_bob_at = now
        return True
    return False


def _detect_blink(blendshapes, now: float) -> bool:
    """Fire on a simultaneous both-eye blink."""
    global _blink_armed, _last_blink_at
    if not blendshapes:
        return False
    l = r = 0.0
    for b in blendshapes:
        if b.category_name == "eyeBlinkLeft":
            l = float(b.score)
        elif b.category_name == "eyeBlinkRight":
            r = float(b.score)
    combined = (l + r) * 0.5
    if (_blink_armed and combined > BLINK_ON_THRESHOLD
            and (now - _last_blink_at) > BLINK_COOLDOWN_S):
        _last_blink_at = now
        _blink_armed = False
        return True
    if not _blink_armed and combined < BLINK_OFF_THRESHOLD:
        _blink_armed = True
    return False


def _update_swipe(hands_list, now: float) -> Optional[str]:
    """Record one-handed X over time; fire 'left'/'right' on a full traversal."""
    global _last_swipe_at
    # Only sample when exactly one hand is in frame, so two-hand shuffling
    # can't produce phantom jumps in the tracked X.
    if len(hands_list) == 1:
        _swipe_xs.append((now, hands_list[0][0].x))
    else:
        # Drop stale history while multi-hand or no-hand.
        while _swipe_xs and now - _swipe_xs[0][0] > SWIPE_WINDOW_S:
            _swipe_xs.popleft()
        return None

    # Trim old samples.
    while _swipe_xs and now - _swipe_xs[0][0] > SWIPE_WINDOW_S:
        _swipe_xs.popleft()
    if len(_swipe_xs) < 4:
        return None

    first_x = _swipe_xs[0][1]
    last_x = _swipe_xs[-1][1]
    delta = last_x - first_x
    direction: Optional[str] = None
    if (delta > SWIPE_MIN_TRAVEL
            and first_x < SWIPE_EDGE_START and last_x > SWIPE_EDGE_END):
        direction = "right"
    elif (delta < -SWIPE_MIN_TRAVEL
            and first_x > SWIPE_EDGE_END and last_x < SWIPE_EDGE_START):
        direction = "left"
    if direction and (now - _last_swipe_at) > SWIPE_COOLDOWN_S:
        _last_swipe_at = now
        _swipe_xs.clear()
        return direction
    return None


def _update_prayer(hands_list, now: float) -> Optional[bool]:
    """Track prayer hold. Returns True on enter, False on exit, None otherwise.

    Release rules (any triggers an exit):
      - palms pull apart past PRAYER_OPEN_THRESHOLD
      - fewer than 2 hands visible for PRAYER_LOST_HANDS_GRACE_S
      - safety timeout after PRAYER_MAX_HOLD_S
    """
    global _prayer_hold_start, _prayer_active
    global _prayer_active_since, _prayer_hands_lost_since

    def _release(reason: str) -> bool:
        global _prayer_active, _prayer_active_since, _prayer_hands_lost_since
        global _prayer_hold_start
        _prayer_active = False
        _prayer_active_since = None
        _prayer_hands_lost_since = None
        _prayer_hold_start = None
        print(f"[viewer] prayer release ({reason})", flush=True)
        return False

    # Safety timeout.
    if (_prayer_active and _prayer_active_since is not None
            and now - _prayer_active_since > PRAYER_MAX_HOLD_S):
        return _release("timeout")

    if len(hands_list) < 2:
        _prayer_hold_start = None
        if _prayer_active:
            if _prayer_hands_lost_since is None:
                _prayer_hands_lost_since = now
            elif now - _prayer_hands_lost_since > PRAYER_LOST_HANDS_GRACE_S:
                return _release("hands lost")
        return None

    _prayer_hands_lost_since = None
    p0 = _palm_center(hands_list[0])
    p1 = _palm_center(hands_list[1])
    d = math.hypot(p0[0] - p1[0], p0[1] - p1[1])

    if d < PRAYER_CLOSE_THRESHOLD:
        if _prayer_hold_start is None:
            _prayer_hold_start = now
        elif not _prayer_active and now - _prayer_hold_start >= PRAYER_ENTER_HOLD_S:
            _prayer_active = True
            _prayer_active_since = now
            print(f"[viewer] prayer enter (d={d:.3f})", flush=True)
            return True
    elif d > PRAYER_OPEN_THRESHOLD:
        _prayer_hold_start = None
        if _prayer_active:
            return _release(f"open d={d:.3f}")
    return None


def _update_hands_up(hands_list, now: float) -> bool:
    """Fire True when both wrists held above HANDS_UP_Y_MAX for HANDS_UP_HOLD_S."""
    global _hands_up_start, _last_hands_up_at
    if len(hands_list) < 2:
        _hands_up_start = None
        return False
    y0 = hands_list[0][0].y
    y1 = hands_list[1][0].y
    if y0 < HANDS_UP_Y_MAX and y1 < HANDS_UP_Y_MAX:
        if _hands_up_start is None:
            _hands_up_start = now
        elif (now - _hands_up_start >= HANDS_UP_HOLD_S
                and now - _last_hands_up_at > HANDS_UP_COOLDOWN_S):
            _last_hands_up_at = now
            _hands_up_start = None
            return True
    else:
        _hands_up_start = None
    return False


def _update_double_clap(hands_list, now: float) -> bool:
    """Fire True on two quick claps (palms colliding twice) within the window."""
    global _clap_state, _clap_times, _last_clap_boot_at
    if len(hands_list) < 2:
        return False
    p0 = _palm_center(hands_list[0])
    p1 = _palm_center(hands_list[1])
    d = math.hypot(p0[0] - p1[0], p0[1] - p1[1])
    if _clap_state == "far" and d < CLAP_CLOSE_THRESHOLD:
        _clap_state = "close"
        _clap_times.append(now)
        _clap_times = [t for t in _clap_times if now - t < CLAP_GAP_MAX_S * 2]
        if len(_clap_times) >= 2:
            gap = _clap_times[-1] - _clap_times[-2]
            if (CLAP_GAP_MIN_S < gap < CLAP_GAP_MAX_S
                    and now - _last_clap_boot_at > CLAP_BOOT_COOLDOWN_S):
                _last_clap_boot_at = now
                _clap_times.clear()
                return True
    elif _clap_state == "close" and d > CLAP_FAR_THRESHOLD:
        _clap_state = "far"
    return False


def _tap_wispr_cgevent(keycode: int, fn_flag: bool = False) -> None:
    if not _QUARTZ_OK:
        print("[viewer] cgevent skipped: Quartz unavailable", flush=True)
        return
    try:
        down = CGEventCreateKeyboardEvent(None, keycode, True)
        up   = CGEventCreateKeyboardEvent(None, keycode, False)
        if fn_flag:
            CGEventSetFlags(down, kCGEventFlagMaskSecondaryFn)
            CGEventSetFlags(up,   kCGEventFlagMaskSecondaryFn)
        CGEventPost(kCGHIDEventTap, down)
        CGEventPost(kCGHIDEventTap, up)
        print(f"[viewer] wispr cgevent tapped (key={keycode} fn={fn_flag})",
              flush=True)
    except Exception as e:
        print(f"[viewer] cgevent tap failed: {e}", flush=True)


def _tap_wispr_applescript_fn() -> None:
    """AppleScript 'key code 63' — the Fn key via System Events.
    Works in some apps but not all because Wispr uses IOKit HID."""
    script = 'tell application "System Events" to key code 63'
    try:
        subprocess.run(["osascript", "-e", script],
                       check=False, timeout=1.0)
        print("[viewer] wispr applescript tapped (key code 63)", flush=True)
    except Exception as e:
        print(f"[viewer] applescript tap failed: {e}", flush=True)


def _tap_apple_dictation() -> None:
    """Trigger built-in macOS Dictation. User must have set the dictation
    shortcut to 'Press Control key twice' in System Settings → Keyboard →
    Dictation. We tap left Control (key code 59) twice with a short gap.

    We also raise a `dictationPending` flag and try to activate a browser
    window + focus the notes textarea, so dictation lands in the page.
    """
    global _dictation_pending
    # Tell the browser to focus the notes textarea.
    with _state_lock:
        _dictation_pending = True
    # Best-effort: activate Arc (Jack's default). Falls through silently.
    try:
        subprocess.run(
            ["osascript", "-e", 'tell application "Arc" to activate'],
            check=False, timeout=0.8,
        )
    except Exception:
        pass
    # Small delay so the browser has time to focus before the trigger fires.
    time.sleep(0.25)
    if not _QUARTZ_OK:
        print("[viewer] apple dictation skipped: Quartz unavailable",
              flush=True)
        return
    try:
        for _ in range(2):
            d = CGEventCreateKeyboardEvent(None, 59, True)
            u = CGEventCreateKeyboardEvent(None, 59, False)
            CGEventPost(kCGHIDEventTap, d)
            CGEventPost(kCGHIDEventTap, u)
            time.sleep(0.08)
        print("[viewer] apple dictation: double-tapped Control", flush=True)
    except Exception as e:
        print(f"[viewer] apple dictation failed: {e}", flush=True)


def _tap_wispr_hotkey() -> None:
    """Dispatch on `_wispr_method`. 'all' fires each variant on a
    background thread with short gaps so we don't stall the capture loop."""
    method = _wispr_method
    if method == "off":
        return
    if method == "applescript_fn":
        _tap_wispr_applescript_fn()
    elif method == "cgevent_f19":
        _tap_wispr_cgevent(80, fn_flag=False)
    elif method == "cgevent_fn":
        _tap_wispr_cgevent(63, fn_flag=True)
    elif method == "apple_dictation":
        _tap_apple_dictation()
    elif method == "all":
        def _fire_all():
            _tap_wispr_cgevent(80, fn_flag=False)
            time.sleep(0.2)
            _tap_wispr_cgevent(63, fn_flag=True)
            time.sleep(0.2)
            _tap_wispr_applescript_fn()
        threading.Thread(target=_fire_all, daemon=True).start()
    else:
        _tap_wispr_cgevent(WISPR_KEYCODE, fn_flag=WISPR_USE_FN_FLAG)


def _hold_wispr_key() -> None:
    _tap_wispr_hotkey()


def _release_wispr_key() -> None:
    pass


def _fire_swipe_action(direction: str) -> None:
    """Swipe right → next desktop (Ctrl+Right). Swipe left → prev (Ctrl+Left).

    Requires System Settings → Keyboard Shortcuts → Mission Control →
    'Move left/right a space' to be enabled (they are by default).
    """
    key_code = 124 if direction == "right" else 123  # Right / Left arrow
    script = (
        'tell application "System Events" to '
        f'key code {key_code} using {{control down}}'
    )
    try:
        subprocess.run(
            ["osascript", "-e", script], check=False, timeout=1.0,
        )
        print(f"[viewer] swipe {direction} → ctrl+{'→' if direction=='right' else '←'}",
              flush=True)
    except Exception as e:
        print(f"[viewer] swipe action failed: {e}", flush=True)


def _split_hands_left_right(hand_landmarks_list):
    """Return (left_y, right_y) — wrist Y for the hand on each side of the
    (mirror-flipped) frame. Either can be None when a hand is missing or
    lowered below the play threshold."""
    lx, ly, rx, ry = _split_hands_xy(hand_landmarks_list)
    return ly, ry


def _split_hands_xy(hand_landmarks_list):
    """Return (left_x, left_y, right_x, right_y) — wrist XY for the hand
    on each side of the (mirror-flipped) frame. Any entry is None when
    that hand is missing or lowered below the play threshold."""
    raised = []
    for hand in hand_landmarks_list:
        x, y = hand[0].x, hand[0].y
        if y < HAND_PLAY_THRESHOLD_Y:
            raised.append((x, y))
    if not raised:
        return None, None, None, None
    if len(raised) == 1:
        x, y = raised[0]
        return (x, y, None, None) if x < 0.5 else (None, None, x, y)
    raised.sort(key=lambda xy: xy[0])
    lx, ly = raised[0]
    rx, ry = raised[1]
    return lx, ly, rx, ry


def _update_motion(face_nose, hand_wrists) -> None:
    """Rough "how much are you moving" score, 0..1, EMA-smoothed."""
    global _motion_val, _prev_face_nose, _prev_hand_wrists
    instant = 0.0
    samples = 0
    if face_nose is not None and _prev_face_nose is not None:
        dx = face_nose[0] - _prev_face_nose[0]
        dy = face_nose[1] - _prev_face_nose[1]
        instant += math.hypot(dx, dy) * 6.0
        samples += 1
    for cur, prev in zip(hand_wrists, _prev_hand_wrists):
        dx = cur[0] - prev[0]
        dy = cur[1] - prev[1]
        instant += math.hypot(dx, dy) * 3.5
        samples += 1
    if samples > 0:
        instant = min(1.0, instant / samples)
    else:
        instant = 0.0
    _motion_val = (1 - MOTION_EMA_ALPHA) * _motion_val + MOTION_EMA_ALPHA * instant
    _prev_face_nose = face_nose
    _prev_hand_wrists = hand_wrists


def _update_two_hand_scroll(hands_list, now: float) -> int:
    """Two hands visible → scroll mode. Returns signed scroll amount
    (positive = up, negative = down, 0 = no scroll this frame).
    Also flips `_scroll_active` so the cursor can freeze."""
    global _scroll_prev_y, _scroll_active, _scroll_accum
    if _jam_mode or len(hands_list) < 2:
        _scroll_prev_y = None
        _scroll_active = False
        _scroll_accum = 0.0
        return 0
    y0 = hands_list[0][0].y
    y1 = hands_list[1][0].y
    if y0 > SCROLL_ZONE_Y_MAX or y1 > SCROLL_ZONE_Y_MAX:
        _scroll_prev_y = None
        _scroll_active = False
        _scroll_accum = 0.0
        return 0
    _scroll_active = True
    avg = (y0 + y1) * 0.5
    if _scroll_prev_y is None:
        _scroll_prev_y = avg
        return 0
    dy = avg - _scroll_prev_y
    _scroll_prev_y = avg
    if abs(dy) < SCROLL_FRAME_DEAD:
        return 0
    gain = SCROLL_GAIN_MAP.get(_scroll_sens, 180.0)
    # hands moving up (dy < 0) → scroll up (positive pyautogui amount)
    _scroll_accum += -dy * gain
    if abs(_scroll_accum) < 1.0:
        return 0
    amount = int(_scroll_accum)
    _scroll_accum -= amount
    return amount


def _pick_right_hand(hands_lm_list):
    """Return the hand landmark list for the rightmost hand (mirror-flipped
    frame → user's right). Falls back to any visible hand so pinch works
    even with a single hand in view."""
    if not hands_lm_list:
        return None
    return max(hands_lm_list, key=lambda h: h[0].x)


def _target_from_head(face_matrix) -> Optional[tuple]:
    """Head pose → screen (target_x, target_y) or None."""
    if face_matrix is None:
        return None
    yaw, pitch = _matrix_to_yaw_pitch(face_matrix)
    # The matrix decomposition in this file labels the axes as (yaw, pitch)
    # but empirically `yaw` drives vertical and `pitch` drives horizontal
    # cursor movement on Jack's camera. Swap them to match reality.
    d_h = pitch - _pitch_center
    d_v = yaw - _yaw_center
    if abs(d_h) < CURSOR_DEAD_ZONE_DEG:
        d_h = 0.0
    if abs(d_v) < CURSOR_DEAD_ZONE_DEG:
        d_v = 0.0
    mid_x = (_virt_x0 + _virt_x1) * 0.5
    mid_y = (_virt_y0 + _virt_y1) * 0.5
    tx = mid_x + d_h * CURSOR_SENSITIVITY_X
    ty = mid_y - d_v * CURSOR_SENSITIVITY_Y
    return tx, ty


def _target_from_finger(hands_lm_list) -> Optional[tuple]:
    """Right-hand index fingertip (landmark 8) → screen."""
    hand = _pick_right_hand(hands_lm_list)
    if hand is None or _finger_center is None:
        return None
    fx, fy = hand[8].x, hand[8].y
    cx, cy = _finger_center
    dx = (fx - cx) * FINGER_GAIN_X
    dy = (fy - cy) * FINGER_GAIN_Y
    mid_x = (_virt_x0 + _virt_x1) * 0.5
    mid_y = (_virt_y0 + _virt_y1) * 0.5
    tx = mid_x + dx * _virt_w
    ty = mid_y + dy * _virt_h
    return tx, ty


def _iris_center(face_landmarks) -> Optional[tuple]:
    """Average of left iris (468) and right iris (473) landmark centers."""
    if face_landmarks is None or len(face_landmarks) < 478:
        return None
    a = face_landmarks[468]
    b = face_landmarks[473]
    return ((a.x + b.x) * 0.5, (a.y + b.y) * 0.5)


def _target_from_gaze(face_landmarks) -> Optional[tuple]:
    iris = _iris_center(face_landmarks)
    if iris is None or _gaze_center is None:
        return None
    gx, gy = iris
    cx, cy = _gaze_center
    dx = (gx - cx) * GAZE_GAIN_X
    dy = (gy - cy) * GAZE_GAIN_Y
    mid_x = (_virt_x0 + _virt_x1) * 0.5
    mid_y = (_virt_y0 + _virt_y1) * 0.5
    tx = mid_x + dx * _virt_w
    ty = mid_y + dy * _virt_h
    return tx, ty


def _detect_brow_click(blendshapes, now: float) -> bool:
    global _last_brow_click_at
    if not blendshapes:
        return False
    v = 0.0
    for b in blendshapes:
        if b.category_name in ("browInnerUp", "browOuterUpLeft",
                               "browOuterUpRight"):
            v = max(v, float(b.score))
    if v > BROW_CLICK_THRESHOLD:
        if now - _last_brow_click_at > BROW_CLICK_COOLDOWN_S:
            _last_brow_click_at = now
            return True
    return False


def _detect_wink_click(blendshapes, now: float) -> bool:
    """Edge-triggered wink: one eye closed, the other open. Accepts either
    eye so the user doesn't have to think about which to use."""
    global _last_wink_click_at, _wink_armed
    if not blendshapes:
        return False
    left = right = 0.0
    for b in blendshapes:
        if b.category_name == "eyeBlinkLeft":
            left = float(b.score)
        elif b.category_name == "eyeBlinkRight":
            right = float(b.score)
    hi, lo = max(left, right), min(left, right)
    winking = hi > WINK_CLOSED_THRESHOLD and lo < WINK_OPEN_THRESHOLD
    fired = False
    if winking and _wink_armed:
        if now - _last_wink_click_at > WINK_COOLDOWN_S:
            _last_wink_click_at = now
            _wink_armed = False
            fired = True
    elif not winking and lo < WINK_OPEN_THRESHOLD:
        # Asymmetry gone (the formerly-open eye is still open enough) → rearm.
        # Using `lo` here means we don't need BOTH eyes fully open, just the
        # steady-open one — friendlier for frequent blinkers.
        _wink_armed = True
    return fired


def _detect_right_wink_click(blendshapes, now: float) -> bool:
    """Edge-triggered right-eye-only wink."""
    global _last_right_wink_click_at, _right_wink_armed
    if not blendshapes:
        return False
    left = right = 0.0
    for b in blendshapes:
        if b.category_name == "eyeBlinkLeft":
            left = float(b.score)
        elif b.category_name == "eyeBlinkRight":
            right = float(b.score)
    winking = right > WINK_CLOSED_THRESHOLD and left < WINK_OPEN_THRESHOLD
    fired = False
    if winking and _right_wink_armed:
        if now - _last_right_wink_click_at > WINK_COOLDOWN_S:
            _last_right_wink_click_at = now
            _right_wink_armed = False
            fired = True
    elif not winking and right < WINK_OPEN_THRESHOLD:
        _right_wink_armed = True
    return fired


def _detect_blink_click(blendshapes, now: float) -> bool:
    """Edge-triggered deliberate blink: BOTH eyes closed hard together."""
    global _last_blink_click_at, _blink_click_armed
    if not blendshapes:
        return False
    left = right = 0.0
    for b in blendshapes:
        if b.category_name == "eyeBlinkLeft":
            left = float(b.score)
        elif b.category_name == "eyeBlinkRight":
            right = float(b.score)
    both_closed = left > BLINK_CLICK_THRESHOLD and right > BLINK_CLICK_THRESHOLD
    fired = False
    if both_closed and _blink_click_armed:
        if now - _last_blink_click_at > BLINK_CLICK_COOLDOWN_S:
            _last_blink_click_at = now
            _blink_click_armed = False
            fired = True
    elif left < BLINK_CLICK_OPEN_THR and right < BLINK_CLICK_OPEN_THR:
        _blink_click_armed = True
    return fired


def _detect_mouth_click(blendshapes, now: float) -> bool:
    """Edge-triggered mouth-open click (jawOpen blendshape)."""
    global _last_mouth_click_at, _mouth_armed
    if not blendshapes:
        return False
    jaw = 0.0
    for b in blendshapes:
        if b.category_name == "jawOpen":
            jaw = float(b.score)
            break
    fired = False
    if jaw > MOUTH_CLICK_THRESHOLD and _mouth_armed:
        if now - _last_mouth_click_at > MOUTH_CLICK_COOLDOWN_S:
            _last_mouth_click_at = now
            _mouth_armed = False
            fired = True
    elif jaw < MOUTH_CLICK_OPEN_THR:
        _mouth_armed = True
    return fired


def _detect_smile_rightclick(blendshapes, now: float) -> bool:
    global _last_smile_click_at, _smile_click_armed
    if not blendshapes:
        return False
    v = 0.0
    for b in blendshapes:
        if b.category_name in ("mouthSmileLeft", "mouthSmileRight"):
            v = max(v, float(b.score))
    fired = False
    if v > SMILE_CLICK_THRESHOLD and _smile_click_armed:
        if now - _last_smile_click_at > SMILE_CLICK_COOLDOWN_S:
            _last_smile_click_at = now
            _smile_click_armed = False
            fired = True
    elif v < SMILE_CLICK_OPEN_THR:
        _smile_click_armed = True
    return fired


def _detect_pucker_rightclick(blendshapes, now: float) -> bool:
    global _last_pucker_click_at, _pucker_click_armed
    if not blendshapes:
        return False
    v = 0.0
    for b in blendshapes:
        if b.category_name == "mouthPucker":
            v = float(b.score)
            break
    fired = False
    if v > PUCKER_CLICK_THRESHOLD and _pucker_click_armed:
        if now - _last_pucker_click_at > PUCKER_CLICK_COOLDOWN_S:
            _last_pucker_click_at = now
            _pucker_click_armed = False
            fired = True
    elif v < PUCKER_CLICK_OPEN_THR:
        _pucker_click_armed = True
    return fired


def _detect_furrow_rightclick(blendshapes, now: float) -> bool:
    global _last_furrow_click_at, _furrow_click_armed
    if not blendshapes:
        return False
    v = 0.0
    for b in blendshapes:
        if b.category_name in ("browDownLeft", "browDownRight"):
            v = max(v, float(b.score))
    fired = False
    if v > FURROW_CLICK_THRESHOLD and _furrow_click_armed:
        if now - _last_furrow_click_at > FURROW_CLICK_COOLDOWN_S:
            _last_furrow_click_at = now
            _furrow_click_armed = False
            fired = True
    elif v < FURROW_CLICK_OPEN_THR:
        _furrow_click_armed = True
    return fired


def _detect_right_click(blendshapes, now: float) -> bool:
    if _right_click_method == "smile":
        return _detect_smile_rightclick(blendshapes, now)
    if _right_click_method == "pucker":
        return _detect_pucker_rightclick(blendshapes, now)
    if _right_click_method == "furrow":
        return _detect_furrow_rightclick(blendshapes, now)
    return False


def _detect_pinch_click(hands_lm_list, now: float) -> bool:
    """Edge-triggered pinch: thumb tip (4) to index tip (8) closes."""
    global _pinch_was_closed, _last_pinch_click_at
    hand = _pick_right_hand(hands_lm_list)
    if hand is None:
        _pinch_was_closed = False
        return False
    dx = hand[4].x - hand[8].x
    dy = hand[4].y - hand[8].y
    d = math.hypot(dx, dy)
    fired = False
    if not _pinch_was_closed and d < PINCH_CLOSE_THRESHOLD:
        _pinch_was_closed = True
        if now - _last_pinch_click_at > PINCH_COOLDOWN_S:
            _last_pinch_click_at = now
            fired = True
    elif _pinch_was_closed and d > PINCH_OPEN_THRESHOLD:
        _pinch_was_closed = False
    return fired


def _update_cursor(face_matrix, face_landmarks, hands_lm_list,
                   blendshapes, now: float, hands_up_toggle: bool) -> str:
    """Move the cursor per `_pointing_method`. Return 'primary', 'right',
    or '' for the click event type this frame."""
    global _cursor_enabled, _cursor_calibrated, _cursor_calib_start
    global _yaw_center, _pitch_center, _cur_x, _cur_y
    global _finger_center, _gaze_center

    if hands_up_toggle:
        _cursor_enabled = not _cursor_enabled
        print(f"[viewer] cursor {'ON' if _cursor_enabled else 'OFF'} "
              f"({_pointing_method}/{_click_method})", flush=True)
        if _cursor_enabled:
            _cursor_calibrated = False
            _cursor_calib_start = now
            _finger_center = None
            _gaze_center = None

    # ---- calibration: capture neutral pose per method ---------------------
    if _cursor_enabled and not _cursor_calibrated:
        if _cursor_calib_start is None:
            return ""
        if now - _cursor_calib_start < CURSOR_CALIB_S:
            return ""
        ok = False
        if _pointing_method == "head":
            if face_matrix is not None:
                y, p = _matrix_to_yaw_pitch(face_matrix)
                _yaw_center, _pitch_center = y, p
                ok = True
        elif _pointing_method == "finger":
            hand = _pick_right_hand(hands_lm_list)
            if hand is not None:
                _finger_center = (hand[8].x, hand[8].y)
                ok = True
        elif _pointing_method == "gaze":
            iris = _iris_center(face_landmarks)
            if iris is not None:
                _gaze_center = iris
                ok = True
        if ok:
            _cursor_calibrated = True
            print(f"[viewer] cursor calibrated ({_pointing_method})",
                  flush=True)
        else:
            # missing data — keep trying each frame
            _cursor_calib_start = now
        return ""

    if not _cursor_enabled or not _cursor_calibrated:
        return ""

    # ---- pointing ---------------------------------------------------------
    # Freeze the cursor while two-hand scroll is engaged so the finger
    # pointer doesn't chase the scrolling hand.
    if _scroll_active:
        return ""
    target = None
    if _pointing_method == "head":
        target = _target_from_head(face_matrix)
    elif _pointing_method == "finger":
        target = _target_from_finger(hands_lm_list)
    elif _pointing_method == "gaze":
        target = _target_from_gaze(face_landmarks)

    if target is not None:
        tx = max(_virt_x0 + 2.0, min(_virt_x1 - 2.0, target[0]))
        ty = max(_virt_y0 + 2.0, min(_virt_y1 - 2.0, target[1]))
        _cur_x += (tx - _cur_x) * CURSOR_SMOOTHING
        _cur_y += (ty - _cur_y) * CURSOR_SMOOTHING
        try:
            pyautogui.moveTo(_cur_x, _cur_y, _pause=False)
        except pyautogui.FailSafeException:
            _cursor_enabled = False
            print("[viewer] failsafe → cursor OFF", flush=True)
            return ""

    # ---- click ------------------------------------------------------------
    primary = False
    if _click_method == "brow":
        primary = _detect_brow_click(blendshapes, now)
    elif _click_method == "pinch":
        primary = _detect_pinch_click(hands_lm_list, now)
    elif _click_method == "wink":
        primary = _detect_wink_click(blendshapes, now)
    elif _click_method == "right_wink":
        primary = _detect_right_wink_click(blendshapes, now)
    elif _click_method == "blink":
        primary = _detect_blink_click(blendshapes, now)
    elif _click_method == "mouth":
        primary = _detect_mouth_click(blendshapes, now)
    right = _detect_right_click(blendshapes, now)
    if right:
        return "right"
    if primary:
        return "primary"
    return ""


def _update_smile(blendshapes) -> None:
    global _smile_val
    if not blendshapes:
        return
    l = r = 0.0
    for b in blendshapes:
        if b.category_name == "mouthSmileLeft":
            l = float(b.score)
        elif b.category_name == "mouthSmileRight":
            r = float(b.score)
    v = max(l, r)
    _smile_val = (1 - SMILE_EMA_ALPHA) * _smile_val + SMILE_EMA_ALPHA * v


def _capture_loop() -> None:
    global _latest_jpeg, _bob_pending, _blink_pending
    global _prayer_start_pending, _prayer_end_pending, _boot_pending
    global _swipe_pending, _left_hand_y, _right_hand_y
    global _left_hand_x, _right_hand_x
    cam = Camera()
    for _ in range(30):
        ok, _ = cam.read()
        if ok:
            break
        time.sleep(0.1)
    print(f"[viewer] camera warm: {cam.name}", flush=True)

    face_landmarker = None
    if FACE_MODEL_PATH.exists():
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(FACE_MODEL_PATH)),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        face_landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        print("[viewer] face landmarker loaded", flush=True)

    hand_landmarker = None
    if _ensure_model(HAND_MODEL_PATH, HAND_MODEL_URL):
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(HAND_MODEL_PATH)),
            num_hands=2,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        hand_landmarker = mp_vision.HandLandmarker.create_from_options(opts)
        print("[viewer] hand landmarker loaded", flush=True)

    t0 = time.time()
    while not _stop.is_set():
        ok, frame = cam.read()
        if not ok:
            time.sleep(0.01)
            continue
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.time() - t0) * 1000)

        face_nose = None
        face_blendshapes = None
        face_matrix = None
        if face_landmarker is not None:
            try:
                fres = face_landmarker.detect_for_video(mp_img, ts_ms)
                if fres.face_landmarks:
                    lm = fres.face_landmarks[0]
                    _draw_face_landmarks(frame, lm)
                    face_nose = (lm[1].x, lm[1].y)  # nose tip
                if fres.face_blendshapes:
                    face_blendshapes = fres.face_blendshapes[0]
                    _update_smile(face_blendshapes)
                else:
                    _update_smile(None)
                if fres.facial_transformation_matrixes:
                    face_matrix = fres.facial_transformation_matrixes[0]
            except Exception as e:
                print(f"[viewer] face err: {e}", flush=True)

        hand_wrists = []
        hands_lm_list = []
        if hand_landmarker is not None:
            try:
                hres = hand_landmarker.detect_for_video(mp_img, ts_ms)
                for hand in (hres.hand_landmarks or []):
                    _draw_hand(frame, hand)
                    hand_wrists.append((hand[0].x, hand[0].y))
                    hands_lm_list.append(hand)
            except Exception as e:
                print(f"[viewer] hand err: {e}", flush=True)

        _update_motion(face_nose, hand_wrists)

        now = time.time()
        nose_y_only = face_nose[1] if face_nose is not None else None
        bobbed = _detect_bob(nose_y_only, now)
        blinked = _detect_blink(face_blendshapes, now)
        prayer_change = _update_prayer(hands_lm_list, now)
        booted = _update_double_clap(hands_lm_list, now)
        swipe_dir = _update_swipe(hands_lm_list, now)
        left_x, left_y, right_x, right_y = _split_hands_xy(hands_lm_list)

        hands_up_toggle = _update_hands_up(hands_lm_list, now)
        scroll_amt = _update_two_hand_scroll(hands_lm_list, now)
        if scroll_amt != 0:
            try:
                pyautogui.scroll(scroll_amt, _pause=False)
            except Exception as e:
                print(f"[viewer] scroll failed: {e}", flush=True)
        face_lms = None
        if face_landmarker is not None:
            try:
                if fres.face_landmarks:
                    face_lms = fres.face_landmarks[0]
            except Exception:
                face_lms = None
        click_event = _update_cursor(
            face_matrix, face_lms, hands_lm_list,
            face_blendshapes, now, hands_up_toggle,
        )
        if _cursor_enabled and _cursor_calibrated:
            _dispatch_click(click_event, now)

        if swipe_dir is not None and not _jam_mode:
            _fire_swipe_action(swipe_dir)
        if prayer_change is True:
            print("[viewer] prayer START → holding wispr key", flush=True)
            _hold_wispr_key()
        elif prayer_change is False:
            print("[viewer] prayer END → releasing wispr key", flush=True)
            _release_wispr_key()
        if booted:
            print("[viewer] double clap → boot", flush=True)

        with _state_lock:
            if bobbed:
                _bob_pending = True
            if blinked:
                _blink_pending = True
            if prayer_change is True:
                _prayer_start_pending = True
            elif prayer_change is False:
                _prayer_end_pending = True
            if booted:
                _boot_pending = True
            if swipe_dir is not None:
                _swipe_pending = swipe_dir
            _left_hand_y = left_y
            _right_hand_y = right_y
            _left_hand_x = left_x
            _right_hand_x = right_x

        okj, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if okj:
            with _jpeg_lock:
                _latest_jpeg = buf.tobytes()

    cam.release()
    print("[viewer] capture loop stopped", flush=True)


# --- http handler ---------------------------------------------------------
# A few common misspellings / homophones. Everything else is passed to
# `open -a` as-is and macOS fuzzy-matches app names.
APP_ALIASES = {
    "ark": "Arc",                  # "arc" often transcribed as "ark"
    "wispr": "Wispr Flow",
    "whisper": "Wispr Flow",
    "wisper": "Wispr Flow",
    "chat gpt": "ChatGPT",
    "vs code": "Visual Studio Code",
    "zoom": "zoom.us",
}


def _scroll(direction: str) -> None:
    try:
        amount = 10 if direction == "up" else -10
        pyautogui.scroll(amount, _pause=False)
        print(f"[viewer] scroll {direction}", flush=True)
    except Exception as e:
        print(f"[viewer] scroll failed: {e}", flush=True)


def _dispatch_click(event: str, now: float) -> None:
    """Turn detector events into pyautogui clicks, honoring double-click mode.

    event: 'primary' | 'right' | ''
    When `_double_click_on`, two primaries within DOUBLE_WINDOW_S collapse
    into one native double-click. Otherwise primary fires immediately.
    """
    global _pending_click_at, _pending_click_waiting
    # Right click bypasses the double-click delay.
    if event == "right":
        try:
            pyautogui.rightClick(_pause=False)
            print(f"[viewer] {_right_click_method} → right-click", flush=True)
        except pyautogui.FailSafeException:
            pass
        return
    if event == "primary":
        if _double_click_on:
            if _pending_click_waiting and (now - _pending_click_at) < DOUBLE_WINDOW_S:
                _pending_click_waiting = False
                try:
                    pyautogui.doubleClick(_pause=False)
                    print(f"[viewer] {_click_method} → double-click", flush=True)
                except pyautogui.FailSafeException:
                    pass
            else:
                _pending_click_waiting = True
                _pending_click_at = now
        else:
            try:
                pyautogui.click(_pause=False)
                print(f"[viewer] {_click_method} → click", flush=True)
            except pyautogui.FailSafeException:
                pass
    # If a single is pending and its double window expired, fire it now.
    if _pending_click_waiting and (now - _pending_click_at) >= DOUBLE_WINDOW_S:
        _pending_click_waiting = False
        try:
            pyautogui.click(_pause=False)
            print(f"[viewer] {_click_method} → click (delayed)", flush=True)
        except pyautogui.FailSafeException:
            pass


def _click() -> None:
    try:
        pyautogui.click(_pause=False)
        print("[viewer] click", flush=True)
    except Exception as e:
        print(f"[viewer] click failed: {e}", flush=True)


def _volume(direction: str) -> None:
    if direction == "mute":
        script = ('set volume output muted '
                  '(not (output muted of (get volume settings)))')
    elif direction == "up":
        script = ('set volume output volume '
                  '((output volume of (get volume settings)) + 10)')
    else:
        script = ('set volume output volume '
                  '((output volume of (get volume settings)) - 10)')
    try:
        subprocess.Popen(["osascript", "-e", script])
        print(f"[viewer] volume {direction}", flush=True)
    except Exception as e:
        print(f"[viewer] volume failed: {e}", flush=True)


def _resolve_app(target: str) -> str:
    t = target.strip().lower()
    return APP_ALIASES.get(t, target.strip())


def _open_app(target: str) -> tuple:
    name = _resolve_app(target)
    try:
        subprocess.Popen(["open", "-a", name])
        print(f"[viewer] open -a {name!r}", flush=True)
        return True, name
    except Exception as e:
        print(f"[viewer] open failed: {e}", flush=True)
        return False, name


def _quit_app(target: str) -> tuple:
    name = _resolve_app(target)
    try:
        subprocess.Popen(
            ["osascript", "-e", f'tell application "{name}" to quit']
        )
        print(f"[viewer] quit {name!r}", flush=True)
        return True, name
    except Exception as e:
        print(f"[viewer] quit failed: {e}", flush=True)
        return False, name


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a, **k) -> None:
        pass

    def _write_status(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        global _wispr_method, _scroll_sens
        global _right_click_method, _double_click_on
        if self.path == "/command":
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            try:
                data = json.loads(raw)
            except Exception:
                self._write_status(400, "text/plain", b"bad json")
                return
            action = (data.get("action") or "").lower()
            target = data.get("target") or ""
            if action == "open" and target:
                ok, name = _open_app(target)
                self._write_status(
                    200 if ok else 500,
                    "application/json",
                    json.dumps({"ok": ok, "resolved": name}).encode(),
                )
                return
            if action == "close" and target:
                ok, name = _quit_app(target)
                self._write_status(
                    200 if ok else 500,
                    "application/json",
                    json.dumps({"ok": ok, "resolved": name}).encode(),
                )
                return
            if action == "scroll":
                _scroll(target)
                self._write_status(200, "application/json", b'{"ok":true}')
                return
            if action == "click":
                _click()
                self._write_status(200, "application/json", b'{"ok":true}')
                return
            if action == "volume":
                _volume(target)
                self._write_status(200, "application/json", b'{"ok":true}')
                return
            if action == "desktop":
                _fire_swipe_action("right" if target == "next" else "left")
                self._write_status(200, "application/json", b'{"ok":true}')
                return
            if action == "jam":
                global _jam_mode
                _jam_mode = bool(data.get("on"))
                print(f"[viewer] jam mode = {_jam_mode}", flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({"ok": True, "jam": _jam_mode}).encode(),
                )
                return
            if action == "cursor_enable":
                global _cursor_enabled, _cursor_calibrated, _cursor_calib_start
                global _finger_center, _gaze_center
                want = bool(data.get("on"))
                _cursor_enabled = want
                if want:
                    _cursor_calibrated = False
                    _cursor_calib_start = time.time()
                    _finger_center = None
                    _gaze_center = None
                print(f"[viewer] cursor {'ON' if want else 'OFF'} "
                      f"(via button)", flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({"ok": True, "cursor": _cursor_enabled}).encode(),
                )
                return
            if action == "cursor_method":
                global _pointing_method, _click_method
                pointing = data.get("pointing")
                click = data.get("click")
                changed = False
                if pointing in ("head", "finger", "gaze"):
                    if pointing != _pointing_method:
                        _pointing_method = pointing
                        changed = True
                if click in ("wink", "brow", "pinch", "blink",
                             "right_wink", "mouth"):
                    _click_method = click
                if changed:
                    # Swapping pointing method ⇒ re-calibrate on next frame.
                    _cursor_calibrated = False
                    _cursor_calib_start = time.time()
                    _finger_center = None
                    _gaze_center = None
                print(f"[viewer] cursor method = {_pointing_method}/"
                      f"{_click_method}", flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({
                        "ok": True,
                        "pointing": _pointing_method,
                        "click": _click_method,
                        "wispr": _wispr_method,
                        "scrollSens": _scroll_sens,
                        "rightClick": _right_click_method,
                        "doubleClick": _double_click_on,
                    }).encode(),
                )
                return
            if action == "wispr_method":
                method = data.get("method")
                if method in ("applescript_fn", "cgevent_f19",
                              "cgevent_fn", "all", "apple_dictation", "off"):
                    _wispr_method = method
                print(f"[viewer] wispr method = {_wispr_method}", flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({"ok": True, "wispr": _wispr_method}).encode(),
                )
                return
            if action == "right_click_method":
                method = data.get("method")
                if method in ("smile", "pucker", "furrow", "off"):
                    _right_click_method = method
                print(f"[viewer] right-click = {_right_click_method}", flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({"ok": True, "rightClick": _right_click_method}).encode(),
                )
                return
            if action == "double_click":
                _double_click_on = bool(data.get("on"))
                print(f"[viewer] double-click = {_double_click_on}", flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({"ok": True, "doubleClick": _double_click_on}).encode(),
                )
                return
            if action == "scroll_sens":
                sens = data.get("sens")
                if sens in SCROLL_GAIN_MAP:
                    _scroll_sens = sens
                print(f"[viewer] scroll sens = {_scroll_sens}", flush=True)
                self._write_status(
                    200, "application/json",
                    json.dumps({"ok": True, "scrollSens": _scroll_sens}).encode(),
                )
                return
            if action == "wispr_test":
                _tap_wispr_hotkey()
                self._write_status(
                    200, "application/json",
                    json.dumps({"ok": True, "fired": _wispr_method}).encode(),
                )
                return
            self._write_status(400, "text/plain", b"unknown action")
            return
        self._write_status(404, "text/plain", b"not found")

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            self._write_status(200, "text/html; charset=utf-8",
                               HTML.encode("utf-8"))
            return

        if self.path == "/stream":
            self.send_response(200)
            self.send_header(
                "Content-Type",
                f"multipart/x-mixed-replace; boundary={BOUNDARY}",
            )
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while not _stop.is_set():
                    with _jpeg_lock:
                        jpg = _latest_jpeg
                    if jpg is None:
                        time.sleep(0.03)
                        continue
                    self.wfile.write(f"--{BOUNDARY}\r\n".encode())
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(jpg)}\r\n\r\n".encode()
                    )
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                    time.sleep(1 / 30.0)
            except (BrokenPipeError, ConnectionResetError):
                return
            return

        if self.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            try:
                while not _stop.is_set():
                    global _bob_pending, _blink_pending
                    global _prayer_start_pending, _prayer_end_pending
                    global _boot_pending, _swipe_pending, _dictation_pending
                    with _state_lock:
                        bob = _bob_pending
                        blink = _blink_pending
                        prayer_start = _prayer_start_pending
                        prayer_end = _prayer_end_pending
                        boot = _boot_pending
                        swipe = _swipe_pending
                        dictation = _dictation_pending
                        _bob_pending = False
                        _blink_pending = False
                        _prayer_start_pending = False
                        _prayer_end_pending = False
                        _boot_pending = False
                        _swipe_pending = None
                        _dictation_pending = False
                        smile = _smile_val
                        l_y = _left_hand_y
                        r_y = _right_hand_y
                        l_x = _left_hand_x
                        r_x = _right_hand_x
                        prayer_live = _prayer_active
                        jam_on = _jam_mode
                        cursor_on = _cursor_enabled
                        cursor_calibrated = _cursor_calibrated
                    payload = {
                        "smile": round(smile, 3),
                        "leftHandY":  None if l_y is None else round(l_y, 4),
                        "rightHandY": None if r_y is None else round(r_y, 4),
                        "leftHandX":  None if l_x is None else round(l_x, 4),
                        "rightHandX": None if r_x is None else round(r_x, 4),
                        "prayerLive": prayer_live,
                        "cursor": cursor_on,
                        "cursorCalibrated": cursor_calibrated,
                        "jam": jam_on,
                        "pointing": _pointing_method,
                        "click": _click_method,
                        "wispr": _wispr_method,
                        "scrollSens": _scroll_sens,
                        "rightClick": _right_click_method,
                        "doubleClick": _double_click_on,
                    }
                    if bob: payload["bob"] = True
                    if blink: payload["blink"] = True
                    if prayer_start: payload["prayerStart"] = True
                    if prayer_end: payload["prayerEnd"] = True
                    if boot: payload["boot"] = True
                    if swipe: payload["swipe"] = swipe
                    if dictation: payload["dictation"] = True
                    line = f"data: {json.dumps(payload)}\n\n"
                    try:
                        self.wfile.write(line.encode("utf-8"))
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    time.sleep(1 / 25.0)
            except Exception as e:
                print(f"[viewer] events err: {e}", flush=True)
            return

        self._write_status(404, "text/plain", b"not found")


class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    t = threading.Thread(target=_capture_loop, daemon=True)
    t.start()

    srv = ThreadingServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}/"
    print(f"[viewer] wispr hotkey = key code {WISPR_KEYCODE} "
          f"(fn flag={'on' if WISPR_USE_FN_FLAG else 'off'}). "
          "bind Wispr Flow to this key for reliable prayer-hold.", flush=True)
    print(f"[viewer] serving {url} — opening in your browser", flush=True)
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[viewer] shutting down", flush=True)
    finally:
        _stop.set()
        srv.server_close()


if __name__ == "__main__":
    main()
