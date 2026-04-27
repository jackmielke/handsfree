# handsfree

A camera-driven, voice-aware gesture OS for macOS. Move the cursor with your
finger, click with a wink, scroll by squinting + waving your hand, copy with
a head nod, dictate with prayer hands, talk to your computer like Jarvis with
"Hey Wonder" — all without touching the trackpad.

Built for RSI relief, and because it's fun.

> Started as "what if I could browse the web through facial expressions."
> Became something the author actually uses every day with sore wrists.

---

## 🚀 Quickstart

```bash
git clone https://github.com/jackmielke/handsfree.git
cd handsfree
./setup.sh                  # creates .venv, installs deps
.venv/bin/python browser_viewer.py
```

Open <http://localhost:8765> — the **Control Center** opens automatically.
First launch downloads the MediaPipe face + hand landmarker models (~10 MB).

The pages you'll use:

| URL | What it is |
|---|---|
| [`/`](http://localhost:8765/) | Control Center — pointing/click method, scroll, voice, experiments |
| [`/vision`](http://localhost:8765/vision) | 🔬 Vision Lab — every blendshape, gesture, and voice transcript live |
| [`/boxing`](http://localhost:8765/boxing) | 🥊 Muay Thai mode — punch a target, take counter-punches |

### Permissions

macOS will prompt for **Camera**, **Microphone**, **Accessibility**, and
**Input Monitoring** the first time. Grant all of them to the Terminal app
you launched from — TCC permissions follow the parent process, not Python
itself. There's also a `Wonder.app` bundle in the project that exposes the
right Info.plist usage descriptions; opening it once gives you a single place
to grant Speech Recognition for the Apple Speech engine.

---

## 👀 What you can do

### 🖱 Cursor (pointing)

| Method | How |
|---|---|
| **finger** | point with your right index — wrist to fingertip vector drives the cursor (default) |
| **head**   | head pose (yaw + pitch); subtle and good once calibrated |
| **gaze**   | eye iris position; experimental |

Cursor traverses **all monitors** correctly via `CGWarpMouseCursorPosition`,
and re-probes the virtual desktop bounds every 3 seconds so newly-plugged-in
displays become reachable without restarting.

### 👇 Click

Pick one in Control Center → **click**:

- **brow raise** — natural; works in meetings
- **😊 smile** — hard smile to click; also natural on camera
- **mouth open** — clear, deliberate, surprisingly comfortable
- **pinch** — thumb + index touch
- **blink** / **either wink** / **right wink** — for if you have the eye control

**Long-press hybrid** is on by default for **brow / smile / mouth / pinch**:
*short hold* = single click, *hold ≥ 0.4s* = mouseDown drag, release =
mouseUp. So a quick eyebrow flick is a click; raising and holding your brow
is a drag. Same gesture, two outcomes — like a long-press on mobile. Wink
and blink stay edge-triggered (you can't really "hold" them).

Toggle via the **⏳ click long-press hybrid** experiment tile (default ON;
turn off to make every gesture just a click, no drag).

### ✊ Press-and-hold options

- **whatever your click method is** — the hybrid above gives you
  press-and-hold for free on brow / smile / mouth / pinch
- **✌️ peace** — hold a peace sign to keep mouseDown; drop it to release. Off by default.
- **👌 OK-sign** — flash 👌 to *toggle* mouseDown on, again to release. Off by default.

### 🖱 Scroll

Pick a mode in Control Center → **scroll mode** (default = squint+hand):

| Mode | Gate | Input | Notes |
|---|---|---|---|
| **squint+hand** | squint either eye > 50% | hand wrist Y delta | RSI-friendly default |
| **squint+head** | same | head pitch | hands-free |
| **fist** | close one hand | wrist Y/X | classic; can be tiring on sore wrists |
| **two hands** | raise both | midpoint Y | needs both hands free |
| **head+lefthand** | left hand up | head pitch | |
| **head+mouth** | mouth open | head pitch | |
| **brow** | always-on | brow up = scroll up, furrow = down | |

Scroll speed is a continuous slider — 0.5× to 6×. Default 3×.

### ✋ Macros and shortcuts

- **Two-hand swipe L↔R** → switch macOS Spaces (`Ctrl + ←/→`)
- **Two-hand swipe alt** → next/prev browser tab (toggle in Control Center)
- **Hands-overhead** → master toggle (auto-mutes audio when off)
- **🤚 T-pose with both hands** → master toggle (alternative)
- **Double-clap** → toggle voice command listener

### 🎤 Voice — "Hey Wonder"

Voice is **always-on standby** by default. The engine auto-starts when the
daemon launches and listens for the wake word.

```
🟠 standby   →  say "hey wonder"   →  🟢 commands armed
                                          ↓
              speak any phrase from the dictionary
                                          ↓
🟠 standby   ←  say "bye wonder" / "later wonder"
```

Sounds + macOS toasts confirm every state change so you never have to wonder
whether it heard you:

| Trigger | Sound | Toast |
|---|---|---|
| arm | Bottle (hollow knock) | armed 🟢 — go |
| disarm | Submarine (low blip) | later 👋 — standby 🟠 |
| arm-but-already-armed | Pop | I'm already here 🟢 |
| disarm-but-already-off | Tink | already on standby 🟠 |

**Three engines, swap live** in the Vision Lab → 🎤 voice panel:
- **vosk** — Kaldi grammar-locked. Fastest, instant partials, default. ~40 MB model.
- **whisper** — `faster-whisper tiny.en` with VAD-segmenting. Higher quality, ~200 ms / utterance.
- **apple speech** — on-device Apple Speech (the engine Siri uses). Highest quality but requires Speech Recognition permission via `Wonder.app`.

**Auto-pause when Wispr Flow is dictating** — voice2 silently goes to
standby while prayer-hands has Wispr listening, so commands can't fire on
words you're dictating into Slack/Notion.

### 📖 Voice command dictionary (live-editable)

Default global commands (matched fuzzy via `difflib` — typos like "pen
figma" still snap to "open figma"):

```
open arc          open chrome      open terminal     open figma
open finder       open slack       open telegram     open notes
open cursor       open notion
switch tab left   switch tab right next tab          previous tab
next desktop      previous desktop
copy              paste            select all        undo / redo
go back           go forward       new tab           close tab
scroll up         scroll down      click             dictate
mute              unmute
```

**Per-app contextual overlays** kick in when an app is frontmost and add
extra commands on top:

- **Arc**: command bar / open url / new tab / little arc / close tab / reopen tab / pin tab / library / toggle sidebar / find in page / next space / previous space / developer tools / incognito / private / reload / hard reload
- **Figma**: frame, rectangle, ellipse, text, line, pen, comment, fit, fit screen, zoom to selection, actual size, group, ungroup, duplicate
- **Slack**: search, jump to, next channel, previous channel, next unread, previous unread, thread, react
- **Cursor**: command palette, find, find in files, go to file, toggle sidebar, toggle terminal, ai chat, comment

**Add your own commands live** — Vision Lab → 📖 command dictionary panel.
Type a phrase + an action like `url:https://app.posthog.com` and click ADD.
Saved to `~/Library/Application Support/handsfree/user_commands.json`,
survives restarts. Action types:

| Action | Effect |
|---|---|
| `hotkey:cmd+t` | fires that keyboard shortcut |
| `open_app:Arc` | launches the app |
| `url:https://...` | opens URL in default browser |
| `scroll:up` / `scroll:down` | one click of scroll |
| `click` | primary click at cursor |
| `wispr_menu` | clicks Wispr Flow menu icon (auto-disarms voice2) |
| `system_mute` / `system_unmute` | macOS volume mute |

### 🎯 Wispr Flow integration

Wispr listens for the Fn key at the IOKit HID layer, which synthetic Fn
events can't reach. Easiest reliable trigger: **menu-bar click via
AppleScript**. Default preset: **🌟 pray → click icon (recommended)**. The
script finds the Wispr menu-bar item by its `description = "status menu"`
attribute so it works regardless of where the icon sits.

There's also a 🧪 **wispr test** row in Control Center: a "fire menu click"
button to test toggle-on/off without the prayer gesture, plus a "probe menu"
button that dumps Wispr's menu-bar layout for debugging.

### 🔬 Vision Lab

[`/vision`](http://localhost:8765/vision) is the X-ray view — everything the
system perceives, live, at 10 Hz:

- **🎭 face blendshapes** — 28 of MediaPipe's 52, as live bars (jawOpen,
  smile, brow, eye blink/squint/wide, mouth pucker/funnel/roll, cheek puff,
  nose sneer, tongueOut, …)
- **🙆 head pose** — yaw + pitch crosshair pad (great for diagnosing jitter)
- **✋ hands** — pose label per hand (✊ ✌️ 👍 👌 🖐), depth scale, wrist
  position, four-finger extended/curled grid
- **⚡ named gestures** — peace / fist / thumbs / OK / prayer / atelier as
  on/off chips
- **📋 recent events** — last 24 fired actions with timestamps
- **🎤 voice** — engine status, live partial, recent transcripts
- **🪟 active app overlay** — current frontmost app + which contextual
  command pack is active
- **📖 dictionary** — full live list with per-row delete + add-row inputs

Useful for designing new actions or diagnosing why something fired.

### ✨ Atelier mode (Figma magic)

Hold an **A-pose** (fingertips together, elbows wide) → ✨ atelier mode
toggles. Off by default — opt in via the experiments tile. While active:

- **two-hand spread** — pan
- **head dolly** (lean closer/farther) — zoom
- **🤛 fist** — push toward camera = zoom in, pull = zoom out
- **pinch grab** — thumb + index touch = mouseDown drag

A-pose again to exit.

### 🥊 Boxing mode

Open [`/boxing`](http://localhost:8765/boxing). Punch / cross / hook /
uppercut classified via wrist velocity. Comic-book FX, screen shake,
McGregor health bar, 3-minute round timer with bell, McGregor counter-
punches you have to dodge. It is, somehow, a serious cardio workout.

### 🎵 Jam mode

Click **jam mode** (top bar) → camera UI hides, audio unlocks. Both hands
become theremins (left=bass, right=phased lead). Smile → Rhodes pad. Head
bob → drums. DJ Board lets you toggle channels, swap presets, and ride a
master volume slider. Auto-mutes when jam mode is off — silence on the
plain Control Center page.

### 🥋 Other experiments

Top bar → **experiments** row. Each tile is a toggle.

| Tile | What | Default |
|---|---|---|
| **T ✋ timeout** | T-pose with both hands toggles master on/off | ON |
| **⏳ click long-press hybrid** | brow / smile / mouth / pinch — short hold = click, long hold = drag | ON |
| **✌️ hold-to-drag** | peace-sign mouseDown | off |
| **👌 drag lock** | OK-sign toggles mouseDown | off |
| **🙆 head-up copy** | chin-lift fires Cmd+C | off |
| **👍 paste** | thumbs-up = Cmd+V | off |
| **👄 mouth paste** | open mouth briefly = Cmd+V | off |
| **🤛 fist depth zoom** | fist + push/pull camera = Cmd+= / Cmd+- (atelier only) | ON |
| **✨ atelier** | A-pose → magic Figma mode | off |

---

## 🗺 Roadmap

What's already built is above ⬆. Here's what's next, roughly in priority order:

### Near term (concrete, plausibly soon)
- **Voice → Wonder loop (v0)** — say "hey wonder, question…" → captures
  the next utterance → posts to the local OpenClaw gateway → reply appears
  as a card in the Vision Lab. Text-only first, voice-out later. Same agent
  as your Telegram Wonder, same memory, single source of truth.
- **Hybrid voice recognizer** — when Vosk's grammar-match fails (low
  confidence), silently fall back to whisper for the same utterance. Keeps
  sub-100 ms response on known commands, unlocks free-form for the long tail.
- **Voice-editable settings** — "hey wonder, scroll speed five" / "turn off
  head copy" / "switch engine to whisper". Each setting becomes
  voice-addressable.
- **More app overlays** — Notion, Cursor full set, Spotify, browser tab-by-
  number ("go to tab three"), Granola, Figma's full toolkit.

### Mid term (interesting, more design needed)
- **Voice-out** — `say -v "Samantha (Premium)"` for $0 TTS, ElevenLabs
  streaming TTS for production quality.
- **Custom wake word** — train a tiny on-device model so users can pick
  their own ("Hey Jarvis" / "Hey Aurora" / whatever).
- **Per-app dictionary editing in the UI** — right now overlays are
  hardcoded; expose them as editable like the global dictionary.
- **Dictionary import/export** — share command sets with friends as a
  single JSON file.
- **Gaze-aware arming** — only arm command mode when you're looking at
  the camera, so you don't accidentally arm while talking to someone else.
- **Combo / macro commands** — single phrase fires a chain ("hey wonder,
  morning" → opens Arc + Slack + Notion, mutes Spotify, sets DND).
- **Visual cursor HUD** — small overlay showing what mode you're in and
  what gesture is currently being interpreted.

### Long term (the Iron Man bit)
- **Voice-to-voice Jarvis loop** — full duplex with the Wonder agent,
  ElevenLabs TTS, interruption handling, ambient awareness.
- **Voice-coded settings + commands via agent** — "hey wonder, make peace
  sign do Cmd+Shift+T" → small Claude API call writes the diff and
  hot-reloads. Lets non-coders extend the system.
- **Multi-user wake words** — bundle up handsfree as something a friend
  can install with their own wake word and vocabulary.
- **Mobile companion** — same dictionary, same memory, available from
  your phone while away from the Mac.
- **Plugin model** — third-party gesture / voice action packs.

---

## 🛠 How it's built

- **Python 3.9+** single-file daemon ([`browser_viewer.py`](browser_viewer.py))
  — HTTP server on port 8765 + capture loop in a worker thread.
- **MediaPipe Tasks** for face landmarks + blendshapes (52 of them) and
  hand landmarks (21 per hand).
- **AVFoundation** via pyobjc for camera + audio (so we hit the actual
  built-in webcam, not whatever Continuity Camera is doing).
- **Quartz `CGEvent`** for cursor + keyboard / scroll injection (works
  across all monitors via `CGWarpMouseCursorPosition`).
- **vosk** + **faster-whisper tiny.en** + **Apple `SFSpeechRecognizer`**
  (via pyobjc) for the three voice engines.
- **Web Audio API** for jam mode and atelier sound effects (no asset files).
- **Server-Sent Events** stream all the live state to the browser pages.
- **`afplay`** + macOS system sounds for voice feedback.

Camera runs at 15 fps to leave headroom for MediaPipe + JPEG encoding. Head
pose, hand poses, and named gestures are computed every frame; voice runs
in its own thread.

---

## 🔒 Privacy

- Everything runs on-device. Camera, mic, gestures, vosk + whisper transcription, command matching — all local.
- Apple Speech is on-device too (we set `requiresOnDeviceRecognition = True`).
- The only network traffic is **optional**: the eventual Wonder agent loop will hit the local OpenClaw gateway (also localhost) which itself talks to Anthropic's API on your behalf. ElevenLabs TTS would be the only thing leaving your machine if you opt into it.
- Saved state lives at `~/Library/Application Support/handsfree/user_commands.json` and a Vosk model directory. Easy to inspect / delete.

---

## 🧯 Common issues

| Symptom | Likely cause / fix |
|---|---|
| Vision page "infinite loading" | Browser tab spinner stays on because SSE streams stay open by design — page is rendered, just look for the green ● live pill at the top. Hard-refresh (⌘+Shift+R) if you suspect a stale cache. |
| Cursor stuck on one display | Restart the daemon — usually means Continuity Camera grabbed the lens or the LaunchAgent inherited a stale display config. |
| Mic not enabled / Vosk hangs at "opening mic" | Microphone permission is missing for whatever launched Python. Easiest fix: launch from `Wonder.app` once to grant Microphone + Speech Recognition. |
| Camera doesn't start | Check Privacy & Security → Camera → enable Terminal (or whatever launches the daemon). |
| Wispr Flow won't toggle | Use the 🧪 **fire menu click** button in Control Center to test the AppleScript path. If it returns OK but Wispr doesn't react, Wispr's Accessibility entitlement may be off. |
| Phantom drags / "press-and-hold" feeling | Atelier mode is probably armed (pinch + fist do drag/zoom there). Use the experiments tile to disable, or A-pose to exit. |

---

## License

MIT — do whatever you want, just don't blame me if you blink too much.
