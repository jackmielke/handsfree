# handsfree

A camera-driven gesture OS for macOS. Point with your finger, click by raising
an eyebrow, scroll with both hands, dictate with prayer hands, jam out in
"jam mode" — all without touching the trackpad.

Built for RSI relief and because it's fun.

## What it does

- **Cursor** — finger pointing, head pose, or eye gaze (pick in Control Center)
- **Click** — brow raise, blink, wink, right-wink, mouth open, or pinch
- **Right-click** — hard smile, pucker, or furrowed brow
- **Double-click** — double-tap any primary gesture within 350 ms
- **Scroll** — raise both hands and move them up/down together
- **Dictation** — press palms together (prayer). Triggers Wispr Flow OR
  built-in macOS Dictation (configurable)
- **Switch desktops** — swipe one hand across the frame
- **Toggle voice control** — double clap
- **Jam Mode** — hides the UI and maps both hands to synths (left = bass,
  right = phased lead à la Tame Impala). Smile → Rhodes pad. Bob your head → drums.
- Shooting stars drift in the background because why not.

All of this runs on your own Mac using [MediaPipe Tasks](https://developers.google.com/mediapipe/solutions)
face + hand landmarkers. Camera capture is via AVFoundation so it targets the
built-in webcam even when Continuity Camera is active.

## Setup

```bash
git clone https://github.com/jackmielke/handsfree.git
cd handsfree
./setup.sh           # creates .venv, installs deps
source .venv/bin/activate
python3 browser_viewer.py
```

Open http://localhost:8765 in your browser. First launch downloads the
MediaPipe face + hand landmarker models (~10 MB total).

### Permissions

macOS will prompt for **Camera**, **Accessibility**, and **Input Monitoring**
permission the first time. Grant all three to the Terminal app you launch
from (not to Python itself — the permission follows the parent process).

If `cv2.imshow` crashes silently on first frame, that's a known issue with
Xcode Command Line Tools Python; the app stays headless and serves the UI
over HTTP instead, which is the intended flow.

### Wispr Flow

Wispr Flow listens for the physical Fn key via IOKit HID, which synthesized
Fn events don't reach. Easiest workaround: bind Wispr's dictation hotkey to
**F19** in its settings, and the app's `cgevent_f19` trigger will wake it.

Or skip Wispr entirely — set the Control Center → wispr row to **apple
dictation**, then set System Settings → Keyboard → Dictation → Shortcut to
"Press Control key twice". Prayer hands will now trigger the built-in macOS
Dictation and type into the notes textarea on the page.

## Status

This is an active solo project. Interfaces change. Expect rough edges.

## License

MIT — do whatever you want, just don't blame me if you blink too much.
