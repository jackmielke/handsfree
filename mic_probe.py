"""Tiny mic permission probe. Run it in the same Terminal you'd use for
browser_viewer.py. It:
  1. lists audio devices
  2. tries to open the default input and read 0.5s
  3. prints a clear pass/fail and the real reason

If Terminal has never been prompted, this call is what triggers the
macOS Microphone consent dialog.
"""
import sys
import traceback

try:
    import sounddevice as sd
    import numpy as np
except Exception as e:
    print(f"FAIL: import error: {e}")
    sys.exit(1)

print("---- devices ----")
try:
    print(sd.query_devices())
except Exception as e:
    print(f"query_devices err: {e}")

print("\n---- default input ----")
try:
    print(sd.default.device)
except Exception as e:
    print(f"default.device err: {e}")

print("\n---- attempting to open mic for 0.5s ----")
try:
    # Explicitly mono 16k float32 — same as the voice daemon.
    data = sd.rec(int(0.5 * 16000), samplerate=16000, channels=1,
                  dtype="float32", blocking=True)
    rms = float(np.sqrt(np.mean(data ** 2)))
    print(f"OK — captured {len(data)} samples, rms={rms:.5f}")
    print("If rms is ~0.00000 you're likely recording silence (mic "
          "muted or denied). Any value >0 means real audio was captured.")
except Exception as e:
    print("FAIL — mic open raised:")
    traceback.print_exc()
    print("\nMost common causes:")
    print("  - Terminal has no macOS Microphone permission (open System")
    print("    Settings → Privacy & Security → Microphone, add Terminal).")
    print("  - Another app has exclusive hold of the mic.")
