#!/usr/bin/env python3
"""handsfree — camera-driven head pointer for macOS.

Head movement drives the cursor. Facial gestures trigger actions:
  - jaw open (say "aaaah")      -> toggle tracking on/off
  - eyebrow raise               -> left click
  - smile (either corner)       -> right click
  - long mouth pucker (>1.5s)   -> quit

Safety: pyautogui FAILSAFE is ON. Slam the cursor into any screen corner
to abort instantly. Ctrl-C in the terminal to quit.

Preview window is OFF by default (cv2.imshow crashes silently on some
macOS + non-bundled-Python setups). Enable with HANDSFREE_PREVIEW=1.

First run will download the MediaPipe face landmarker model (~3 MB).
"""
import os
import sys
import time
import traceback
import urllib.request
from pathlib import Path

# Unbuffered stdio so logs reach tee / terminal immediately.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from avcamera import Camera as AVCamera

SHOW_PREVIEW = os.environ.get("HANDSFREE_PREVIEW", "0") == "1"
SHOW_UI = os.environ.get("HANDSFREE_UI", "1") == "1"
CAMERA_NAME_HINT = os.environ.get("HANDSFREE_CAMERA_NAME")  # e.g. "MacBook"
SAVE_TEST_FRAME = os.environ.get("HANDSFREE_TEST_FRAME", "0") == "1"

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

HERE = Path(__file__).parent
MODEL_PATH = HERE / "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

# --- Tuning knobs -----------------------------------------------------------
SENSITIVITY_X = 55.0     # pixels per degree of yaw
SENSITIVITY_Y = 55.0     # pixels per degree of pitch
SMOOTHING = 0.28         # lower = smoother (laggier), higher = snappier
DEAD_ZONE_DEG = 1.8      # head movement smaller than this = no cursor change

BROW_CLICK_THRESHOLD = 0.55
JAW_TOGGLE_THRESHOLD = 0.55
SMILE_RIGHTCLICK_THRESHOLD = 0.6
PUCKER_QUIT_THRESHOLD = 0.55
PUCKER_QUIT_HOLD_S = 1.5
GESTURE_COOLDOWN_S = 0.7
# ----------------------------------------------------------------------------


def ensure_model() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    print(f"[handsfree] downloading face landmarker model -> {MODEL_PATH}")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def matrix_to_euler_degrees(matrix) -> tuple[float, float, float]:
    r = np.array(matrix).reshape(4, 4)[:3, :3]
    sy = float(np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2))
    if sy > 1e-6:
        yaw = np.degrees(np.arctan2(r[1, 0], r[0, 0]))
        pitch = np.degrees(np.arctan2(-r[2, 0], sy))
        roll = np.degrees(np.arctan2(r[2, 1], r[2, 2]))
    else:
        yaw = 0.0
        pitch = np.degrees(np.arctan2(-r[2, 0], sy))
        roll = np.degrees(np.arctan2(-r[1, 2], r[1, 1]))
    return float(yaw), float(pitch), float(roll)


def blend_score(blendshapes, name: str) -> float:
    for b in blendshapes:
        if b.category_name == name:
            return float(b.score)
    return 0.0


def main() -> int:
    ensure_model()

    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    # Use our AVFoundation-backed camera so we target the MacBook cam by name,
    # bypassing OpenCV's flaky index ordering when Continuity Camera is active.
    try:
        cap = AVCamera(name_hint=CAMERA_NAME_HINT)
    except Exception as e:
        print(f"[handsfree] could not start camera: {e}", flush=True)
        return 1
    print(f"[handsfree] opened camera: {cap.name}", flush=True)

    # Warm up: wait for the first frame to land from the capture delegate.
    warm_ok = False
    for attempt in range(50):
        ok, _ = cap.read()
        if ok:
            warm_ok = True
            if attempt > 0:
                print(f"[handsfree] camera warm after {attempt * 0.1:.1f}s",
                      flush=True)
            break
        time.sleep(0.1)
    if not warm_ok:
        print("[handsfree] camera opened but no frames after 5s.", flush=True)
        cap.release()
        return 1

    screen_w, screen_h = pyautogui.size()

    tracking = False
    last_gesture_at = 0.0
    pucker_start: float | None = None

    yaw_center = 0.0
    pitch_center = 0.0
    calibrated = False
    calib_started_at = time.time()

    cur_x = float(screen_w) / 2
    cur_y = float(screen_h) / 2

    print("[handsfree] look at the camera and hold head still ~2s to calibrate",
          flush=True)
    print("[handsfree] jaw open = toggle | brow raise = click | "
          "smile = right-click | long pucker = quit | ctrl-c = quit", flush=True)
    print(f"[handsfree] UI window: {'ON' if SHOW_UI else 'OFF'}", flush=True)

    ui = None
    if SHOW_UI:
        from ui import PreviewWindow  # local import keeps Tk out of headless runs
        ui = PreviewWindow()
        ui.set_camera_name(cap.name)

    frame_count = 0
    t0 = time.time()

    def process_frame() -> bool:
        """Run one frame of capture -> detect -> cursor/gesture -> UI update.
        Returns False when the app should exit."""
        nonlocal tracking, last_gesture_at, pucker_start
        nonlocal yaw_center, pitch_center, calibrated
        nonlocal cur_x, cur_y, frame_count

        ok, frame = cap.read()
        if not ok:
            return True  # no new frame yet, try again next tick
        frame_count += 1
        if frame_count == 1:
            print(f"[handsfree] first frame captured ({frame.shape})",
                  flush=True)
            if SAVE_TEST_FRAME:
                cv2.imwrite("/tmp/handsfree-test-frame.jpg", frame)

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.time() - t0) * 1000)
        result = landmarker.detect_for_video(mp_img, ts_ms)

        status = "SEARCHING"
        dy = dp = 0.0

        if result.face_landmarks and result.facial_transformation_matrixes:
            yaw, pitch, _ = matrix_to_euler_degrees(
                result.facial_transformation_matrixes[0]
            )
            if not calibrated:
                status = "CALIBRATING"
                if time.time() - calib_started_at > 2.0:
                    yaw_center, pitch_center = yaw, pitch
                    calibrated = True
                    print(f"[handsfree] calibrated (yaw={yaw:.1f} "
                          f"pitch={pitch:.1f})", flush=True)
            else:
                dy = yaw - yaw_center
                dp = pitch - pitch_center
                if abs(dy) < DEAD_ZONE_DEG:
                    dy = 0.0
                if abs(dp) < DEAD_ZONE_DEG:
                    dp = 0.0
                target_x = screen_w / 2 + dy * SENSITIVITY_X
                target_y = screen_h / 2 - dp * SENSITIVITY_Y
                target_x = max(2.0, min(screen_w - 2.0, target_x))
                target_y = max(2.0, min(screen_h - 2.0, target_y))
                cur_x += (target_x - cur_x) * SMOOTHING
                cur_y += (target_y - cur_y) * SMOOTHING
                if tracking:
                    try:
                        pyautogui.moveTo(cur_x, cur_y, _pause=False)
                    except pyautogui.FailSafeException:
                        tracking = False
                        print("[handsfree] failsafe triggered, tracking OFF",
                              flush=True)
                status = "TRACKING" if tracking else "IDLE"

            if result.face_blendshapes and calibrated:
                bs = result.face_blendshapes[0]
                jaw = blend_score(bs, "jawOpen")
                brow = max(
                    blend_score(bs, "browInnerUp"),
                    blend_score(bs, "browOuterUpLeft"),
                    blend_score(bs, "browOuterUpRight"),
                )
                smile = max(
                    blend_score(bs, "mouthSmileLeft"),
                    blend_score(bs, "mouthSmileRight"),
                )
                pucker = blend_score(bs, "mouthPucker")
                now = time.time()
                cooldown_ok = (now - last_gesture_at) > GESTURE_COOLDOWN_S

                if pucker > PUCKER_QUIT_THRESHOLD:
                    if pucker_start is None:
                        pucker_start = now
                    elif now - pucker_start > PUCKER_QUIT_HOLD_S:
                        print("[handsfree] long pucker, quitting", flush=True)
                        return False
                else:
                    pucker_start = None

                if cooldown_ok and jaw > JAW_TOGGLE_THRESHOLD:
                    tracking = not tracking
                    last_gesture_at = now
                    print(f"[handsfree] tracking "
                          f"{'ON' if tracking else 'OFF'}", flush=True)
                elif cooldown_ok and tracking and brow > BROW_CLICK_THRESHOLD:
                    pyautogui.click(_pause=False)
                    last_gesture_at = now
                    print("[handsfree] click", flush=True)
                elif (cooldown_ok and tracking
                      and smile > SMILE_RIGHTCLICK_THRESHOLD):
                    pyautogui.rightClick(_pause=False)
                    last_gesture_at = now
                    print("[handsfree] right-click", flush=True)

        if ui is not None:
            bs_dict: dict[str, float] = {}
            lms = None
            if result.face_blendshapes:
                for b in result.face_blendshapes[0]:
                    bs_dict[b.category_name] = float(b.score)
            if result.face_landmarks:
                lms = [(lm.x, lm.y) for lm in result.face_landmarks[0][::4]]
            ui.update(frame, status, dy, dp, bs_dict, lms)
        elif frame_count % 60 == 0:
            print(f"[handsfree] alive f={frame_count} {status} "
                  f"dy={dy:+.1f} dp={dp:+.1f}", flush=True)
        return True

    if ui is not None:
        ui.run(process_frame, interval_ms=10)
    else:
        while process_frame():
            pass

    cap.release()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[handsfree] interrupted, bye", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"[handsfree] unhandled error: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
