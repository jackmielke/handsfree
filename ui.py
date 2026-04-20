"""Tk-based preview window for handsfree — shows what the camera sees plus
head-pose and facial-gesture scores so you can see the model's brain work."""
from __future__ import annotations

import tkinter as tk
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageTk

BG = "#0e0e12"
PANEL_BG = "#15151c"
TEXT_DIM = "#8b8b96"
TEXT = "#ececf1"
ACCENT = "#6ee7b7"
ACCENT_RED = "#f87171"
ACCENT_AMBER = "#fbbf24"

STATUS_COLORS = {
    "TRACKING": ACCENT,
    "IDLE": ACCENT_AMBER,
    "CALIBRATING": "#60a5fa",
    "SEARCHING": TEXT_DIM,
}


class PreviewWindow:
    def __init__(self, title: str = "handsfree", video_width: int = 640):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.configure(bg=BG)
        self.root.geometry("1040x540")
        self.video_width = video_width

        video_frame = tk.Frame(self.root, bg=BG)
        video_frame.pack(side="left", padx=12, pady=12)
        self.video_label = tk.Label(video_frame, bg="#000000")
        self.video_label.pack()

        side = tk.Frame(self.root, bg=BG, width=360)
        side.pack(side="right", fill="both", expand=True, padx=(0, 16), pady=16)

        self.status_var = tk.StringVar(value="SEARCHING")
        self.status_label = tk.Label(
            side, textvariable=self.status_var,
            font=("Menlo", 30, "bold"), fg=TEXT_DIM, bg=BG, anchor="w",
        )
        self.status_label.pack(fill="x", anchor="w")

        self.cam_var = tk.StringVar(value="camera: —")
        tk.Label(side, textvariable=self.cam_var, font=("Menlo", 11),
                 fg=TEXT_DIM, bg=BG, anchor="w").pack(fill="x", anchor="w",
                                                      pady=(2, 14))

        # Head pose readout
        pose_row = tk.Frame(side, bg=BG)
        pose_row.pack(fill="x", anchor="w")
        self.dy_var = tk.StringVar(value="dy +0.0°")
        self.dp_var = tk.StringVar(value="dp +0.0°")
        for var in (self.dy_var, self.dp_var):
            tk.Label(pose_row, textvariable=var, font=("Menlo", 14),
                     fg=TEXT, bg=BG, width=10, anchor="w"
                     ).pack(side="left", padx=(0, 6))

        # Gesture score bars
        bars_frame = tk.Frame(side, bg=BG)
        bars_frame.pack(fill="x", anchor="w", pady=(18, 0))
        tk.Label(bars_frame, text="GESTURES", font=("Menlo", 10, "bold"),
                 fg=TEXT_DIM, bg=BG, anchor="w").pack(fill="x", anchor="w")

        self._bar_canvases: dict[str, tuple[tk.Canvas, int]] = {}
        self._bar_thresholds = {
            "jaw (toggle)": 0.55,
            "brow (click)": 0.55,
            "smile (r-click)": 0.60,
            "pucker (quit)": 0.55,
        }
        for label in self._bar_thresholds:
            row = tk.Frame(bars_frame, bg=BG)
            row.pack(fill="x", pady=4)
            tk.Label(row, text=label, font=("Menlo", 11), fg=TEXT_DIM,
                     bg=BG, width=18, anchor="w").pack(side="left")
            canvas = tk.Canvas(row, width=200, height=14, bg=PANEL_BG,
                               highlightthickness=0, bd=0)
            canvas.pack(side="left", fill="x", expand=True)
            fill_id = canvas.create_rectangle(0, 0, 0, 14, fill=ACCENT,
                                              width=0)
            # threshold marker
            thr_x = int(200 * self._bar_thresholds[label])
            canvas.create_line(thr_x, 0, thr_x, 14, fill=TEXT_DIM, dash=(2, 2))
            self._bar_canvases[label] = (canvas, fill_id)

        tk.Label(
            side,
            text="head = cursor\njaw open = toggle on/off\n"
                 "eyebrow raise = click\nsmile = right-click\n"
                 "long pucker (1.5s) = quit\nor just close this window",
            font=("Menlo", 10), fg=TEXT_DIM, bg=BG, justify="left",
            anchor="w",
        ).pack(fill="x", anchor="w", pady=(22, 0))

        self._photo: Optional[ImageTk.PhotoImage] = None
        self._closed = False
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def set_camera_name(self, name: str) -> None:
        self.cam_var.set(f"camera: {name}")

    def _on_close(self) -> None:
        self._closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def closed(self) -> bool:
        return self._closed

    def update(
        self,
        frame_bgr: np.ndarray,
        status: str,
        dy: float,
        dp: float,
        blendshapes: dict[str, float],
        landmarks: Optional[list[tuple[float, float]]] = None,
    ) -> None:
        if self._closed:
            return

        # BGR (camera) -> RGB (PIL).
        rgb = frame_bgr[:, :, ::-1]
        h, w = rgb.shape[:2]
        scale = self.video_width / float(w)
        new_w = self.video_width
        new_h = int(h * scale)
        img = Image.fromarray(rgb).resize((new_w, new_h), Image.BILINEAR)

        if landmarks:
            draw = ImageDraw.Draw(img)
            for (nx, ny) in landmarks:
                x, y = int(nx * new_w), int(ny * new_h)
                draw.ellipse((x - 1, y - 1, x + 1, y + 1),
                             fill=(110, 231, 183, 220))

        self._photo = ImageTk.PhotoImage(img)
        self.video_label.configure(image=self._photo)

        self.status_var.set(status)
        self.status_label.configure(fg=STATUS_COLORS.get(status, TEXT))
        self.dy_var.set(f"dy {dy:+5.1f}°")
        self.dp_var.set(f"dp {dp:+5.1f}°")

        # Gesture bars
        bar_values = {
            "jaw (toggle)": blendshapes.get("jawOpen", 0.0),
            "brow (click)": max(
                blendshapes.get("browInnerUp", 0.0),
                blendshapes.get("browOuterUpLeft", 0.0),
                blendshapes.get("browOuterUpRight", 0.0),
            ),
            "smile (r-click)": max(
                blendshapes.get("mouthSmileLeft", 0.0),
                blendshapes.get("mouthSmileRight", 0.0),
            ),
            "pucker (quit)": blendshapes.get("mouthPucker", 0.0),
        }
        for label, value in bar_values.items():
            canvas, fill_id = self._bar_canvases[label]
            threshold = self._bar_thresholds[label]
            width = max(0, min(200, int(value * 200)))
            color = ACCENT if value >= threshold else ACCENT_AMBER
            canvas.itemconfigure(fill_id, fill=color)
            canvas.coords(fill_id, 0, 0, width, 14)

    def pump(self) -> None:
        if self._closed:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self._closed = True

    def run(self, on_tick, interval_ms: int = 20) -> None:
        """Hand control to Tk's mainloop and call on_tick periodically.

        on_tick is a zero-arg callable that does one frame of work
        (camera read + detect + update UI). Returning False from on_tick
        stops the loop.
        """

        def _tick():
            if self._closed:
                return
            try:
                keep_going = on_tick()
            except Exception:
                import traceback
                traceback.print_exc()
                keep_going = False
            if keep_going is False:
                self._on_close()
                return
            if not self._closed:
                self.root.after(interval_ms, _tick)

        self.root.after(0, _tick)
        self.root.mainloop()
