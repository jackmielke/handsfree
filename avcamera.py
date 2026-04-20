"""AVFoundation-backed camera that can target a device by name or uniqueID.

Works around OpenCV's flaky VideoCapture index ordering when Continuity
Camera is active. Exposes a cv2-compatible `read()` method so the rest
of the pipeline doesn't need to care.
"""
from __future__ import annotations

import ctypes
import threading
from typing import Optional

import numpy as np
import objc
from AVFoundation import (
    AVCaptureDeviceDiscoverySession,
    AVCaptureDeviceInput,
    AVCaptureDevicePositionUnspecified,
    AVCaptureDeviceTypeBuiltInWideAngleCamera,
    AVCaptureDeviceTypeExternal,
    AVCaptureSession,
    AVCaptureVideoDataOutput,
    AVMediaTypeVideo,
)
from CoreMedia import CMSampleBufferGetImageBuffer
from Quartz import (
    CVPixelBufferGetBaseAddress,
    CVPixelBufferGetBytesPerRow,
    CVPixelBufferGetHeight,
    CVPixelBufferGetWidth,
    CVPixelBufferLockBaseAddress,
    CVPixelBufferUnlockBaseAddress,
    kCVPixelFormatType_32BGRA,
)
from Foundation import NSObject

# libdispatch: we need a dispatch queue to hand to AVFoundation.
_libdispatch = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
_libdispatch.dispatch_queue_create.restype = ctypes.c_void_p
_libdispatch.dispatch_queue_create.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
_libdispatch.dispatch_get_global_queue.restype = ctypes.c_void_p
_libdispatch.dispatch_get_global_queue.argtypes = [ctypes.c_long, ctypes.c_ulong]

# CoreVideo, called directly so we get a raw void* instead of pyobjc's
# objc.varlist wrapper.
_corevideo = ctypes.CDLL(
    "/System/Library/Frameworks/CoreVideo.framework/CoreVideo"
)
_corevideo.CVPixelBufferGetBaseAddress.restype = ctypes.c_void_p
_corevideo.CVPixelBufferGetBaseAddress.argtypes = [ctypes.c_void_p]
_corevideo.CVPixelBufferLockBaseAddress.restype = ctypes.c_int
_corevideo.CVPixelBufferLockBaseAddress.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
_corevideo.CVPixelBufferUnlockBaseAddress.restype = ctypes.c_int
_corevideo.CVPixelBufferUnlockBaseAddress.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
_corevideo.CVPixelBufferGetWidth.restype = ctypes.c_size_t
_corevideo.CVPixelBufferGetWidth.argtypes = [ctypes.c_void_p]
_corevideo.CVPixelBufferGetHeight.restype = ctypes.c_size_t
_corevideo.CVPixelBufferGetHeight.argtypes = [ctypes.c_void_p]
_corevideo.CVPixelBufferGetBytesPerRow.restype = ctypes.c_size_t
_corevideo.CVPixelBufferGetBytesPerRow.argtypes = [ctypes.c_void_p]


def list_devices():
    s = AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
        [AVCaptureDeviceTypeBuiltInWideAngleCamera, AVCaptureDeviceTypeExternal],
        AVMediaTypeVideo,
        AVCaptureDevicePositionUnspecified,
    )
    return list(s.devices())


def pick_device(name_hint: Optional[str] = None,
                exclude_hints=("iPhone", "Continuity", "External")):
    devices = list_devices()
    if not devices:
        return None
    if name_hint:
        for d in devices:
            if name_hint.lower() in str(d.localizedName()).lower():
                return d
    # Prefer names that don't look like external/phone cameras.
    for d in devices:
        name = str(d.localizedName())
        if not any(h.lower() in name.lower() for h in exclude_hints):
            return d
    return devices[0]


def _pixbuf_to_bgr(pixbuf) -> Optional[np.ndarray]:
    # Get the raw CFTypeRef pointer for ctypes calls.
    pixbuf_ptr = objc.pyobjc_id(pixbuf)
    if _corevideo.CVPixelBufferLockBaseAddress(pixbuf_ptr, 0) != 0:
        return None
    try:
        w = _corevideo.CVPixelBufferGetWidth(pixbuf_ptr)
        h = _corevideo.CVPixelBufferGetHeight(pixbuf_ptr)
        bpr = _corevideo.CVPixelBufferGetBytesPerRow(pixbuf_ptr)
        base_addr = _corevideo.CVPixelBufferGetBaseAddress(pixbuf_ptr)
        if not base_addr or w == 0 or h == 0:
            return None
        buf = (ctypes.c_uint8 * (bpr * h)).from_address(base_addr)
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, bpr // 4, 4))
        return arr[:, :w, :3].copy()  # BGRA -> BGR; copy before unlock.
    finally:
        _corevideo.CVPixelBufferUnlockBaseAddress(pixbuf_ptr, 0)


class _FrameDelegate(NSObject):
    def init(self):
        self = objc.super(_FrameDelegate, self).init()
        if self is None:
            return None
        self._latest = None
        self._lock = threading.Lock()
        self._frame_count = 0
        return self

    @objc.signature(b"v@:@@@")
    def captureOutput_didOutputSampleBuffer_fromConnection_(
        self, _output, sample_buffer, _connection
    ):
        pixbuf = CMSampleBufferGetImageBuffer(sample_buffer)
        if pixbuf is None:
            return
        try:
            frame = _pixbuf_to_bgr(pixbuf)
        except Exception:
            return
        if frame is None:
            return
        with self._lock:
            self._latest = frame
            self._frame_count += 1

    @objc.python_method
    def latest(self):
        with self._lock:
            return self._latest, self._frame_count


class Camera:
    """AVFoundation capture with a cv2-compatible read() API."""

    def __init__(self, device=None, name_hint: Optional[str] = None):
        if device is None:
            device = pick_device(name_hint=name_hint)
        if device is None:
            raise RuntimeError("no video devices found")
        self.device = device

        self.session = AVCaptureSession.alloc().init()

        input_, err = AVCaptureDeviceInput.deviceInputWithDevice_error_(device, None)
        if input_ is None:
            raise RuntimeError(f"could not create input: {err}")
        if not self.session.canAddInput_(input_):
            raise RuntimeError("session can't add input")
        self.session.addInput_(input_)

        output = AVCaptureVideoDataOutput.alloc().init()
        output.setVideoSettings_({
            "PixelFormatType": int(kCVPixelFormatType_32BGRA),
        })
        output.setAlwaysDiscardsLateVideoFrames_(True)

        self.delegate = _FrameDelegate.alloc().init()
        # Use the default-priority global concurrent queue. Simpler and
        # more robust than creating a serial queue from Python.
        queue_ptr = _libdispatch.dispatch_get_global_queue(0, 0)
        if not queue_ptr:
            raise RuntimeError("dispatch_get_global_queue failed")
        self._queue = objc.objc_object(c_void_p=queue_ptr)
        output.setSampleBufferDelegate_queue_(self.delegate, self._queue)

        if not self.session.canAddOutput_(output):
            raise RuntimeError("session can't add output")
        self.session.addOutput_(output)
        self.output = output

        self.session.startRunning()
        self._last_count = 0

    @property
    def name(self) -> str:
        return str(self.device.localizedName())

    def read(self):
        frame, count = self.delegate.latest()
        if frame is None:
            return False, None
        self._last_count = count
        return True, frame

    def release(self):
        try:
            self.session.stopRunning()
        except Exception:
            pass


if __name__ == "__main__":
    # Quick smoke test: list devices.
    for i, d in enumerate(list_devices()):
        print(f"{i}: {d.localizedName()}  {d.uniqueID()}")
    picked = pick_device()
    if picked is not None:
        print(f"\nwould pick: {picked.localizedName()}")
