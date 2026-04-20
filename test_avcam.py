"""Quick smoke test to run inside Terminal.app to verify AV capture works."""
import time
from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
from avcamera import Camera, list_devices

labels = {0: "NotDetermined", 1: "Restricted", 2: "Denied", 3: "Authorized"}
status = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeVideo)
print(f"camera auth: {labels.get(status, status)}", flush=True)

print("devices:", flush=True)
for i, d in enumerate(list_devices()):
    print(f"  {i}: {d.localizedName()}  ({d.uniqueID()})", flush=True)

cam = Camera()
print(f"picked: {cam.name}  running={cam.session.isRunning()}", flush=True)
for i in range(50):
    ok, frame = cam.read()
    if ok:
        print(f"got frame after {i*0.1:.1f}s: shape={frame.shape} "
              f"count={cam._last_count}", flush=True)
        break
    time.sleep(0.1)
else:
    print("no frames after 5s", flush=True)
cam.release()
print("released", flush=True)
