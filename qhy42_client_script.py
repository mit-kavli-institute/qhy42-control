import time
from qhy42_camera import QHY42Camera

cam = QHY42Camera()

# One call handles all setup
cam.connect_and_configure(read_mode="std", stream_mode="stream", bit_depth=16, exposure_ms=200)

# Take 10 frames and save as FITS stack
cam.capture_stream_stack(n_frames=10, fname="std_stream_200_ms.fits")
cam.close()


time.sleep(1)
cam = QHY42Camera()

# do a single
cam.connect_and_configure(read_mode="std", stream_mode="single", bit_depth=16, exposure_ms=60*1000)

cam.capture_single(fname="std_single_60_s.fits")
cam.close()