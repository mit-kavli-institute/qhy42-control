import os
import time
from ctypes import *
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

import numpy as np
from astropy.io import fits

from qcam.qCam import Qcam

class QHY42Camera:
    def __init__(self, dll_path=None, output_dir="frames"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.dll_path = dll_path or Path("C:/Program Files/QHYCCD/AllInOne/sdk/x64/qhyccd.dll")
        self.cam = Qcam(self.dll_path)

        self.handle = None
        self.cam_id = None
        self.buffer = None

        self.image_width = c_uint32()
        self.image_height = c_uint32()
        self.bits_per_pixel = c_uint32()
        self.channels = c_uint32()
        self.mem_len = 0

        # Parameters
        self.read_mode_index = 0  # 0 = STD, 1 = HDR (if supported)
        self.stream_mode = 0      # 0 = single, 1 = stream
        self.bit_depth = 16
        self.exposure_us = 200e3
        self.gain = 30
        self.offset = 40
        self.read_mode_map = {
            "hdr": 0,  # QHY confirmed via demo output
            "std": 1
        }
        self.read_mode_index = None  # to be set later
        
        self.stream_mode_map = {
            "single": 0,
            "stream": 1
        }
        self.stream_mode = None  # to be set by configuration
        
    def _map_stream_mode(self, name: str) -> int:
        key = name.lower()
        if key not in self.stream_mode_map:
            raise ValueError(f"Invalid stream mode '{name}'. Use one of: {list(self.stream_mode_map)}")
        return self.stream_mode_map[key]
    
    def set_exposure_ms(self, ms):      # in QHY42Camera
        self.exposure_us = ms * 1e3
        self.cam.so.SetQHYCCDParam(self.handle, self.cam.CONTROL_EXPOSURE, c_double(self.exposure_us))
        
    def set_exposure_us(self, us):
        self.exposure_us = us
        self.cam.so.SetQHYCCDParam(self.handle, self.cam.CONTROL_EXPOSURE, c_double(us))


    def set_cooler_enabled(self, on: bool):
        self.cam.so.SetQHYCCDParam(self.handle, self.cam.CONTROL_COOLER, 1.0 if on else 0.0)

    def set_read_mode(self, mode_name: str):
        """Set read mode using a name like 'std' or 'hdr'."""
        key = mode_name.lower()
        if key not in self.read_mode_map:
            raise ValueError(f"Unknown read mode '{mode_name}'. Choose from: {list(self.read_mode_map.keys())}")
        self.read_mode_index = self.read_mode_map[key]
    
    def log_read_modes(self):
        """List available readout modes and their indices."""
        num_modes = c_uint32()
        self.cam.so.GetReadModesNumber(self.cam_id, byref(num_modes))
        print(f"[INFO] Camera has {num_modes.value} read modes:")
        for idx in range(num_modes.value):
            name_buf = create_string_buffer(self.cam.STR_BUFFER_SIZE)
            self.cam.so.GetReadModeName(self.cam_id, idx, name_buf)
            print(f"  [{idx}] {name_buf.value.decode()}")

    def connect_and_configure(self, read_mode="std", stream_mode="single", bit_depth=16, exposure_ms=200, gain=1, offset=200):
        self.set_read_mode(read_mode)
        self.stream_mode = self._map_stream_mode(stream_mode)
        self.bit_depth = bit_depth
        self.exposure_us = exposure_ms*1e3
        print(f"exposure: {exposure_ms} ms = {self.exposure_us} us")
        self.gain = gain
        self.offset = offset
        self.cam.so.InitQHYCCDResource()
        num_cams = self.cam.so.ScanQHYCCD()
        if num_cams <= 0:
            raise RuntimeError("No QHY cameras found.")
    
        cam_id_buf = create_string_buffer(self.cam.STR_BUFFER_SIZE)
        self.cam.so.GetQHYCCDId(0, cam_id_buf)
        self.cam_id = cam_id_buf.value
        print(f"[INFO] Found camera: {self.cam_id.decode()}")
    
        self.handle = c_void_p(self.cam.so.OpenQHYCCD(self.cam_id))

        if not self.handle:
            raise RuntimeError("Failed to open camera.")
            
        num_modes = c_uint32()
        self.cam.so.GetReadModesNumber(self.cam_id, byref(num_modes))
        print(f"[DEBUG] Camera has {num_modes.value} read modes:")
        
        for idx in range(num_modes.value):
            name_buf = create_string_buffer(self.cam.STR_BUFFER_SIZE)
            self.cam.so.GetReadModeName(self.cam_id, idx, name_buf)
            print(f"  [{idx}] {name_buf.value.decode()}")

        # 1. SET READ MODE BEFORE INIT
        ret = self.cam.so.SetQHYCCDReadMode(self.handle, self.read_mode_index)
        print(f"[DEBUG] SetQHYCCDReadMode({self.read_mode_index}) returned {ret}")
    
        # 2. SET STREAM MODE
        ret = self.cam.so.SetQHYCCDStreamMode(self.handle, self.stream_mode)
        print(f"[DEBUG] SetQHYCCDStreamMode({self.stream_mode}) returned {ret}")
    
        # 3. INIT CAMERA
        ret = self.cam.so.InitQHYCCD(self.handle)
        print(f"[DEBUG] InitQHYCCD returned {ret}")
    
        # 4. SET BITS
        self.cam.so.SetQHYCCDBitsMode(self.handle, c_uint32(self.bit_depth))
    
        # 5. GET CHIP INFO
        self.cam.so.GetQHYCCDChipInfo(
            self.handle,
            byref(c_double()), byref(c_double()),
            byref(self.image_width), byref(self.image_height),
            byref(c_double()), byref(c_double()),
            byref(self.bits_per_pixel)
        )
    
        print(f"[INFO] Detected geometry: {self.image_width.value} x {self.image_height.value}")
    
        # 6. SET RESOLUTION
        self.cam.so.SetQHYCCDResolution(
            self.handle,
            c_uint32(0), c_uint32(0),
            self.image_width, self.image_height
        )
    
        # 7. SET PARAMETERS
        self.cam.so.SetQHYCCDParam(self.handle, self.cam.CONTROL_EXPOSURE, c_double(self.exposure_us))
        self.cam.so.SetQHYCCDParam(self.handle, self.cam.CONTROL_GAIN, c_double(self.gain))
        self.cam.so.SetQHYCCDParam(self.handle, self.cam.CONTROL_OFFSET, c_double(self.offset))
    
        # 8. ALLOCATE BUFFER
        self.mem_len = self.cam.so.GetQHYCCDMemLength(self.handle)
        if self.bit_depth == 16:
            self.buffer = (c_uint16 * (self.mem_len // 2))()
        else:
            self.buffer = (c_uint8 * self.mem_len)()
    
        if self.stream_mode == 1:
            ret = self.cam.so.BeginQHYCCDLive(self.handle)
            print(f"[DEBUG] BeginQHYCCDLive returned {ret}")
    
        print(f"[INFO] Final geometry: {self.image_width.value} x {self.image_height.value} @ {self.bit_depth} bits")
    
    def capture_single(self, fname="frame_single.fits"):
        """Single‑frame capture with bullet‑proof tqdm progress bar."""
        assert self.stream_mode == 0, "capture_single() only valid in single‑frame mode"
    
        total_s = self.exposure_us / 1e6
        print(f"[INFO] Capturing single frame ({total_s:.1f} s)…")
    
        # 1) trigger exposure
        self.cam.so.ExpQHYCCDSingleFrame(self.handle)
        t0 = time.perf_counter()           # wall clock start
    
        # 2) poll remaining time
        us_left = c_double()
        bar_done = 0.0                     # seconds already sent to tqdm
    
        with tqdm(total=total_s, unit="s", desc="Exposing") as bar:
            while bar_done < total_s:
                # --- wall‑clock estimate
                elapsed = time.perf_counter() - t0
    
                # --- SDK estimate
                self.cam.so.GetQHYCCDExposureRemaining(self.handle, byref(us_left))
                sec_left_sdk = max(0.0, us_left.value / 1_000_000)
                
                # we're also keeping track ourselves since the sdk is not reliably reading back remaining time to expose.
                # choose whichever says “less done”
                sec_left = max(total_s - elapsed, sec_left_sdk)
                done = total_s - sec_left
    
                # update if progress advanced by ≥0.3 s
                delta = done - bar_done
                if delta >= 0.3:
                    bar.update(delta)
                    bar_done += delta
    
                if sec_left <= 0.0:
                    break
                time.sleep(0.3)
    
            # guarantee bar is full
            if bar_done < total_s:
                bar.update(total_s - bar_done)
    
        # 3) read the frame
        success = self.cam.so.GetQHYCCDSingleFrame(
            self.handle,
            byref(self.image_width), byref(self.image_height),
            byref(self.bits_per_pixel), byref(self.channels),
            byref(self.buffer)
        )
        if success != self.cam.QHYCCD_SUCCESS:
            raise RuntimeError("Failed to acquire single frame.")
    
        # 4) save
        image = self._extract_image()
        if fname is None:
            return image.copy()            # ← deep copy fixes list issue
        else:
            self._write_fits(fname, image)

        print(f"[INFO] Single frame saved ➜ {fname}")
        
    def capture_stream_stack(self, n_frames=5, fname="frame_stack.fits", flatten = False):
        if self.stream_mode != self.stream_mode_map["stream"]:
            raise RuntimeError("capture_stream_stack() called while not in stream mode.")
    
        print("[INFO] Capturing stream stack...")
        images = []
        warmup = 3
        acquired = 0
        failures = 0
        expected_dt = self.exposure_us / 1e6
    
        frame_read_times = []
        frame_timestamps = []
    
        time.sleep(2.0)  # allow pipeline to fill
    
        with tqdm(total=n_frames, desc="Capturing frames", unit="frame") as pbar:
            while len(images) < n_frames and failures < 50:
                t0 = time.time()
                success = self.cam.so.GetQHYCCDLiveFrame(
                    self.handle,
                    byref(self.image_width), byref(self.image_height),
                    byref(self.bits_per_pixel), byref(self.channels),
                    byref(self.buffer)
                )
                t1 = time.time()
                dt = t1 - t0
    
                if success != self.cam.QHYCCD_SUCCESS:
                    failures += 1
                    continue
    
                if acquired < warmup:
                    acquired += 1
                    continue
    
                image = self._extract_image()
                images.append(image.copy())
                frame_read_times.append(dt)
                frame_timestamps.append(t1)
    
                #tqdm.write(f"[INFO] Captured frame {len(images)}/{n_frames} (Δt={dt:.3f}s)")
                pbar.update(1)
                acquired += 1
    
                # Sleep until next frame should be ready
                delay = t0 + expected_dt - time.time()
                if delay > 0:
                    time.sleep(delay)
    
        if not images:
            raise RuntimeError("No valid frames captured.")
        
        if fname is None:
            return images
        else:
            self._write_fits_stack(fname, images, flatten = flatten)
        print("[INFO] Stream stack saved.")
    
        # ──────────────────────────────────────────────
        # 📊 Summary Report
        # ──────────────────────────────────────────────
        gaps = np.diff(frame_timestamps)
        avg_read_time = np.mean(frame_read_times)
        avg_gap = np.mean(gaps) if len(gaps) > 0 else float('nan')
        fps = 1.0 / avg_gap if avg_gap > 0 else float('nan')
    
        print("\n[📊 Summary]")
        print(f"  Frames requested: {n_frames}")
        print(f"  Frames acquired : {len(images)}")
        print(f"  Frames failed   : {failures}")
        print(f"  Avg read Δt     : {avg_read_time:.3f} s")
        print(f"  Avg gap between : {avg_gap:.3f} s")
        print(f"  Estimated FPS   : {fps:.2f}")

   # ---- Add these helper methods to QHY42Camera ----
    
    
    def start_live(self):
        """Begin live‑stream mode (after connect_and_configure)."""
        self.cam.so.BeginQHYCCDLive(self.handle)
    
    def get_live_frame(self):
        """
        Pull one live frame.
        Returns (success_bool, ndarray)
        """
        w   = c_uint32()
        h   = c_uint32()
        bpp = c_uint32()
        ch  = c_uint32()
    
        success = self.cam.so.GetQHYCCDLiveFrame(
            self.handle,
            byref(w), byref(h), byref(bpp), byref(ch),
            byref(self.buffer)
        )
    
        if success != self.cam.QHYCCD_SUCCESS:
            return False, None
    
        frame = self._extract_image()      # uses self.buffer, self.image_width/height if you prefer
        return True, frame
    
    def set_gain(self, val):
        self.cam.so.SetQHYCCDParam(self.handle, self.cam.CONTROL_GAIN, c_double(val))
    
    def set_offset(self, val):
        self.cam.so.SetQHYCCDParam(self.handle, self.cam.CONTROL_OFFSET, c_double(val))
    
    def set_tec_temperature(self, val):
        self.cam.so.SetQHYCCDParam(self.handle, self.cam.CONTROL_COOLER, 1.0)
        self.cam.so.SetQHYCCDParam(self.handle, self.cam.CONTROL_CURTEMP, c_double(val))
    
    def get_temperature(self):
        return self.cam.so.GetQHYCCDParam(self.handle, self.cam.CONTROL_CURTEMP)
    
    
    # ---------------------------------------------------------------------
    # Public RPC‑friendly helper
    # ---------------------------------------------------------------------
    def do_exposure(self, exptime: float, n_imgs: int = 1, flatten = False) -> str:
        """
        Take `n_imgs` images of length `exptime` seconds in STD mode,
        bit‑depth 16.  Returns the path of the FITS file saved.
    
        •  If exptime > stream_max_time s  → single captures, stacked manually.
        •  If exptime ≤ stream_max_time s  → live‐stream stack unless n_imgs == 1.
        """
        stream_max_time = 0.5
        # ---------- mandatory config ----------
        self.connect_and_configure(
            read_mode="std",
            stream_mode="single" if exptime >= stream_max_time or n_imgs == 1 else "stream",
            bit_depth=16,
            exposure_ms=int(exptime * 1000)
        )
    
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out = self.output_dir / f"exp_{exptime:.1f}s_{n_imgs}_{ts}.fits"
    
        # ---------- choose acquisition path ----
        if n_imgs == 1:
            self.capture_single(out.name)
    
        elif exptime >= stream_max_time:                     # long → repeated singles
            stack = []
            for i in range(n_imgs):
                layer_data = self.capture_single(fname = None)
                stack.append(layer_data)
            # flatten?
            print(f"shape of stack: {np.shape(stack)}")
            self._write_fits_stack(out.name, stack, flatten = flatten)
    
        else:                                  # short → live stack
            self.capture_stream_stack(n_frames=n_imgs, fname=out.name, flatten = flatten)
    
        print(f"[RPC] saved -> {out}")
        return str(out)

    

    def _extract_image(self):
        arr = np.ctypeslib.as_array(self.buffer)
        size = self.image_width.value * self.image_height.value
        return arr[:size].reshape((self.image_height.value, self.image_width.value))

    def _write_fits(self, fname, image):
        fits.writeto(self.output_dir / fname, image, overwrite=True)

    def _write_fits_stack(self, fname, images, flatten=False):
        """
        Save a stack of images to FITS.
    
        Parameters
        ----------
        fname : str
            Output filename (relative to self.output_dir).
        images : List[np.ndarray] or np.ndarray
            List/array with shape (N, H, W).
        flatten : bool, default False
            If True, sum along the stack axis and save the 2‑D image only.
        """
        if flatten:
            summed = np.sum(images, axis=0).astype(images[0].dtype)
            print(f'flattened shape: {np.shape(summed)}')
            fits.writeto(self.output_dir / fname, summed, overwrite=True)
            print(f"[INFO] Flattened stack saved -> {fname}")
            return
    
        # --- regular multi‑extension cube ---
        hdul = fits.HDUList()
        for i, img in enumerate(images):
            hdu = fits.ImageHDU(img)
            hdu.header["FRAME"] = i
            hdul.append(hdu)
        hdul.writeto(self.output_dir / fname, overwrite=True)
        print(f"[INFO] Stack saved -> {fname}")


    def close(self):
        if self.handle:
            self.cam.so.CloseQHYCCD(self.handle)
            self.handle = None


if __name__ == "__main__":
    cam = QHY42Camera()

    # ---- SINGLE MODE TEST ----
    cam.read_mode_index = 0  # 0 = STD
    cam.stream_mode = 0      # 0 = single
    cam.bit_depth = 16
    cam.exposure_ms = 200
    cam.connect_and_configure()
    cam.capture_single("std_single.fits")
    cam.close()

    # ---- STREAM MODE TEST ----
    cam = QHY42Camera()
    cam.read_mode_index = 0  # 0 = STD
    cam.stream_mode = 1      # 1 = stream
    cam.bit_depth = 16
    cam.exposure_ms = 200
    cam.connect_and_configure()
    cam.capture_stream_stack(5, "std_stream_stack.fits")
    cam.close()
