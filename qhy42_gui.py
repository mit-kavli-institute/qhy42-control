import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from astropy.visualization import ZScaleInterval
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QGridLayout,
    QWidget, QSpinBox, QFileDialog, QHBoxLayout, QComboBox
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QThread, Signal, Qt, Slot

from qhy42_camera import QHY42Camera  # backend wrapper

# =============================================================================
# Worker thread
# =============================================================================
class CameraWorker(QThread):
    """Background acquisition in live mode."""
    frame_ready = Signal(np.ndarray)
    temperature_ready = Signal(float)
    fps_ready = Signal(float)

    def __init__(self, cam: QHY42Camera):
        super().__init__()
        self.cam = cam
        self._run = False
        self._last_ts = None

    # --- public controls ---------------------------------------------------
    def start_stream(self):
        if not self.isRunning():
            self.start()
        self._run = True

    def stop_stream(self):
        self._run = False
    def stop_thread(self):
        """Graceful thread shutdown."""
        self.stop_stream()                  # stop inner live loop
        self.requestInterruption()          # ➊ signal run() loop to exit
        self.quit()                         # ➋ stop Qt event loop (safe if none)
        self.wait()                         # ➌ wait until finished
        try:
            self.cam.close()
        except Exception as e:
            print("Camera close error:", e)

    # --- thread entry ------------------------------------------------------
    def run(self):
        # one‑time camera setup (done once thread starts)
        self.cam.connect_and_configure(read_mode="std", stream_mode="stream", exposure_ms=200)
        self.cam.start_live()
        # wait for explicit start_stream

        while self.isInterruptionRequested() is False:
            if self._run:
                t0 = time.time()
                ok, frame = self.cam.get_live_frame()
                if ok:
                    self.frame_ready.emit(frame)
                    if self._last_ts:
                        dt = t0 - self._last_ts
                        if dt > 0:
                            self.fps_ready.emit(1.0 / dt)
                    self._last_ts = t0
                    self.temperature_ready.emit(self.cam.get_temperature())
            time.sleep(0.003)

    # --- parameter passthroughs -------------------------------------------
    def set_gain(self, val):   self.cam.set_gain(val)
    def set_offset(self, val): self.cam.set_offset(val)
    def set_tec(self, val):    self.cam.set_tec_temperature(val)

# =============================================================================
# Histogram widget
# =============================================================================
class HistogramCanvas(FigureCanvas):
    def __init__(self, every: int = 10):
        fig = Figure(figsize=(3, 2))
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self._every = every
        self._counter = 0

    def update_histogram(self, img: np.ndarray):
        self._counter += 1
        if self._counter % self._every:
            return  # throttle updates
        self.ax.clear()
        self.ax.hist(img.ravel(), bins=100, color="black")
        self.ax.set_title("Histogram")
        self.draw_idle()

# =============================================================================
# Main GUI
# =============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QHY42 Live Viewer")

        # ---------- backend ------------
        self.camera = QHY42Camera()
        self.worker = CameraWorker(self.camera)
        self.worker.frame_ready.connect(self.on_new_frame)
        self.worker.temperature_ready.connect(lambda t: self.temp_lbl.setText(f"Temp: {t:.1f} °C"))
        self.worker.fps_ready.connect(lambda f: self.fps_lbl.setText(f"FPS: {f:.2f}"))

        # ---------- widgets ------------
        self.frame_lbl = QLabel("No frame yet"); self.frame_lbl.setFixedSize(600, 600)
        self.hist_canvas = HistogramCanvas()
        self.frame_lbl = QLabel("No frame yet")
        self.frame_lbl.setFixedSize(600, 600)          # preview stays 600 px wide
        
        self.hist_canvas = HistogramCanvas()
        self.hist_canvas.setFixedWidth(self.frame_lbl.width())  # ← add this line

        self.start_btn = QPushButton("Start Preview")
        self.stop_btn  = QPushButton("Stop Preview"); self.stop_btn.setEnabled(False)
        self.save_btn  = QPushButton("Save Current Frame")

        self.n_frames_ed = QSpinBox(); self.n_frames_ed.setRange(1, 1000); self.n_frames_ed.setValue(100)
        self.acquire_btn = QPushButton("Acquire Stack")

        self.gain_ed   = QSpinBox(); self.gain_ed.setRange(0, 100); self.gain_ed.setValue(30)
        self.offset_ed = QSpinBox(); self.offset_ed.setRange(0, 1000); self.offset_ed.setValue(40)
        self.tec_ed    = QSpinBox(); self.tec_ed.setRange(-40, 20); self.tec_ed.setValue(0)
        self.tec_button = QPushButton("Enable TEC"); self.tec_enabled = False

        self.exp_ed   = QSpinBox(); self.exp_ed.setRange(1, 2_000_000); self.exp_ed.setValue(200)
        self.exp_unit = QComboBox(); self.exp_unit.addItems(["ms", "us", "s"]); self.exp_unit.setCurrentText("ms")

        self.fps_lbl  = QLabel("FPS: —"); self.temp_lbl = QLabel("Temp: —")
        
        self.frame_lbl = QLabel("No frame yet")
        self.frame_lbl.setFixedSize(600, 600)

        # ---------- layout -------------
        grid = QGridLayout()
        grid.addWidget(QLabel("Gain"),     0, 0); grid.addWidget(self.gain_ed,   0, 1)
        grid.addWidget(QLabel("Offset"),   1, 0); grid.addWidget(self.offset_ed, 1, 1)
        grid.addWidget(QLabel("TEC °C"),   2, 0); grid.addWidget(self.tec_ed,    2, 1); grid.addWidget(self.tec_button, 2, 2)
        grid.addWidget(QLabel("Exp"),      3, 0); grid.addWidget(self.exp_ed,    3, 1); grid.addWidget(self.exp_unit,   3, 2)

        hctrl = QHBoxLayout(); hctrl.addWidget(self.start_btn); hctrl.addWidget(self.stop_btn)
        hctrl.addWidget(QLabel("Stack N:")); hctrl.addWidget(self.n_frames_ed); hctrl.addWidget(self.acquire_btn)
        hctrl.addStretch(); hctrl.addWidget(self.save_btn)

        ctrl_box = QVBoxLayout()
        ctrl_box.addLayout(grid)
        ctrl_box.addLayout(hctrl)
        ctrl_box.addWidget(self.fps_lbl)
        ctrl_box.addWidget(self.temp_lbl)
        
        ctrl_container = QWidget()         # ➋ lock controls to preview width
        ctrl_container.setLayout(ctrl_box)
        ctrl_container.setMaximumWidth(self.frame_lbl.width())
        
        vbox = QVBoxLayout()
        vbox.addWidget(self.frame_lbl, alignment=Qt.AlignHCenter)
        vbox.addWidget(self.hist_canvas, alignment=Qt.AlignHCenter)
        vbox.addWidget(ctrl_container, alignment=Qt.AlignHCenter)
        
        container = QWidget()
        container.setLayout(vbox)
        self.setCentralWidget(container)

        # ---------- signals ------------
        self.start_btn.clicked.connect(self.start_preview)
        self.stop_btn.clicked.connect(self.stop_preview)
        self.save_btn.clicked.connect(self.save_frame)
        self.acquire_btn.clicked.connect(self.acquire_stack)

        self.gain_ed.valueChanged.connect(self.worker.set_gain)
        self.offset_ed.valueChanged.connect(self.worker.set_offset)
        self.tec_ed.valueChanged.connect(self.worker.set_tec)
        self.tec_button.clicked.connect(self.toggle_tec)
        self.exp_ed.valueChanged.connect(self.update_exposure)
        self.exp_unit.currentTextChanged.connect(self.update_exposure)

        # worker thread will be started on first "Start Preview" click
        # self.worker.start(); self._last_frame = None

    # ---------- preview control --------
    def start_preview(self):
        if not self.worker.isRunning():
            self.worker.start()
        self.worker.start_stream()
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True); self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)

    def stop_preview(self):
        self.worker.stop_stream(); self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        

    # ---------- frame slot -------------
    @Slot(np.ndarray)
    def on_new_frame(self, img: np.ndarray):
        # --- contrast stretch (AstroPy zscale) ---
        vmin, vmax = self._auto_scale(img)
    
        img8 = ((np.clip(img, vmin, vmax) - vmin) /
                (vmax - vmin + 1e-8) * 255).astype(np.uint8)
    
        h, w = img8.shape
        qimg = QImage(img8.data, w, h, w, QImage.Format_Grayscale8)
    
        # scale to the label’s fixed size (600 × 600)
        pix = QPixmap.fromImage(qimg).scaled(
            self.frame_lbl.size(),           # (600, 600)
            Qt.KeepAspectRatio,
            Qt.FastTransformation
        )
        self.frame_lbl.setPixmap(pix)
    
        # update histogram (throttled inside the canvas)
        self.hist_canvas.update_histogram(img)

    def _auto_scale(self, img):
        try:
            return ZScaleInterval().get_limits(img)
        except Exception:
            return img.min(), img.max()

    # ---------- actions -----------------
    def save_frame(self):
        if self._last_frame is None:
            return
        fname,_ = QFileDialog.getSaveFileName(self, "Save FITS", f"frame_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.fits")
        if fname:
            from astropy.io import fits; fits.writeto(fname, self._last_frame, overwrite=True)

    def acquire_stack(self):
        n = self.n_frames_ed.value()
        fname,_ = QFileDialog.getSaveFileName(self, "Save FITS stack", f"stack_{n}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.fits")
        if not fname:
            return
        self.stop_preview()
        self.camera.capture_stream_stack(n_frames=n, fname=Path(fname).name)
        self.start_preview()

    def toggle_tec(self):
        """Toggle cooler on/off and update button text."""
        self.tec_enabled = not self.tec_enabled
        self.camera.set_cooler_enabled(self.tec_enabled)
        self.tec_button.setText("Disable TEC" if self.tec_enabled else "Enable TEC")

    def update_exposure(self, *_):
        """Convert exposure value + selected unit to µs and send to camera."""
        val = self.exp_ed.value()
        unit = self.exp_unit.currentText()
        if unit == "us":
            us = val
        elif unit == "ms":
            us = val * 1_000
        else:  # seconds
            us = val * 1_000_000
        self.camera.set_exposure_us(us)

    # ---------------- cleanup -------------------------------------------
    def closeEvent(self, e):
        self.worker.stop_thread()
        super().closeEvent(e)

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow(); win.resize(820, 980); win.show()
    sys.exit(app.exec())
