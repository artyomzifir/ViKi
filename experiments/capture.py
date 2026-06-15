"""
capture.py — RealSense video capture tool
Run: python3 capture.py

Requirements: pyrealsense2, opencv-python, Pillow
  pip install pyrealsense2 opencv-python Pillow
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading
import time
import os
import datetime
import numpy as np

try:
    import pyrealsense2 as rs
    RS_AVAILABLE = True
except ImportError:
    RS_AVAILABLE = False

try:
    import cv2
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ── Config ───────────────────────────────────────────────────────────────────

PREVIEW_W   = 640
PREVIEW_H   = 360
LOG_MAX     = 500     # max lines in log widget
FPS_OPTIONS = [6, 15, 30, 60]
RES_OPTIONS = ["1280x720", "848x480", "640x480", "424x240"]

BG          = "#1a1a2e"
PANEL       = "#16213e"
ACCENT      = "#0f3460"
GREEN       = "#00b894"
RED         = "#d63031"
AMBER       = "#fdcb6e"
TEXT        = "#dfe6e9"
TEXT_DIM    = "#636e72"
FONT_MONO   = ("Courier", 9)
FONT_UI     = ("Helvetica", 10)
FONT_TITLE  = ("Helvetica", 13, "bold")


# ── Main App ─────────────────────────────────────────────────────────────────

class CaptureApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ViKi — RealSense Capture")
        self.configure(bg=BG)
        self.resizable(False, False)

        # State
        self.pipeline     = None
        self.recording    = False
        self.streaming    = False
        self.video_writer = None
        self.output_dir   = os.path.expanduser("~/recordings")
        self.frame_count  = 0
        self.rec_start    = None
        self.preview_img  = None
        self._stop_event  = threading.Event()
        self._stream_thread = None

        self._build_ui()
        self._check_deps()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Left column: controls ─────────────────────────────────────────
        left = tk.Frame(self, bg=BG, padx=12, pady=12)
        left.grid(row=0, column=0, sticky="nsew")

        tk.Label(left, text="ViKi Capture", bg=BG, fg=TEXT,
                 font=FONT_TITLE).pack(anchor="w", pady=(0, 12))

        # Device
        self._section(left, "DEVICE")
        self.device_var = tk.StringVar(value="Scanning...")
        self.device_cb  = ttk.Combobox(left, textvariable=self.device_var,
                                        state="readonly", width=34, font=FONT_UI)
        self.device_cb.pack(fill="x", pady=(0, 4))
        tk.Button(left, text="⟳ Refresh", command=self._refresh_devices,
                  bg=ACCENT, fg=TEXT, font=FONT_UI, relief="flat",
                  cursor="hand2").pack(fill="x", pady=(0, 10))

        # Stream settings
        self._section(left, "STREAM")
        row = tk.Frame(left, bg=BG); row.pack(fill="x", pady=(0, 4))
        tk.Label(row, text="Resolution", bg=BG, fg=TEXT_DIM,
                 font=FONT_UI, width=10, anchor="w").pack(side="left")
        self.res_var = tk.StringVar(value="848x480")
        ttk.Combobox(row, textvariable=self.res_var, values=RES_OPTIONS,
                     state="readonly", width=12, font=FONT_UI).pack(side="left")

        row2 = tk.Frame(left, bg=BG); row2.pack(fill="x", pady=(0, 10))
        tk.Label(row2, text="FPS", bg=BG, fg=TEXT_DIM,
                 font=FONT_UI, width=10, anchor="w").pack(side="left")
        self.fps_var = tk.IntVar(value=30)
        ttk.Combobox(row2, textvariable=self.fps_var, values=FPS_OPTIONS,
                     state="readonly", width=12, font=FONT_UI).pack(side="left")

        # Streams to enable
        self._section(left, "CHANNELS")
        self.en_color = tk.BooleanVar(value=True)
        self.en_depth = tk.BooleanVar(value=True)
        tk.Checkbutton(left, text="Color (RGB)", variable=self.en_color,
                       bg=BG, fg=TEXT, selectcolor=ACCENT, font=FONT_UI,
                       activebackground=BG, activeforeground=TEXT).pack(anchor="w")
        tk.Checkbutton(left, text="Depth", variable=self.en_depth,
                       bg=BG, fg=TEXT, selectcolor=ACCENT, font=FONT_UI,
                       activebackground=BG, activeforeground=TEXT).pack(anchor="w", pady=(0, 10))

        # Output folder
        self._section(left, "OUTPUT")
        self.dir_var = tk.StringVar(value=self.output_dir)
        dir_row = tk.Frame(left, bg=BG); dir_row.pack(fill="x", pady=(0, 10))
        tk.Entry(dir_row, textvariable=self.dir_var, bg=PANEL, fg=TEXT,
                 font=FONT_MONO, insertbackground=TEXT, relief="flat",
                 width=26).pack(side="left", ipady=4)
        tk.Button(dir_row, text="…", command=self._pick_dir,
                  bg=ACCENT, fg=TEXT, font=FONT_UI, relief="flat",
                  cursor="hand2", width=3).pack(side="left", padx=(4, 0))

        # Start / Stop stream
        self._section(left, "STREAM CONTROL")
        self.stream_btn = tk.Button(left, text="▶  Start Preview",
                                    command=self._toggle_stream,
                                    bg=GREEN, fg="#fff", font=("Helvetica", 11, "bold"),
                                    relief="flat", cursor="hand2", pady=6)
        self.stream_btn.pack(fill="x", pady=(0, 6))

        # Record
        self.rec_btn = tk.Button(left, text="⏺  Record",
                                 command=self._toggle_record,
                                 bg=PANEL, fg=TEXT_DIM,
                                 font=("Helvetica", 11, "bold"),
                                 relief="flat", cursor="hand2", pady=6,
                                 state="disabled")
        self.rec_btn.pack(fill="x", pady=(0, 10))

        # Status bar
        self._section(left, "STATUS")
        self.status_lbl = tk.Label(left, text="●  Idle", bg=BG, fg=TEXT_DIM,
                                   font=FONT_MONO, anchor="w")
        self.status_lbl.pack(anchor="w")
        self.timer_lbl  = tk.Label(left, text="", bg=BG, fg=AMBER,
                                   font=FONT_MONO, anchor="w")
        self.timer_lbl.pack(anchor="w")
        self.fps_lbl    = tk.Label(left, text="", bg=BG, fg=TEXT_DIM,
                                   font=FONT_MONO, anchor="w")
        self.fps_lbl.pack(anchor="w", pady=(0, 10))

        # ── Right column: preview + log ───────────────────────────────────
        right = tk.Frame(self, bg=BG, padx=0, pady=12)
        right.grid(row=0, column=1, sticky="nsew")

        # Preview canvas
        preview_frame = tk.Frame(right, bg=PANEL, bd=0)
        preview_frame.pack(padx=(0, 12))
        self.canvas = tk.Canvas(preview_frame, width=PREVIEW_W, height=PREVIEW_H,
                                bg="#000", highlightthickness=0)
        self.canvas.pack()
        self._draw_placeholder()

        # Depth toggle
        self.show_depth = tk.BooleanVar(value=False)
        tk.Checkbutton(right, text="Show depth preview", variable=self.show_depth,
                       bg=BG, fg=TEXT_DIM, selectcolor=ACCENT, font=FONT_UI,
                       activebackground=BG, activeforeground=TEXT).pack(
                       anchor="w", padx=(0, 12), pady=(4, 6))

        # Log
        log_hdr = tk.Frame(right, bg=BG); log_hdr.pack(fill="x", padx=(0, 12))
        tk.Label(log_hdr, text="LOG", bg=BG, fg=TEXT_DIM,
                 font=("Helvetica", 8, "bold")).pack(side="left")
        tk.Button(log_hdr, text="Clear", command=self._clear_log,
                  bg=BG, fg=TEXT_DIM, font=("Helvetica", 8),
                  relief="flat", cursor="hand2").pack(side="right")

        self.log = scrolledtext.ScrolledText(
            right, width=76, height=10, bg=PANEL, fg=TEXT,
            font=FONT_MONO, relief="flat", state="disabled",
            insertbackground=TEXT)
        self.log.pack(padx=(0, 12), pady=(2, 0))

        # Style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=PANEL, background=PANEL,
                        foreground=TEXT, selectbackground=ACCENT,
                        selectforeground=TEXT, borderwidth=0)

        self.after(200, self._refresh_devices)

    def _section(self, parent, text):
        tk.Label(parent, text=text, bg=BG, fg=TEXT_DIM,
                 font=("Helvetica", 8, "bold")).pack(anchor="w", pady=(6, 2))

    def _draw_placeholder(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, PREVIEW_W, PREVIEW_H, fill="#000")
        self.canvas.create_text(PREVIEW_W//2, PREVIEW_H//2,
                                text="No preview", fill=TEXT_DIM,
                                font=("Helvetica", 14))

    # ── Dependency check ──────────────────────────────────────────────────────

    def _check_deps(self):
        if not RS_AVAILABLE:
            self._log("ERROR: pyrealsense2 not installed — pip install pyrealsense2", "err")
        if not CV_AVAILABLE:
            self._log("ERROR: opencv not installed — pip install opencv-python", "err")
        if not PIL_AVAILABLE:
            self._log("ERROR: Pillow not installed — pip install Pillow", "err")
        if RS_AVAILABLE and CV_AVAILABLE and PIL_AVAILABLE:
            self._log("Dependencies OK")

    # ── Device enumeration ────────────────────────────────────────────────────

    def _refresh_devices(self):
        self._log("Scanning for RealSense devices...")
        if not RS_AVAILABLE:
            self.device_cb["values"] = []
            self.device_var.set("pyrealsense2 not installed")
            return

        ctx     = rs.context()
        devices = ctx.query_devices()
        entries = []
        for dev in devices:
            serial = dev.get_info(rs.camera_info.serial_number)
            name   = dev.get_info(rs.camera_info.name)
            entries.append(f"{name}  [{serial}]")

        if entries:
            self.device_cb["values"] = entries
            self.device_var.set(entries[0])
            self._log(f"Found {len(entries)} device(s): {', '.join(entries)}")
        else:
            self.device_cb["values"] = ["No device found"]
            self.device_var.set("No device found")
            self._log("No RealSense device found — check USB connection")

    def _selected_serial(self):
        """Extract serial number from combobox entry like 'Intel... [123456789]'."""
        val = self.device_var.get()
        if "[" in val and "]" in val:
            return val.split("[")[-1].rstrip("]")
        return None

    # ── Stream control ────────────────────────────────────────────────────────

    def _toggle_stream(self):
        if not self.streaming:
            self._start_stream()
        else:
            self._stop_stream()

    def _start_stream(self):
        if not RS_AVAILABLE or not CV_AVAILABLE or not PIL_AVAILABLE:
            self._log("Cannot stream — missing dependencies", "err")
            return

        serial = self._selected_serial()
        if not serial:
            self._log("No device selected", "warn")
            return

        res_str = self.res_var.get()
        w, h    = map(int, res_str.split("x"))
        fps     = int(self.fps_var.get())

        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            if self.en_color.get():
                config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
            if self.en_depth.get():
                config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
            self.pipeline.start(config)
        except Exception as e:
            self._log(f"Failed to start pipeline: {e}", "err")
            self.pipeline = None
            return

        self.streaming   = True
        self._stop_event.clear()
        self._fps_times  = []
        self.stream_btn.configure(text="■  Stop Preview", bg=RED)
        self.rec_btn.configure(state="normal", bg=ACCENT, fg=TEXT)
        self._log(f"Stream started — {res_str} @ {fps} fps  serial={serial}")

        self._stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._stream_thread.start()
        self._update_timer()

    def _stop_stream(self):
        if self.recording:
            self._stop_record()
        self._stop_event.set()
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
        self.streaming = False
        self.stream_btn.configure(text="▶  Start Preview", bg=GREEN)
        self.rec_btn.configure(state="disabled", bg=PANEL, fg=TEXT_DIM)
        self._draw_placeholder()
        self.status_lbl.configure(text="●  Idle", fg=TEXT_DIM)
        self.timer_lbl.configure(text="")
        self.fps_lbl.configure(text="")
        self._log("Stream stopped")

    def _stream_loop(self):
        align = rs.align(rs.stream.color) if self.en_color.get() else None
        while not self._stop_event.is_set():
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=2000)
            except Exception as e:
                self._log(f"Frame timeout: {e}", "warn")
                continue

            if align and self.en_color.get() and self.en_depth.get():
                frames = align.process(frames)

            color_frame = frames.get_color_frame() if self.en_color.get() else None
            depth_frame = frames.get_depth_frame() if self.en_depth.get() else None

            color_img = None
            depth_img = None

            if color_frame:
                color_img = np.asanyarray(color_frame.get_data())   # BGR

            if depth_frame:
                depth_raw  = np.asanyarray(depth_frame.get_data())  # uint16 mm
                depth_norm = cv2.normalize(depth_raw, None, 0, 255,
                                           cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_img  = cv2.applyColorMap(depth_norm, cv2.COLORMAP_PLASMA)

            # Write to file if recording
            if self.recording and self.video_writer and color_img is not None:
                self.video_writer.write(color_img)
                self.frame_count += 1

            # Update preview
            if self.show_depth.get() and depth_img is not None:
                preview = depth_img
            elif color_img is not None:
                preview = color_img
            else:
                continue

            self._push_frame(preview)

        # Thread exits here

    def _push_frame(self, bgr):
        """Convert BGR numpy array → PhotoImage and schedule canvas update."""
        h, w = bgr.shape[:2]
        # Fit inside PREVIEW_W x PREVIEW_H maintaining aspect ratio
        scale  = min(PREVIEW_W / w, PREVIEW_H / h)
        nw, nh = int(w * scale), int(h * scale)
        small  = cv2.resize(bgr, (nw, nh))
        rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        img    = Image.fromarray(rgb)
        self.preview_img = ImageTk.PhotoImage(img)   # keep reference
        self.after(0, self._draw_frame, self.preview_img, nw, nh)

        # FPS tracking
        now = time.monotonic()
        self._fps_times.append(now)
        self._fps_times = [t for t in self._fps_times if now - t < 2.0]

    def _draw_frame(self, photo, nw, nh):
        x = (PREVIEW_W - nw) // 2
        y = (PREVIEW_H - nh) // 2
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, PREVIEW_W, PREVIEW_H, fill="#000")
        self.canvas.create_image(x, y, anchor="nw", image=photo)
        if self.recording:
            self.canvas.create_oval(10, 10, 24, 24, fill=RED, outline="")
            self.canvas.create_text(32, 17, text="REC", fill=RED,
                                    font=("Helvetica", 9, "bold"), anchor="w")

    # ── Record control ────────────────────────────────────────────────────────

    def _toggle_record(self):
        if not self.recording:
            self._start_record()
        else:
            self._stop_record()

    def _start_record(self):
        out_dir = self.dir_var.get().strip()
        os.makedirs(out_dir, exist_ok=True)

        ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(out_dir, f"capture_{ts}.mp4")

        res_str  = self.res_var.get()
        w, h     = map(int, res_str.split("x"))
        fps      = int(self.fps_var.get())

        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
        if not self.video_writer.isOpened():
            self._log(f"Failed to open video writer: {filename}", "err")
            self.video_writer = None
            return

        self.recording   = True
        self.frame_count = 0
        self.rec_start   = time.monotonic()
        self.rec_btn.configure(text="⏹  Stop Recording", bg=RED)
        self.status_lbl.configure(text="⏺  Recording", fg=RED)
        self._log(f"Recording started → {filename}")
        self._current_file = filename

    def _stop_record(self):
        self.recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        duration = time.monotonic() - self.rec_start if self.rec_start else 0
        self._log(f"Recording stopped — {self.frame_count} frames  "
                  f"({duration:.1f}s)  → {self._current_file}")

        self.rec_btn.configure(text="⏺  Record", bg=ACCENT, fg=TEXT)
        self.status_lbl.configure(text="●  Streaming", fg=GREEN)
        self.timer_lbl.configure(text="")
        self.rec_start = None

    # ── Timer & status updates ────────────────────────────────────────────────

    def _update_timer(self):
        if not self.streaming:
            return
        # FPS display
        if hasattr(self, "_fps_times") and self._fps_times:
            fps_actual = len(self._fps_times) / 2.0
            self.fps_lbl.configure(text=f"fps  {fps_actual:.1f}")

        # Recording timer
        if self.recording and self.rec_start:
            elapsed = time.monotonic() - self.rec_start
            m, s    = divmod(int(elapsed), 60)
            self.timer_lbl.configure(
                text=f"⏺  {m:02d}:{s:02d}  {self.frame_count} frames")
            self.status_lbl.configure(text="⏺  Recording", fg=RED)
        elif self.streaming:
            self.status_lbl.configure(text="●  Streaming", fg=GREEN)

        self.after(500, self._update_timer)

    # ── Output folder ─────────────────────────────────────────────────────────

    def _pick_dir(self):
        d = filedialog.askdirectory(initialdir=self.dir_var.get())
        if d:
            self.dir_var.set(d)
            self._log(f"Output folder: {d}")

    # ── Log ───────────────────────────────────────────────────────────────────

    def _log(self, msg, level="info"):
        ts    = datetime.datetime.now().strftime("%H:%M:%S")
        color = {"info": TEXT, "warn": AMBER, "err": RED}.get(level, TEXT)
        tag   = f"l{level}"

        self.log.configure(state="normal")
        self.log.insert("end", f"[{ts}]  {msg}\n", tag)
        self.log.tag_configure(tag, foreground=color)

        # Trim if too long
        lines = int(self.log.index("end-1c").split(".")[0])
        if lines > LOG_MAX:
            self.log.delete("1.0", f"{lines - LOG_MAX}.0")

        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def _on_close(self):
        self._stop_stream()
        self.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = CaptureApp()
    app.mainloop()