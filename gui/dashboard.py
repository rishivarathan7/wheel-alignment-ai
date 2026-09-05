"""
Desktop Monitoring Dashboard Application.

Tkinter desktop GUI application providing live video analytics preview, per-wheel telemetry metrics,
alert notifications console, interactive camera/video controls, and settings configuration dialog.
"""

from datetime import datetime
import logging
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np
from PIL import Image, ImageTk

from src.config import ConfigManager, LOGS_DIR
from src.pipeline import WheelAlignmentPipeline
from src.video_capture import VideoCaptureManager
from src.alert_system import MANDATORY_WARNING_TEXT, NON_DIAGNOSTIC_DISCLAIMER

logger = logging.getLogger(__name__)

# Dark Modern Color Palette
DARK_BG = "#121212"
CARD_BG = "#1e1e1e"
PANEL_BG = "#252526"
ACCENT_BLUE = "#00adb5"
TEXT_WHITE = "#ffffff"
TEXT_MUTED = "#aaaaaa"

STATE_COLORS = {
    "NORMAL": "#00e676",               # Vibrant Green
    "POSSIBLE_MISALIGNMENT": "#ffd600",  # Vivid Yellow
    "SEVERE_MISALIGNMENT": "#ff1744",    # Alarm Red
    "INSUFFICIENT_DATA": "#9e9e9e",      # Neutral Gray
    "IDLE": "#757575"
}


class SettingsDialog(tk.Toplevel):
    """Modal dialog for configuring pipeline settings."""

    def __init__(self, parent: tk.Tk, config_manager: ConfigManager) -> None:
        super().__init__(parent)
        self.title("Pipeline Configuration Settings")
        self.geometry("450x520")
        self.resizable(False, False)
        self.configure(bg=DARK_BG)
        self.transient(parent)
        self.grab_set()

        self.config = config_manager
        self._build_ui()

    def _build_ui(self) -> None:
        lbl_title = tk.Label(
            self,
            text="⚙️ Pipeline Settings",
            font=("Segoe UI", 14, "bold"),
            bg=DARK_BG,
            fg=ACCENT_BLUE
        )
        lbl_title.pack(anchor=tk.W, padx=20, pady=(15, 10))

        frame_fields = tk.Frame(self, bg=CARD_BG, padx=15, pady=15)
        frame_fields.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)

        # 1. Detection Confidence
        tk.Label(frame_fields, text="Detection Confidence Threshold:", font=("Segoe UI", 9), bg=CARD_BG, fg=TEXT_WHITE).grid(row=0, column=0, sticky="w", pady=8)
        self.scale_det_conf = tk.Scale(frame_fields, from_=0.1, to=0.9, resolution=0.05, orient=tk.HORIZONTAL, bg=CARD_BG, fg=TEXT_WHITE, highlightthickness=0)
        self.scale_det_conf.set(self.config.get("detection.confidence_threshold", 0.5))
        self.scale_det_conf.grid(row=0, column=1, sticky="ew", padx=10)

        # 2. Min Prediction Confidence
        tk.Label(frame_fields, text="Min Prediction Confidence:", font=("Segoe UI", 9), bg=CARD_BG, fg=TEXT_WHITE).grid(row=1, column=0, sticky="w", pady=8)
        self.scale_pred_conf = tk.Scale(frame_fields, from_=0.3, to=0.95, resolution=0.05, orient=tk.HORIZONTAL, bg=CARD_BG, fg=TEXT_WHITE, highlightthickness=0)
        self.scale_pred_conf.set(self.config.get("alert.min_confidence", 0.60))
        self.scale_pred_conf.grid(row=1, column=1, sticky="ew", padx=10)

        # 3. Consecutive Alert Count Threshold
        tk.Label(frame_fields, text="Consecutive Alert Threshold:", font=("Segoe UI", 9), bg=CARD_BG, fg=TEXT_WHITE).grid(row=2, column=0, sticky="w", pady=8)
        self.spin_consecutive = ttk.Spinbox(frame_fields, from_=1, to=10, width=8)
        self.spin_consecutive.set(self.config.get("alert.consecutive_threshold", 3))
        self.spin_consecutive.grid(row=2, column=1, sticky="w", padx=10)

        # 4. Alert Cooldown Seconds
        tk.Label(frame_fields, text="Alert Cooldown (seconds):", font=("Segoe UI", 9), bg=CARD_BG, fg=TEXT_WHITE).grid(row=3, column=0, sticky="w", pady=8)
        self.spin_cooldown = ttk.Spinbox(frame_fields, from_=1, to=20, width=8)
        self.spin_cooldown.set(int(self.config.get("alert.cooldown_seconds", 3.0)))
        self.spin_cooldown.grid(row=3, column=1, sticky="w", padx=10)

        # 5. Audio Alert Toggle
        self.var_audio = tk.BooleanVar(value=self.config.get("alert.enable_audio", False))
        chk_audio = tk.Checkbutton(frame_fields, text="Enable Audio Alerts", variable=self.var_audio, font=("Segoe UI", 9), bg=CARD_BG, fg=TEXT_WHITE, selectcolor=DARK_BG, activebackground=CARD_BG, activeforeground=TEXT_WHITE)
        chk_audio.grid(row=4, column=0, columnspan=2, sticky="w", pady=8)

        # 6. Pipeline Debug Mode Toggle
        self.var_debug = tk.BooleanVar(value=self.config.get("pipeline.debug_mode", False))
        chk_debug = tk.Checkbutton(frame_fields, text="Enable Visual Debug HUD Mode", variable=self.var_debug, font=("Segoe UI", 9), bg=CARD_BG, fg=TEXT_WHITE, selectcolor=DARK_BG, activebackground=CARD_BG, activeforeground=TEXT_WHITE)
        chk_debug.grid(row=5, column=0, columnspan=2, sticky="w", pady=8)

        # Buttons Frame
        frame_btns = tk.Frame(self, bg=DARK_BG)
        frame_btns.pack(fill=tk.X, padx=20, pady=15)

        btn_save = tk.Button(frame_btns, text="💾 Save Settings", font=("Segoe UI", 9, "bold"), bg=ACCENT_BLUE, fg="white", relief=tk.FLAT, padx=15, pady=5, command=self._save_settings)
        btn_save.pack(side=tk.RIGHT, padx=5)

        btn_cancel = tk.Button(frame_btns, text="Cancel", font=("Segoe UI", 9), bg="#424242", fg="white", relief=tk.FLAT, padx=15, pady=5, command=self.destroy)
        btn_cancel.pack(side=tk.RIGHT, padx=5)

    def _save_settings(self) -> None:
        try:
            self.config.set("detection.confidence_threshold", float(self.scale_det_conf.get()))
            self.config.set("alert.min_confidence", float(self.scale_pred_conf.get()))
            self.config.set("alert.consecutive_threshold", int(self.spin_consecutive.get()))
            self.config.set("alert.cooldown_seconds", float(self.spin_cooldown.get()))
            self.config.set("alert.enable_audio", bool(self.var_audio.get()))
            self.config.set("pipeline.debug_mode", bool(self.var_debug.get()))

            self.config.save()
            messagebox.showinfo("Settings Saved", "Pipeline configuration settings successfully updated!")
            self.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")


class DashboardApp:
    """
    AI Wheel Alignment Monitoring System Desktop Application.
    Non-blocking Tkinter dashboard executing computer vision background worker pipeline.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AI Wheel Alignment Monitoring System")
        self.root.geometry("1280x800")
        self.root.minsize(1024, 650)
        self.root.configure(bg=DARK_BG)

        # Initialize Config & Pipeline
        self.config = ConfigManager()
        self.pipeline = WheelAlignmentPipeline(config_manager=self.config)

        # Application State
        self.video_source: Union[int, str] = self.config.get("video.source", 0)
        self.is_streaming = False
        self.stream_thread: Optional[threading.Thread] = None

        # Thread-safe queue for frame & telemetry updates
        self.update_queue: queue.Queue = queue.Queue(maxsize=2)

        self._build_ui()
        self._start_queue_poller()

        # Handle window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

    def _build_ui(self) -> None:
        """Constructs responsive dark-themed dashboard layout."""

        # -------------------------------------------------------------
        # 1. Header Toolbar
        # -------------------------------------------------------------
        header_frame = tk.Frame(self.root, bg=CARD_BG, height=60, padx=15, pady=10)
        header_frame.pack(fill=tk.X, side=tk.TOP)

        lbl_logo = tk.Label(
            header_frame,
            text="🚗 AI WHEEL ALIGNMENT MONITOR",
            font=("Segoe UI", 13, "bold"),
            bg=CARD_BG,
            fg=ACCENT_BLUE
        )
        lbl_logo.pack(side=tk.LEFT, padx=(0, 20))

        # Camera / Source Selection Dropdown
        tk.Label(header_frame, text="Source:", font=("Segoe UI", 9), bg=CARD_BG, fg=TEXT_MUTED).pack(side=tk.LEFT, padx=(10, 5))
        self.combo_source = ttk.Combobox(
            header_frame,
            values=["Camera 0 (Default)", "Camera 1", "Camera 2", "📁 Select Video File..."],
            state="readonly",
            width=22
        )
        self.combo_source.current(0)
        self.combo_source.pack(side=tk.LEFT, padx=5)
        self.combo_source.bind("<<ComboboxSelected>>", self._on_source_selected)

        # Action Control Buttons
        self.btn_start = tk.Button(
            header_frame,
            text="▶ Start Monitor",
            font=("Segoe UI", 9, "bold"),
            bg="#00c853",
            fg="white",
            relief=tk.FLAT,
            padx=12,
            pady=4,
            command=self.start_stream
        )
        self.btn_start.pack(side=tk.LEFT, padx=5)

        self.btn_stop = tk.Button(
            header_frame,
            text="⏹ Stop Monitor",
            font=("Segoe UI", 9, "bold"),
            bg="#d50000",
            fg="white",
            relief=tk.FLAT,
            padx=12,
            pady=4,
            state=tk.DISABLED,
            command=self.stop_stream
        )
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        btn_settings = tk.Button(
            header_frame,
            text="⚙️ Settings",
            font=("Segoe UI", 9),
            bg="#424242",
            fg="white",
            relief=tk.FLAT,
            padx=10,
            pady=4,
            command=self.open_settings
        )
        btn_settings.pack(side=tk.LEFT, padx=5)

        btn_exit = tk.Button(
            header_frame,
            text="❌ Exit",
            font=("Segoe UI", 9),
            bg="#37474f",
            fg="white",
            relief=tk.FLAT,
            padx=10,
            pady=4,
            command=self.on_exit
        )
        btn_exit.pack(side=tk.RIGHT, padx=5)

        # -------------------------------------------------------------
        # 2. Main Content Split View (Video Feed Left, Telemetry Right)
        # -------------------------------------------------------------
        main_split = tk.Frame(self.root, bg=DARK_BG, padx=10, pady=10)
        main_split.pack(fill=tk.BOTH, expand=True)

        # --- LEFT PANEL: Live Video Feed & Performance HUD Bar ---
        left_panel = tk.Frame(main_split, bg=CARD_BG, padx=10, pady=10)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        lbl_video_hdr = tk.Label(
            left_panel,
            text="LIVE VIDEO ANALYTICS PREVIEW",
            font=("Segoe UI", 10, "bold"),
            bg=CARD_BG,
            fg=TEXT_MUTED
        )
        lbl_video_hdr.pack(anchor=tk.W, pady=(0, 5))

        self.canvas_video = tk.Canvas(left_panel, bg="black", highlightthickness=0)
        self.canvas_video.pack(fill=tk.BOTH, expand=True)

        # Sub-video Performance HUD Bar
        hud_bar = tk.Frame(left_panel, bg=PANEL_BG, padx=10, pady=5)
        hud_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))

        self.lbl_fps = tk.Label(hud_bar, text="FPS: 0.0", font=("Consolas", 9, "bold"), bg=PANEL_BG, fg=ACCENT_BLUE)
        self.lbl_fps.pack(side=tk.LEFT, padx=10)

        self.lbl_total_lat = tk.Label(hud_bar, text="Latency: 0.0 ms", font=("Consolas", 9), bg=PANEL_BG, fg=TEXT_WHITE)
        self.lbl_total_lat.pack(side=tk.LEFT, padx=10)

        self.lbl_det_lat = tk.Label(hud_bar, text="Det: 0.0 ms", font=("Consolas", 9), bg=PANEL_BG, fg=TEXT_MUTED)
        self.lbl_det_lat.pack(side=tk.LEFT, padx=10)

        self.lbl_ml_lat = tk.Label(hud_bar, text="ML: 0.0 ms", font=("Consolas", 9), bg=PANEL_BG, fg=TEXT_MUTED)
        self.lbl_ml_lat.pack(side=tk.LEFT, padx=10)

        self.lbl_model_status = tk.Label(hud_bar, text="Model: RandomForest (Active)", font=("Consolas", 9), bg=PANEL_BG, fg="#00e676")
        self.lbl_model_status.pack(side=tk.RIGHT, padx=10)

        # --- RIGHT PANEL: Alert Status, Active Wheels Table, Event Log ---
        right_panel = tk.Frame(main_split, bg=DARK_BG, width=420)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(5, 0))

        # A. System Alert Status Banner Box
        self.alert_box = tk.Frame(right_panel, bg=CARD_BG, padx=15, pady=12)
        self.alert_box.pack(fill=tk.X, pady=(0, 10))

        tk.Label(self.alert_box, text="SYSTEM ALIGNMENT STATUS", font=("Segoe UI", 9, "bold"), bg=CARD_BG, fg=TEXT_MUTED).pack(anchor=tk.W)

        self.lbl_global_status = tk.Label(
            self.alert_box,
            text="IDLE",
            font=("Segoe UI", 16, "bold"),
            bg=CARD_BG,
            fg=STATE_COLORS["IDLE"]
        )
        self.lbl_global_status.pack(anchor=tk.W, pady=(2, 5))

        self.lbl_warning_text = tk.Label(
            self.alert_box,
            text="System ready for real-time monitoring.",
            font=("Segoe UI", 9),
            bg=CARD_BG,
            fg=TEXT_WHITE,
            wraplength=380,
            justify=tk.LEFT
        )
        self.lbl_warning_text.pack(anchor=tk.W)

        self.lbl_disclaimer = tk.Label(
            self.alert_box,
            text=f"NOTICE: {NON_DIAGNOSTIC_DISCLAIMER}",
            font=("Segoe UI", 8, "italic"),
            bg=CARD_BG,
            fg=TEXT_MUTED,
            wraplength=380,
            justify=tk.LEFT
        )
        self.lbl_disclaimer.pack(anchor=tk.W, pady=(6, 0))

        # B. Active Wheel Telemetry Table Box
        table_box = tk.Frame(right_panel, bg=CARD_BG, padx=10, pady=10)
        table_box.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        tk.Label(table_box, text="ACTIVE WHEELS TELEMETRY", font=("Segoe UI", 9, "bold"), bg=CARD_BG, fg=TEXT_MUTED).pack(anchor=tk.W, pady=(0, 5))

        # Treeview for Wheel Telemetry
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#2a2a2a", foreground="white", fieldbackground="#2a2a2a", rowheight=24, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", background="#333333", foreground="white", font=("Segoe UI", 9, "bold"))
        style.map("Treeview", background=[("selected", ACCENT_BLUE)])

        cols = ("id", "status", "conf", "camber", "toe", "speed")
        self.tree_telemetry = ttk.Treeview(table_box, columns=cols, show="headings", height=6)
        self.tree_telemetry.heading("id", text="ID")
        self.tree_telemetry.heading("status", text="Status")
        self.tree_telemetry.heading("conf", text="Conf")
        self.tree_telemetry.heading("camber", text="Camber")
        self.tree_telemetry.heading("toe", text="Toe")
        self.tree_telemetry.heading("speed", text="Speed")

        self.tree_telemetry.column("id", width=35, anchor=tk.CENTER)
        self.tree_telemetry.column("status", width=110, anchor=tk.CENTER)
        self.tree_telemetry.column("conf", width=50, anchor=tk.CENTER)
        self.tree_telemetry.column("camber", width=60, anchor=tk.CENTER)
        self.tree_telemetry.column("toe", width=50, anchor=tk.CENTER)
        self.tree_telemetry.column("speed", width=55, anchor=tk.CENTER)

        self.tree_telemetry.pack(fill=tk.BOTH, expand=True)

        # C. Real-Time Event Log Console
        log_box = tk.Frame(right_panel, bg=CARD_BG, padx=10, pady=10)
        log_box.pack(fill=tk.BOTH, expand=True)

        tk.Label(log_box, text="REAL-TIME SYSTEM EVENT LOG", font=("Segoe UI", 9, "bold"), bg=CARD_BG, fg=TEXT_MUTED).pack(anchor=tk.W, pady=(0, 5))

        self.txt_log = tk.Text(log_box, height=8, bg="#1a1a1a", fg="#00ff00", font=("Consolas", 8), state=tk.DISABLED, relief=tk.FLAT)
        self.txt_log.pack(fill=tk.BOTH, expand=True)

        self.log_event("AI Wheel Alignment Monitoring System initialized.")

    def log_event(self, msg: str, level: str = "INFO") -> None:
        """Appends timestamped message to event log console."""
        ts = datetime.now().strftime("%H:%M:%S")
        prefix = f"[{ts}] [{level}] "
        full_msg = f"{prefix}{msg}\n"

        self.txt_log.config(state=tk.NORMAL)
        self.txt_log.insert(tk.END, full_msg)
        self.txt_log.see(tk.END)
        self.txt_log.config(state=tk.DISABLED)

    def _on_source_selected(self, event: Any) -> None:
        """Handles source selection dropdown changes."""
        selection = self.combo_source.get()
        if "Select Video File" in selection:
            file_path = filedialog.askopenfilename(
                title="Select Video File for Monitoring",
                filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
            )
            if file_path:
                self.video_source = file_path
                self.log_event(f"Selected video file: {Path(file_path).name}")
            else:
                self.combo_source.current(0)
                self.video_source = 0
        elif "Camera 1" in selection:
            self.video_source = 1
            self.log_event("Selected Camera 1.")
        elif "Camera 2" in selection:
            self.video_source = 2
            self.log_event("Selected Camera 2.")
        else:
            self.video_source = 0
            self.log_event("Selected Camera 0 (Default).")

    def open_settings(self) -> None:
        """Opens pipeline configuration settings dialog."""
        SettingsDialog(self.root, self.config)

    def start_stream(self) -> None:
        """Starts real-time monitoring worker thread."""
        if self.is_streaming:
            return

        self.is_streaming = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)

        # Re-initialize pipeline with updated config settings
        self.pipeline = WheelAlignmentPipeline(config_manager=self.config)

        self.lbl_global_status.config(text="MONITORING", fg=STATE_COLORS["NORMAL"])
        self.lbl_warning_text.config(text="Monitoring active. Analyzing video feed for alignment anomalies...")

        # Spawn background worker thread
        self.stream_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.stream_thread.start()
        self.log_event("Started monitoring execution pipeline thread.")

    def stop_stream(self) -> None:
        """Stops background monitoring worker thread."""
        self.is_streaming = False
        self.pipeline.stop()
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)

        self.lbl_global_status.config(text="STOPPED", fg=STATE_COLORS["IDLE"])
        self.lbl_warning_text.config(text="Monitoring stream stopped by user.")
        self.log_event("Stopped monitoring stream.")

    def _worker_loop(self) -> None:
        """Background worker thread processing video stream frames non-blockingly."""
        src = self.video_source
        try:
            with VideoCaptureManager(source=src) as cap:
                for frame in cap.stream_frames():
                    if not self.is_streaming:
                        break

                    annotated, telemetry, perf = self.pipeline.process_frame(frame)

                    # Push frame & telemetry to queue, discarding old frame if queue full
                    if self.update_queue.full():
                        try:
                            self.update_queue.get_nowait()
                        except queue.Empty:
                            pass

                    self.update_queue.put((annotated, telemetry, perf))
                    time.sleep(0.02)  # ~50 FPS processing loop target

        except Exception as e:
            logger.error(f"Worker thread error: {e}", exc_info=True)
            self.root.after(0, lambda: self.log_event(f"Capture error: {e}", level="ERROR"))
        finally:
            self.root.after(0, self.stop_stream)

    def _start_queue_poller(self) -> None:
        """Schedules periodic GUI update polling loop on the main Tkinter thread."""
        self._poll_queue()

    def _poll_queue(self) -> None:
        """Polls frame queue and updates GUI components smoothly."""
        try:
            while not self.update_queue.empty():
                annotated, telemetry, perf = self.update_queue.get_nowait()
                self._update_gui(annotated, telemetry, perf)
        except queue.Empty:
            pass
        finally:
            self.root.after(30, self._poll_queue)  # ~33 FPS UI refresh rate

    def _update_gui(self, frame: np.ndarray, telemetry: List[Dict[str, Any]], perf: Dict[str, Any]) -> None:
        """Updates video canvas, HUD metrics, telemetry tree, and alert banners."""
        if frame is None or frame.size == 0:
            return

        # 1. Render Video Frame onto Canvas
        try:
            c_w = self.canvas_video.winfo_width() or 640
            c_h = self.canvas_video.winfo_height() or 480

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img_resized = img.resize((c_w, c_h), Image.Resampling.BILINEAR)
            photo = ImageTk.PhotoImage(image=img_resized)

            self.canvas_video.create_image(0, 0, image=photo, anchor=tk.NW)
            self.canvas_video.image = photo  # Prevent garbage collection
        except Exception as e:
            logger.error(f"Canvas render error: {e}")

        # 2. Update HUD Latency & FPS Bar
        self.lbl_fps.config(text=f"FPS: {perf.get('fps', 0.0):.1f}")
        self.lbl_total_lat.config(text=f"Latency: {perf.get('total_latency_ms', 0.0):.1f} ms")
        self.lbl_det_lat.config(text=f"Det: {perf.get('det_latency_ms', 0.0):.1f} ms")
        self.lbl_ml_lat.config(text=f"ML: {perf.get('ml_latency_ms', 0.0):.1f} ms")
        self.lbl_model_status.config(text=f"Model: {perf.get('model_status', 'Active')}")

        # 3. Update Telemetry Treeview Table
        for item in self.tree_telemetry.get_children():
            self.tree_telemetry.delete(item)

        global_highest_severity = "NORMAL"

        for record in telemetry:
            tid = record.get("track_id", -1)
            status = record.get("status", "NORMAL")
            conf = record.get("confidence", 0.0)
            camber = record.get("camber_proxy", 0.0)
            toe = record.get("toe_proxy", 0.0)
            motion = record.get("motion", {})
            spd = motion.get("velocity_magnitude", 0.0)

            self.tree_telemetry.insert(
                "",
                tk.END,
                values=(
                    tid,
                    status,
                    f"{conf * 100:.0f}%",
                    f"{camber:+0.1f}°",
                    f"{toe:+0.1f}°",
                    f"{spd:.0f}px/s"
                )
            )

            # Track highest severity status for global alert banner
            if status == "SEVERE_MISALIGNMENT":
                global_highest_severity = "SEVERE_MISALIGNMENT"
            elif status == "POSSIBLE_MISALIGNMENT" and global_highest_severity != "SEVERE_MISALIGNMENT":
                global_highest_severity = "POSSIBLE_MISALIGNMENT"
            elif status == "INSUFFICIENT_DATA" and global_highest_severity == "NORMAL":
                global_highest_severity = "INSUFFICIENT_DATA"

        # 4. Update Alert Status Banner
        if global_highest_severity in ["POSSIBLE_MISALIGNMENT", "SEVERE_MISALIGNMENT"]:
            self.lbl_global_status.config(text=global_highest_severity, fg=STATE_COLORS[global_highest_severity])
            self.lbl_warning_text.config(text=f"ALERT: {MANDATORY_WARNING_TEXT}", fg=STATE_COLORS[global_highest_severity])
        elif global_highest_severity == "INSUFFICIENT_DATA":
            self.lbl_global_status.config(text="INSUFFICIENT_DATA", fg=STATE_COLORS["INSUFFICIENT_DATA"])
            self.lbl_warning_text.config(text="Low confidence or insufficient region data.", fg=TEXT_MUTED)
        else:
            self.lbl_global_status.config(text="NORMAL", fg=STATE_COLORS["NORMAL"])
            self.lbl_warning_text.config(text="Wheel alignment operating within nominal parameters.", fg=TEXT_WHITE)

    def on_exit(self) -> None:
        """Gracefully shuts down monitoring thread and closes application window."""
        if messagebox.askokcancel("Exit Application", "Are you sure you want to stop monitoring and exit?"):
            self.is_streaming = False
            self.pipeline.stop()
            self.root.destroy()


def launch_gui() -> None:
    """Launches Tkinter GUI desktop monitoring dashboard."""
    root = tk.Tk()
    app = DashboardApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
