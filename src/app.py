from ctypes.wintypes import RGB
import cv2
import json
import time
import threading
import numpy as np
import os
from datetime import datetime
from typing import Any, Optional, cast

from depth_estimator import DepthEstimator
from camera import CameraStream
from memory import DetectionMemory
from config_loader import load_config, save_config
from settings_ui import launch_settings_ui

class SmartSightApp:
    def __init__(self, detector, tts, memory: Optional[DetectionMemory]=None, camera: Optional[CameraStream]=None):
        self.detector = detector
        self.tts = tts
        self.prev_boxes: dict = {}  # Store smoothed positions
        # sensible defaults for memory to allow re-speaking when appropriate
        self.memory = memory if memory else DetectionMemory(memory_duration=5, cooldown=3, distance_tolerance=0.4)
        self.camera = camera if camera else CameraStream(0)
        self.config = load_config()
        self.frame_skip_rate = int(self.config.get("skip_rate", 2))
        self.frame_count = 0
        self.depth_est = DepthEstimator("../models/midas_small.onnx")
        self.log_file = f"logs/{datetime.now().strftime('%d-%m-%Y')}.txt"
        os.makedirs("logs", exist_ok=True)
        self.settings_open = False

    def update_live_config(self, new_config: dict):
        self.config.update(new_config)
        self.frame_skip_rate = int(self.config.get("skip_rate", 2))
        save_config(self.config)

    def estimate_distance(self, raw_depth: float) -> float:
        normalized = np.clip(raw_depth, 0.01, 1.0)
        distance = 0.1 + (1.0 - normalized) * 9.9  # Scale to 0.1m - 10m
        return round(float(distance), 1)

    def draw_subtitle(self, frame: np.ndarray, text: str, x: int = 10, y_offset: int = 30, max_width: int = 60):
        lines, line = [], ""
        for word in text.split():
            test_line = f"{line} {word}".strip()
            if len(test_line) <= max_width:
                line = test_line
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)
        for i, line in enumerate(lines):
            y = frame.shape[0] - y_offset - (len(lines) - i - 1) * 25
            cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (200, 200, 200), 2, cv2.LINE_AA)

    def _force_camera_format_fix(self, frame):
        return frame

    def apply_image_enhancements(self, frame: Any) -> np.ndarray:

        # Normalize to numpy uint8 array early (prevents cvtColor type-check warnings)
        frame = np.asarray(frame)

        if frame.dtype != np.uint8:
            frame = np.uint8(np.clip(frame, 0, 255))

        # Protect color channels so if a grayscale camera is used, we convert to BGR
        if frame.ndim == 2:
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            except Exception:
                # fallback: stack grayscale into 3 channels
                frame = np.stack([frame, frame, frame], axis=-1)

        # operate in float for more stable processing
        f = frame.astype(np.float32) / 255.0

        if self.config.get("enable_gamma_correction", False):
            # safer adaptive gamma mapping
            try:
                mean_lum = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
            except Exception:
                mean_lum = 128.0
            gamma = np.clip(1.0 + (128.0 - mean_lum) / 256.0, 0.7, 1.6)
            f = np.power(f, gamma)

        if self.config.get("enable_brightness_boost", False):
            alpha = float(self.config.get("brightness_alpha", 1.1))
            beta = float(self.config.get("brightness_beta", 10)) / 255.0
            f = np.clip(alpha * f + beta, 0.0, 1.0)

        frame = np.uint8(np.clip(f * 255.0, 0, 255))


        return frame

    def _draw_aesthetic_box(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, label: str, confidence: float, distance: float):
        """
        Draw a soft, aesthetic bounding box and label showing label, distance, and confidence.
        Coordinates are (x1,y1,x2,y2).
        """
        # Unified soft color for boxes and label background (no distance-based tint)
        color = (0, 220, 220) # soft pastel yellow (BGR)
        thickness = 1

        # Cap coordinates within frame
        h, w = frame.shape[:2]
        x1 = max(0, min(w-1, int(x1)))
        y1 = max(0, min(h-1, int(y1)))
        x2 = max(0, min(w-1, int(x2)))
        y2 = max(0, min(h-1, int(y2)))

        # Draw anti-aliased rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        # Compose label with confidence and distance
        conf_pct = confidence * 100.0
        label_text = f"{label} {distance:.1f}m ({conf_pct:.0f}%)"

        # Compute text size and draw filled background for readability
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
        box_x2 = x1 + text_w + 10
        box_y1 = max(0, y1 - text_h - 10)
        cv2.rectangle(frame, (x1, box_y1), (box_x2, y1), color, -1)

        # Put label text in white atop background
        cv2.putText(frame, label_text, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def run(self):
        print("[INFO] SmartSight is running. Press 'q' or X to quit, 's' for settings.")
        window_name = "SmartSight"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        subtitle = ""
        subtitle_time = 0.0
        fps = 0.0
        last_time = time.time()

        try:
            while True:
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                # Mirror for user-facing camera
                frame = cv2.flip(frame, 1)

                self.frame_count += 1
                if self.frame_count % self.frame_skip_rate != 0:
                    continue

                # Validate frame
                if not hasattr(frame, "shape") or frame.shape[0] == 0:
                    continue

                h, w = frame.shape[:2]

                # Apply safe enhancements (keeps natural camera color unless forced)
                frame = self.apply_image_enhancements(frame)

                # Depth map (either estimated or dummy)
                depth_map = self.depth_est.estimate(frame) if self.config.get("enable_depth", True) else np.ones((h, w), dtype=np.float32)

                raw_detections = self.detector.detect(frame)
                detections = []

                # Convert detections into consistent dicts and smooth boxes
                for det in raw_detections:
                    if det.get("confidence", 0.0) < 0.5:
                        continue
                    x1, y1, x2, y2 = det["box"]
                    label = det["label"]
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    if not (0 <= cx < w and 0 <= cy < h):
                        continue

                    raw_depth = float(depth_map[cy, cx])
                    distance = self.estimate_distance(raw_depth)

                    key = label
                    prev = self.prev_boxes.get(key, (x1, y1, x2, y2))
                    smooth_x1 = int(prev[0] * 0.7 + x1 * 0.3)
                    smooth_y1 = int(prev[1] * 0.7 + y1 * 0.3)
                    smooth_x2 = int(prev[2] * 0.7 + x2 * 0.3)
                    smooth_y2 = int(prev[3] * 0.7 + y2 * 0.3)
                    self.prev_boxes[key] = (smooth_x1, smooth_y1, smooth_x2, smooth_y2)

                    detections.append({
                        "box": (smooth_x1, smooth_y1, smooth_x2, smooth_y2),
                        "label": label,
                        "confidence": det.get("confidence", 0.0),
                        "distance": distance
                    })

                # Keep the closest two detections
                detections = sorted(detections, key=lambda d: d["distance"])[:2]

                for det in detections:
                    x1, y1, x2, y2 = det["box"]
                    label = det["label"]
                    distance = det["distance"]
                    confidence = det["confidence"]

                    # Draw clean aesthetic box + label (no distance tint)
                    frame = self._draw_aesthetic_box(frame, x1, y1, x2, y2, label, confidence, distance)

                    # Prepare speaking logic
                    cx = (x1 + x2) // 2
                    if cx < w // 3:
                        direction = "left"
                    elif cx > 2 * w // 3:
                        direction = "right"
                    else:
                        direction = "center"

                    speak_key = f"{label}_{direction}"
                    if self.memory.should_speak(speak_key, distance):
                        speech_text = f"{label} ahead on your {direction}, {distance} meters"
                        try:
                            self.tts.speak(speech_text)
                        except Exception:
                            # if TTS fails, Log
                            pass

                        subtitle = speech_text
                        subtitle_time = time.time()
                        with open(self.log_file, "a") as log:
                            log.write(f"[{datetime.now().strftime('%H:%M:%S')}] — {speech_text}\n")
                        # speak for the first detection then keep other detections visible
                        break

                # FPS calculation
                current_time = time.time()
                fps = 1.0 / (current_time - last_time + 1e-6)
                last_time = current_time

                # Subtitle display
                if subtitle and time.time() - subtitle_time < 3.0:
                    self.draw_subtitle(frame, subtitle)

                # Show FPS if enabled
                if self.config.get("show_fps", True):
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF

                # Window controls
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('c') or key == ord('C'):
                    self.memory.clear_memory()
                    print("[INFO] Detection memory cleared.")
                elif key == ord('s') or key == ord('S'):
                    if not self.settings_open:
                        self.settings_open = True
                        print("[INFO] Opening Settings UI...")
                        launch_settings_ui(on_change=self.update_live_config)
                        self.settings_open = False
                elif key == ord('d') or key == ord('D'):
                    self.config["enable_depth"] = not self.config.get("enable_depth", True)
                    save_config(self.config)
                    print(f"[TOGGLE] Depth Estimation: {self.config['enable_depth']}")
                elif key == ord('f') or key == ord('F'):
                    self.config["show_fps"] = not self.config.get("show_fps", True)
                    save_config(self.config)
                    print(f"[TOGGLE] Show FPS: {self.config['show_fps']}")
                elif key == ord('t') or key == ord('T'):
                    self.config["theme"] = "dark" if self.config.get("theme") == "light" else "light"
                    save_config(self.config)
                    print(f"[TOGGLE] Theme: {self.config['theme']}")
                elif key == ord('a') or key == ord('A'):
                    self.config["enable_audio_panning"] = not self.config.get("enable_audio_panning", False)
                    save_config(self.config)
                    print(f"[TOGGLE] Audio Panning: {self.config['enable_audio_panning']}")

        finally:
            self.camera.stop()
            cv2.destroyAllWindows()
            print("[INFO] SmartSight shut down.")
