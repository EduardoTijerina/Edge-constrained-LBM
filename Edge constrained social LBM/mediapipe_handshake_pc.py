#!/usr/bin/env python3
"""
Stage 1 — Handshake Gesture Detector (Laptop Webcam)
====================================================
Master's Thesis: Low-Latency Social Gesture Recognition for Humanoid Robots
Osaka University — Humanoid Robotics Laboratory (Ishiguro Lab)

Pipeline:  Laptop webcam (cv2.VideoCapture) → MediaPipe Hands → Gesture Heuristic → ZeroMQ PUB
Target:    Desktop PC / Kubuntu  (later: Jetson Nano + OAK-D)

Usage:
    pip install opencv-python mediapipe pyzmq
    python handshake_detector.py                  # default: /dev/video0
    python handshake_detector.py --camera 2       # pick a different camera index

Subscriber test (separate terminal):
    python -c "
    import zmq
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect('tcp://localhost:5555')
    sub.setsockopt_string(zmq.SUBSCRIBE, 'GESTURE')
    while True:
        print(sub.recv_string())
    "
"""

from __future__ import annotations

import csv
import math
import os
import time
import argparse
import urllib.request

# Force X11/XWayland backend so cv2.imshow works under Wayland
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
# Suppress Qt warnings (threading, fonts) from OpenCV's HighGUI
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.*=false"
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"

import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import zmq


# ---------------------------------------------------------------------------
#  OAK-D CAPTURE  (DepthAI — RGB + aligned stereo depth)
# ---------------------------------------------------------------------------

class OakDCapture:
    """DepthAI wrapper with the same read()/release() interface as cv2.VideoCapture."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        import depthai as dai

        pipeline = dai.Pipeline()

        # depthai v3: Camera.build() replaces ColorCamera + setBoardSocket/setFps
        camRgb = pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_A, sensorFps=float(fps)
        )
        rgbOut = camRgb.requestOutput((width, height), dai.ImgFrame.Type.BGR888p)

        # depthai v3: StereoDepth.build(autoCreateCameras=True) replaces manual MonoCamera setup
        stereo = pipeline.create(dai.node.StereoDepth).build(
            autoCreateCameras=True,
            presetMode=dai.node.StereoDepth.PresetMode.DENSITY,
            size=(640, 400),
            fps=float(fps),
        )
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(width, height)

        # depthai v3: createOutputQueue() on an output replaces XLinkOut nodes
        self._rgb_q = rgbOut.createOutputQueue(maxSize=4, blocking=False)
        self._depth_q = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

        # depthai v3: pipeline owns the implicit device; start() replaces dai.Device(pipeline)
        self._pipeline = pipeline
        pipeline.start()
        self.width = width
        self.height = height

    def read(self):
        msg = self._rgb_q.get()
        if msg is None:
            return False, None
        return True, msg.getCvFrame()

    def get_depth(self) -> "np.ndarray | None":
        msg = self._depth_q.tryGet()
        return msg.getFrame() if msg is not None else None

    def release(self):
        self._pipeline.stop()
        self._pipeline.wait()


# ---------------------------------------------------------------------------
#  MODEL DOWNLOAD
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

POSE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


def _ensure_model() -> None:
    if not os.path.exists(MODEL_PATH):
        print(f"[INFO] Downloading hand landmarker model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Model downloaded.")


def _ensure_pose_model() -> None:
    if not os.path.exists(POSE_MODEL_PATH):
        print(f"[INFO] Downloading pose landmarker model to {POSE_MODEL_PATH} ...")
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
        print("[INFO] Pose model downloaded.")


# ---------------------------------------------------------------------------
#  GESTURE LOGIC — fully decoupled from capture backend
# ---------------------------------------------------------------------------

# MediaPipe landmark indices (same numbering across all backends)
WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_MCP = 17
PINKY_TIP = 20
INDEX_TIP = 8
THUMB_TIP = 4

# Joint triples (MCP, PIP, TIP) — finger is "extended" when the angle at the
# PIP joint is close to 180°. Using the joint angle (rather than wrist→tip
# distance) is robust to perspective foreshortening when fingers point at
# the camera, where a closed fist used to give false "extended" counts.
_FINGER_JOINT_TRIPLES = [
    (5, 6, 8),     # index:  MCP-PIP-TIP
    (9, 10, 12),   # middle
    (13, 14, 16),  # ring
    (17, 18, 20),  # pinky
]

# Hand connections for manual drawing (mediapipe tasks API no longer exports these)
_HAND_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
])

# Pose landmark indices (MediaPipe Pose) for arm tracking.
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW     = 13
RIGHT_ELBOW    = 14
LEFT_WRIST_POSE  = 15
RIGHT_WRIST_POSE = 16

_ARM_CONNECTIONS = [
    (LEFT_SHOULDER,  LEFT_ELBOW),  (LEFT_ELBOW,  LEFT_WRIST_POSE),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST_POSE),
    (LEFT_SHOULDER,  RIGHT_SHOULDER),
]
_ARM_LANDMARK_IDS = (
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST_POSE, RIGHT_WRIST_POSE,
)


def _dist(a, b) -> float:
    """Euclidean pixel distance between two landmark-like objects."""
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def _angle_at_joint_3d(a, b, c) -> float:
    """Angle in degrees at joint b formed by a-b-c, using MediaPipe's 3D
    landmarks. 3D is needed because when a finger points at the camera the
    2D projection collapses and the planar angle becomes meaningless."""
    v1 = (a.x - b.x, a.y - b.y, a.z - b.z)
    v2 = (c.x - b.x, c.y - b.y, c.z - b.z)
    dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    mag1 = (v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2) ** 0.5
    mag2 = (v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2) ** 0.5
    if mag1 < 1e-6 or mag2 < 1e-6:
        return 0.0
    cos_a = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_a))


def _count_extended_fingers(landmarks, angle_threshold: float = 150.0) -> int:
    """A finger is extended when its MCP-PIP-TIP angle is roughly straight
    (≥150°). Curled (fist) fingers sit around 30–90°."""
    count = 0
    for mcp, pip, tip in _FINGER_JOINT_TRIPLES:
        if _angle_at_joint_3d(landmarks[mcp], landmarks[pip], landmarks[tip]) >= angle_threshold:
            count += 1
    return count


def _palm_facing_ratio(landmarks) -> float:
    """Ratio of knuckle width to palm length in 2D image coordinates.

    When the hand is edge-on (handshake pose), the knuckle-to-knuckle span
    appears compressed → low ratio (~0.2–0.5).
    When the palm or back of the hand faces the camera, both dimensions
    are fully visible → high ratio (~0.8–1.2).
    """
    knuckle_width = _dist(landmarks[INDEX_MCP], landmarks[PINKY_MCP])
    palm_length = _dist(landmarks[WRIST], landmarks[MIDDLE_MCP])
    return knuckle_width / max(palm_length, 1e-6)


def detect_handshake(
    landmarks,
    depth_frame: np.ndarray | None = None,
    frame_w: int = 640,
    frame_h: int = 480,
    proximity_threshold: float = 50.0,
) -> tuple[bool, float, dict, dict]:
    """
    Detect a handshake offer pose with fingers pointing toward the camera.

    Camera is mounted on the robot's head; the person extends their arm
    so that fingertips point at the robot's chest/camera.

    Four conditions:
      1. Pointing — fingertips are closer to camera than wrist.
         With stereo depth: difference in mm (threshold = proximity_threshold mm).
         Without stereo: MediaPipe z estimate (threshold = proximity_threshold, ~0.05).
      2. Fingers extended — at least 3 of 4 fingers extended.
      3. Together — index-to-pinky spread < 0.25 (fingers not splayed).
      4. Edge-on — palm is sideways (not flat facing camera).
         Uses the ratio of visible knuckle width to palm length.

    Returns
    -------
    triggered : bool
    pointing_depth : float
        Depth difference (wrist − mean_fingertip). mm when stereo available, else normalised.
    conds : dict[str, bool]
    metrics : dict
    """
    lm = landmarks
    finger_spread = _dist(lm[INDEX_TIP], lm[PINKY_TIP])
    n_extended = _count_extended_fingers(lm)
    palm_ratio = _palm_facing_ratio(lm)

    if depth_frame is not None:
        # --- Real stereo depth (mm) ----------------------------------------
        dh, dw = depth_frame.shape[:2]

        def _sample(pt) -> float:
            px = int(np.clip(pt.x * dw, 0, dw - 1))
            py = int(np.clip(pt.y * dh, 0, dh - 1))
            return float(depth_frame[py, px])

        wrist_d = _sample(lm[WRIST])
        tip_depths = [_sample(lm[t]) for t in (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)]
        valid_tips = [d for d in tip_depths if d > 0]

        if wrist_d > 0 and valid_tips:
            mean_tip_d = sum(valid_tips) / len(valid_tips)
            # Smaller depth value = closer to camera; positive → fingers closer
            pointing_depth = wrist_d - mean_tip_d
        else:
            pointing_depth = 0.0
    else:
        # --- MediaPipe z estimate (normalised) --------------------------------
        mean_fingertip_z = (
            lm[INDEX_TIP].z + lm[MIDDLE_TIP].z +
            lm[RING_TIP].z  + lm[PINKY_TIP].z
        ) / 4.0
        pointing_depth = lm[WRIST].z - mean_fingertip_z

    # Threshold semantics differ by mode: stereo reports millimetres, and
    # MediaPipe's normalised z is roughly 1–2 orders of magnitude smaller.
    pointing_threshold = proximity_threshold if depth_frame is not None else 0.04

    conds = {
        "pointing": pointing_depth > pointing_threshold,
        "fingers":  n_extended >= 3,
        "together": finger_spread < 0.22,
        "edge_on":  palm_ratio > 1.0,
    }
    triggered = all(conds.values())
    metrics = {
        "n_extended":     n_extended,
        "finger_spread":  finger_spread,
        "pointing_depth": pointing_depth,
        "palm_ratio":     palm_ratio,
    }
    return triggered, pointing_depth, conds, metrics


def _draw_landmarks(frame, landmarks, frame_w: int, frame_h: int) -> None:
    """Draw hand landmarks and connections onto frame (in-place)."""
    pts = [
        (int(lm.x * frame_w), int(lm.y * frame_h))
        for lm in landmarks
    ]
    for start, end in _HAND_CONNECTIONS:
        cv2.line(frame, pts[start], pts[end], (0, 200, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 255, 0), -1)


def _draw_arms(frame, pose_landmarks, frame_w: int, frame_h: int,
               vis_threshold: float = 0.5) -> None:
    """Draw shoulder–elbow–wrist of both arms (visualisation only).

    The handshake trigger ignores these landmarks; this is purely so the
    operator can see arm pose alongside the hand.
    """
    pts: dict[int, tuple[int, int]] = {}
    for idx in _ARM_LANDMARK_IDS:
        lm = pose_landmarks[idx]
        if getattr(lm, "visibility", 1.0) < vis_threshold:
            continue
        pts[idx] = (int(lm.x * frame_w), int(lm.y * frame_h))

    for start, end in _ARM_CONNECTIONS:
        if start in pts and end in pts:
            cv2.line(frame, pts[start], pts[end], (0, 165, 255), 3)
    for pt in pts.values():
        cv2.circle(frame, pt, 6, (0, 100, 255), -1)


def _write_latency_summary(latency_samples, latency_writer) -> None:
    if not latency_samples:
        return
    latency_samples.sort()
    n = len(latency_samples)
    def _pct(p):
        return latency_samples[min(int(p / 100 * n), n - 1)]
    latency_writer.writerow([])
    latency_writer.writerow(["# SUMMARY", "stat", "value_ms"])
    for label, value in [
        ("min",    latency_samples[0]),
        ("max",    latency_samples[-1]),
        ("mean",   sum(latency_samples) / n),
        ("median", _pct(50)),
        ("p95",    _pct(95)),
        ("p99",    _pct(99)),
        ("frames", n),
    ]:
        latency_writer.writerow(["#", label, f"{value:.3f}"])


# ---------------------------------------------------------------------------
#  CAPTURE LOOP
# ---------------------------------------------------------------------------

def _run_loop(
    cap, frame_w, frame_h,
    landmarker, pub, args,
    csv_writer, latency_writer, latency_samples,
    depth_fn=None,
    ipc_socket=None,
    pose_landmarker=None,
) -> None:
    # Continuous-state model: HANDSHAKE on rising edge, RELEASE after the
    # detector has been silent for `release_debounce` seconds. The orchestrator
    # holds the peak-extension pose between those two events.
    handshake_active: bool = False
    last_active_t: float = 0.0
    t_start = time.perf_counter()
    frame_idx = 0
    latest_depth: np.ndarray | None = None
    fps_t = time.perf_counter()
    fps: float = 0.0

    # Smoothed metrics (exponential moving average)
    _EMA_ALPHA = 0.15          # lower = smoother, higher = more responsive
    smooth = {
        "pointing_depth": 0.0,
        "finger_spread":  0.0,
        "palm_ratio":     0.0,
        "n_extended":     0.0,
    }

    while True:
        t0 = time.perf_counter()

        ok, frame = cap.read()
        if not ok or frame is None:
            print("[WARN] Frame grab failed — retrying")
            continue
        latest_depth = depth_fn() if depth_fn is not None else None
        # Mirror so the preview feels like a mirror (natural for webcams)
        frame = cv2.flip(frame, 1)

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.perf_counter() - t_start) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        pose_result = (
            pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            if pose_landmarker is not None else None
        )

        # Arm overlay is drawn first so the hand sits on top of it.
        if (not args.no_display
                and pose_result is not None
                and pose_result.pose_landmarks):
            _draw_arms(frame, pose_result.pose_landmarks[0], frame_w, frame_h)

        triggered = False
        palm_size = 0.0
        conds: dict = {}
        raw_metrics: dict = {"n_extended": 0, "finger_spread": 0.0, "pointing_depth": 0.0, "palm_ratio": 0.0}
        hand_detected = False

        if result.hand_landmarks:
            hand_lm = result.hand_landmarks[0]
            hand_detected = True

            triggered, palm_size, conds, raw_metrics = detect_handshake(
                hand_lm,
                depth_frame=latest_depth,
                frame_w=frame_w,
                frame_h=frame_h,
                proximity_threshold=args.threshold,
            )

            # Update smoothed metrics
            for key in smooth:
                raw_val = float(raw_metrics.get(key, 0.0))
                smooth[key] = smooth[key] * (1 - _EMA_ALPHA) + raw_val * _EMA_ALPHA

            if not args.no_display:
                _draw_landmarks(frame, hand_lm, frame_w, frame_h)

        # --- Edge-triggered state machine ---------------------------------
        # Rising edge of `triggered` -> HANDSHAKE.
        # `release_debounce` seconds without `triggered` -> RELEASE.
        now = time.time()
        zmq_sent = False
        if triggered:
            last_active_t = now
            if not handshake_active:
                msg = f"GESTURE HANDSHAKE_REQ {now:.3f}"
                pub.send_string(msg)
                if ipc_socket is not None:
                    ipc_socket.send_string("TRIGGER: HANDSHAKE")
                handshake_active = True
                zmq_sent = True
                print(f"  >>> START  {msg}")
        elif handshake_active and (now - last_active_t) >= args.release_debounce:
            msg = f"GESTURE HANDSHAKE_END {now:.3f}"
            pub.send_string(msg)
            if ipc_socket is not None:
                ipc_socket.send_string("TRIGGER: RELEASE")
            handshake_active = False
            zmq_sent = True
            print(f"  >>> END    {msg}")

        # --- CSV logging ---------------------------------------------------
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if not args.no_log:
            latency_writer.writerow([frame_idx, f"{now:.6f}", f"{dt_ms:.3f}"])
            latency_samples.append(dt_ms)
            csv_writer.writerow({
                "wall_time_s":        f"{now:.6f}",
                "frame_idx":          frame_idx,
                "latency_ms":         f"{dt_ms:.3f}",
                "hand_detected":      int(hand_detected),
                "palm_size":          f"{palm_size:.6f}",
                "n_extended_fingers": raw_metrics["n_extended"],
                "finger_spread":      f"{raw_metrics['finger_spread']:.6f}",
                "cond_close":         int(conds.get("close", False)),
                "cond_fingers":       int(conds.get("fingers", False)),
                "cond_together":      int(conds.get("together", False)),
                "cond_vertical":      int(conds.get("vertical", False)),
                "cond_thumb_up":      int(conds.get("thumb_up", False)),
                "triggered":          int(triggered),
                "zmq_sent":           int(zmq_sent),
            })
        frame_idx += 1

        # --- Latency + condition readout (throttled to ~1 Hz) ---------------
        if frame_idx % max(int(args.fps), 1) == 0:
            depth_unit = "mm" if latest_depth is not None else "norm"
            print(
                f"point={smooth['pointing_depth']:.1f}{depth_unit}  "
                f"spread={smooth['finger_spread']:.3f}  "
                f"palm={smooth['palm_ratio']:.2f}  "
                f"ext={smooth['n_extended']:.1f}"
            )

        # --- Display -------------------------------------------------------
        if not args.no_display:
            n_conds_met = sum(conds.values()) if conds else 0

            if hand_detected:
                if triggered:
                    label_text = "HANDSHAKE DETECTED"
                    label_color = (0, 220, 0)
                elif n_conds_met >= 3:
                    label_text = "reading handshake..."
                    label_color = (255, 200, 0)
                else:
                    label_text = "hand detected"
                    label_color = (200, 200, 200)
            else:
                label_text = "no hand detected"
                label_color = (100, 100, 100)

            cv2.putText(
                frame, label_text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, label_color, 2,
            )

            # Show individual condition status on screen
            if hand_detected:
                for i, (k, v) in enumerate(conds.items()):
                    color = (0, 220, 0) if v else (0, 0, 200)
                    cv2.putText(
                        frame, f"{k}: {'OK' if v else '--'}",
                        (10, 80 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                    )
                depth_unit = "mm" if latest_depth is not None else "norm"
                cv2.putText(
                    frame,
                    f"point: {smooth['pointing_depth']:.1f}{depth_unit}  spread: {smooth['finger_spread']:.3f}  palm: {smooth['palm_ratio']:.2f}  ext: {smooth['n_extended']:.1f}",
                    (10, 80 + len(conds) * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
                )

            # FPS counter (top-right corner)
            now_t = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now_t - fps_t, 1e-6))
            fps_t = now_t
            fps_text = f"FPS: {fps:.1f}"
            (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(
                frame, fps_text,
                (frame_w - tw - 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )

            cv2.imshow("Handshake Detector — Stage 1", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[INFO] Quit requested.")
                break


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Handshake gesture detector — laptop webcam"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Webcam index passed to cv2.VideoCapture (default: 0; only used when --no-oak is set)",
    )
    parser.add_argument(
        "--no-oak", action="store_true",
        help="Skip OAK-D and use a plain webcam instead",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Requested webcam FPS (default: 30; driver may clamp)",
    )
    parser.add_argument(
        "--width", type=int, default=640,
        help="Preview frame width in pixels (default: 640)",
    )
    parser.add_argument(
        "--height", type=int, default=480,
        help="Preview frame height in pixels (default: 480)",
    )
    parser.add_argument(
        "--zmq-port", type=int, default=5555,
        help="ZeroMQ PUB port (default: 5555)",
    )
    parser.add_argument(
        "--cooldown", type=float, default=4.0,
        help="Seconds between consecutive triggers (default: 4.0)",
    )
    parser.add_argument(
        "--release-debounce", type=float, default=0.5,
        help="Seconds of silence after the last detected handshake before emitting HANDSHAKE_END (default: 0.5)",
    )
    parser.add_argument(
        "--threshold", type=float, default=20.0,
        help="Pointing depth threshold in mm when stereo is active (default: 20mm)",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run headless (no cv2.imshow window)",
    )
    parser.add_argument(
        "--log", type=str, default=None,
        help="Path to CSV file for saving per-frame metrics (default: auto-generated)",
    )
    parser.add_argument(
        "--no-log", action="store_true", default=True,
        help="Disable CSV logging (no files written, enabled by default)",
    )
    parser.add_argument(
        "--no-arm", action="store_true",
        help="Skip the MediaPipe Pose arm overlay (saves a few ms/frame)",
    )
    args = parser.parse_args()

    # Auto-generate a log filename if not provided
    log_path = args.log or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"mediapipe_run_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    )

    # ---- Download model(s) if needed --------------------------------------
    _ensure_model()
    if not args.no_arm:
        _ensure_pose_model()

    # ---- ZeroMQ PUB socket ------------------------------------------------
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, 0)
    pub.setsockopt(zmq.IMMEDIATE, 1)
    pub.bind(f"tcp://*:{args.zmq_port}")
    print(f"[ZMQ] PUB socket bound on tcp://*:{args.zmq_port}")

    # ---- ZeroMQ IPC trigger (C++ orchestration stack) ---------------------
    ipc_endpoint = "ipc:///tmp/handshake_trigger"
    socket = ctx.socket(zmq.PUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(ipc_endpoint)
    print(f"[ZMQ] IPC trigger socket bound on {ipc_endpoint}")

    # ---- MediaPipe Hands (tasks API) --------------------------------------
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    # ---- MediaPipe Pose (arm overlay only — does NOT gate the trigger) ----
    pose_landmarker = None
    if not args.no_arm:
        pose_options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        pose_landmarker = mp_vision.PoseLandmarker.create_from_options(pose_options)

    frame_w = args.width
    frame_h = args.height

    # ---- CSV logger -------------------------------------------------------
    _CSV_FIELDS = [
        "wall_time_s", "frame_idx", "latency_ms",
        "hand_detected", "palm_size", "n_extended_fingers", "finger_spread",
        "cond_close", "cond_fingers", "cond_together", "cond_vertical", "cond_thumb_up",
        "triggered", "zmq_sent",
    ]
    if not args.no_log:
        log_file = open(log_path, "w", newline="")
        csv_writer = csv.DictWriter(log_file, fieldnames=_CSV_FIELDS)
        csv_writer.writeheader()
        print(f"[LOG]  Writing per-frame metrics to {log_path}")
        _base = os.path.splitext(log_path)[0]
        latency_log_path = _base + "_latency.csv"
        latency_file = open(latency_log_path, "w", newline="")
        latency_writer = csv.writer(latency_file)
        latency_writer.writerow(["frame_idx", "wall_time_s", "latency_ms"])
        print(f"[LOG]  Writing latency log     to {latency_log_path}")
    else:
        log_file = latency_file = csv_writer = latency_writer = None
        latency_log_path = None
        print("[LOG]  CSV logging disabled (--no-log)")
    latency_samples: list[float] = []

    if not args.no_display:
        cv2.namedWindow("Handshake Detector — Stage 1", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Handshake Detector — Stage 1", frame_w, frame_h)

    depth_fn = None
    if not args.no_oak:
        try:
            cap = OakDCapture(width=frame_w, height=frame_h, fps=args.fps)
            depth_fn = cap.get_depth
            print(f"[CAM]  OAK-D Pro  {frame_w}x{frame_h}@{args.fps}fps  (stereo depth enabled)")
        except Exception as e:
            print(f"[WARN] OAK-D init failed ({e}), falling back to webcam index {args.camera}")
            cap = None

    if depth_fn is None:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam at index {args.camera}. "
                f"Try --camera 1 or check /dev/video* and permissions."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or frame_w
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or frame_h
        frame_w, frame_h = actual_w, actual_h
        print(f"[CAM]  Webcam /dev/video{args.camera}  {frame_w}x{frame_h}@{args.fps}fps (no depth)")

    depth_mode = "stereo depth" if depth_fn is not None else "normalised z"
    try:
        print(f"[CFG]  Cooldown={args.cooldown}s  PointingThreshold={args.threshold} ({depth_mode})")
        print("[INFO] Press 'q' to quit.\n")

        _run_loop(
            cap, frame_w, frame_h,
            landmarker, pub, args,
            csv_writer, latency_writer, latency_samples,
            depth_fn=depth_fn,
            ipc_socket=socket,
            pose_landmarker=pose_landmarker,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        cap.release()
        landmarker.close()
        if pose_landmarker is not None:
            pose_landmarker.close()
        pub.close()
        socket.close()
        ctx.term()
        cv2.destroyAllWindows()
        if not args.no_log:
            _write_latency_summary(latency_samples, latency_writer)
            log_file.close()
            latency_file.close()
            print(f"[DONE] Log saved        → {log_path}")
            print(f"[DONE] Latency log saved → {latency_log_path}")
        print("[DONE] Cleaned up resources.")


if __name__ == "__main__":
    main()