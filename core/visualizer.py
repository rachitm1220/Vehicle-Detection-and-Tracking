"""
visualizer.py
-------------
All OpenCV drawing lives here so app.py stays focused on Streamlit/UI logic.
"""

from typing import List, Tuple

import cv2
import numpy as np

from core.detector import Detection

# Deterministic color per class id, for visual consistency across frames
_PALETTE = [
    (255, 99, 71), (60, 179, 113), (255, 215, 0), (30, 144, 255),
    (218, 112, 214), (255, 140, 0), (0, 206, 209), (255, 105, 180),
    (154, 205, 50), (100, 149, 237),
]


def _color_for(class_id: int) -> Tuple[int, int, int]:
    return _PALETTE[class_id % len(_PALETTE)]


def draw_detections(frame: np.ndarray, detections: List[Detection], speeds: dict = None) -> np.ndarray:
    speeds = speeds or {}
    for det in detections:
        color = _color_for(det.class_id)
        p1, p2 = (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2))
        cv2.rectangle(frame, p1, p2, color, 2)

        label = f"{det.class_name} {det.confidence:.2f}"
        if det.track_id is not None:
            label = f"ID {det.track_id} | " + label
        speed = speeds.get(det.track_id) if det.track_id is not None else None
        if speed:
            label += f" | {speed:.0f} km/h"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (p1[0], p1[1] - th - 8), (p1[0] + tw + 6, p1[1]), color, -1)
        cv2.putText(frame, label, (p1[0] + 3, p1[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def draw_counting_line(frame: np.ndarray, line: Tuple[Tuple[int, int], Tuple[int, int]],
                        count_in: int, count_out: int) -> np.ndarray:
    a, b = line
    cv2.line(frame, a, b, (0, 255, 255), 2)
    cv2.putText(frame, f"IN: {count_in}  OUT: {count_out}", (a[0], a[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return frame


def draw_hud(frame: np.ndarray, fps: float, density: str, total_objects: int) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (260, 90), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    density_color = {"Low": (0, 200, 0), "Medium": (0, 165, 255), "High": (0, 0, 255)}.get(density, (255, 255, 255))

    cv2.putText(frame, f"FPS: {fps:.1f}", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Objects: {total_objects}", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Traffic: {density}", (12, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, density_color, 2, cv2.LINE_AA)
    return frame
