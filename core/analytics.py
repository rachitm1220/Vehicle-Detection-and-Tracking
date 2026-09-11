"""
analytics.py
------------
The "make it stronger" features:
  1. LineCounter        -> vehicle counting via virtual line-crossing
  2. SpeedEstimator      -> per-track speed in km/h using a pixel-to-meter calibration
  3. TrafficDensityMeter -> Low / Medium / High traffic classification

All classes are stateful across frames (they track history per `track_id`),
so a single instance should persist for the lifetime of a video stream.
"""

import time
from collections import deque
from typing import Dict, List, Tuple

from core.detector import Detection


class LineCounter:
    """Counts unique track IDs crossing a user-defined virtual line."""

    def __init__(self, line: Tuple[Tuple[int, int], Tuple[int, int]]):
        self.line = line
        self._last_side: Dict[int, float] = {}
        self.counted_ids = set()
        self.count_in = 0
        self.count_out = 0

    @staticmethod
    def _side_of_line(p, a, b) -> float:
        # Cross product sign tells us which side of the line a point is on
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

    def update(self, detections: List[Detection]):
        a, b = self.line
        for det in detections:
            if det.track_id is None:
                continue
            side = self._side_of_line(det.centroid, a, b)
            prev = self._last_side.get(det.track_id)
            self._last_side[det.track_id] = side

            if prev is None:
                continue
            # Sign flip => the object crossed the line this frame
            if prev < 0 and side >= 0 and det.track_id not in self.counted_ids:
                self.count_in += 1
                self.counted_ids.add(det.track_id)
            elif prev > 0 and side <= 0 and det.track_id not in self.counted_ids:
                self.count_out += 1
                self.counted_ids.add(det.track_id)

    @property
    def total(self) -> int:
        return self.count_in + self.count_out


class SpeedEstimator:
    """
    Estimates per-object speed from pixel displacement between frames,
    converted to real-world units via a pixels-per-meter calibration factor.
    """

    def __init__(self, pixels_per_meter: float, smoothing_window: int = 5):
        self.pixels_per_meter = max(pixels_per_meter, 1e-6)
        self.smoothing_window = smoothing_window
        self._history: Dict[int, deque] = {}  # track_id -> deque[(t, x, y)]
        self.speeds_kmh: Dict[int, float] = {}

    def update(self, detections: List[Detection]):
        now = time.time()
        for det in detections:
            if det.track_id is None:
                continue
            cx, cy = det.centroid
            hist = self._history.setdefault(det.track_id, deque(maxlen=self.smoothing_window))
            hist.append((now, cx, cy))

            if len(hist) >= 2:
                (t0, x0, y0), (t1, x1, y1) = hist[0], hist[-1]
                dt = max(t1 - t0, 1e-6)
                dist_px = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
                dist_m = dist_px / self.pixels_per_meter
                speed_ms = dist_m / dt
                self.speeds_kmh[det.track_id] = speed_ms * 3.6

    def get_speed(self, track_id: int) -> float:
        return self.speeds_kmh.get(track_id, 0.0)

    def cleanup(self, active_ids: List[int]):
        """Drop history for tracks no longer visible, to bound memory use."""
        stale = set(self._history) - set(active_ids)
        for tid in stale:
            self._history.pop(tid, None)
            self.speeds_kmh.pop(tid, None)


class TrafficDensityMeter:
    """
    Classifies traffic density from a rolling average of vehicle counts
    per frame within a region of interest (or the whole frame).
    """

    THRESHOLDS = {"Low": 5, "Medium": 12}  # <=5 Low, <=12 Medium, else High

    def __init__(self, window: int = 30):
        self.window = window
        self._history = deque(maxlen=window)

    def update(self, vehicle_count: int) -> str:
        self._history.append(vehicle_count)
        avg = sum(self._history) / len(self._history)
        if avg <= self.THRESHOLDS["Low"]:
            return "Low"
        elif avg <= self.THRESHOLDS["Medium"]:
            return "Medium"
        return "High"

    @property
    def rolling_average(self) -> float:
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)
