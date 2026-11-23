"""First-stage corridor control using body-frame LiDAR points in metres."""

from math import isfinite, radians, tan
from statistics import median
from typing import Iterable


FORWARD_SLOPE = tan(radians(25))


def corridor_command(
    points: Iterable[tuple[float, float]],
    *,
    stop_distance: float,
    half_width: float,
    scan_fresh: bool,
) -> tuple[float, float]:
    """Return forward and yaw commands in m/s and rad/s.

    Points use x forward and y left. The caller determines freshness from the
    scan timestamp and supplies a stop envelope measured on the actual chassis.
    """
    if not (isfinite(stop_distance) and isfinite(half_width)
            and stop_distance > 0 and half_width > 0):
        raise ValueError("stop envelope dimensions must be finite and positive")
    if not scan_fresh:
        return 0.0, 0.0

    left = []
    right = []
    forward_returns = 0
    for x, y in points:
        if not (isfinite(x) and isfinite(y)):
            continue
        if 0 <= x <= stop_distance and abs(y) <= half_width:
            return 0.0, 0.0
        if x > 0 and abs(y) <= x * FORWARD_SLOPE:
            forward_returns += 1
        if 0.20 <= x <= 0.70:
            if y > 0:
                left.append(y)
            elif y < 0:
                right.append(-y)

    if forward_returns < 3 or len(left) < 3 or len(right) < 3:
        return 0.0, 0.0


    clearance_error = median(left) - median(right)
    yaw = max(-0.4, min(0.4, clearance_error))
    return 0.10, yaw
