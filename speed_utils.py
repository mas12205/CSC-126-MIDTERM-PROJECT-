"""
speed_utils.py
--------------
Helper functions for camera calibration and speed estimation.

Topic d (Image Analysis and Transformation):
    A perspective transform (homography) maps four points that the user clicks
    on the road (a rectangle of known real-world size) to a top-down "bird's-eye"
    plane measured in METERS. Distances measured on that plane are real distances,
    so speed = distance / time.
"""

import json
import cv2
import numpy as np

WINDOW = "Calibration - click 4 points"


def build_homography(image_pts, real_width_m, real_length_m):
    """
    image_pts : 4 points (x, y) clicked in the image, in this order:
                top-left, top-right, bottom-right, bottom-left
                of a road rectangle seen from the camera.
    real_width_m  : real width of that rectangle (left -> right) in meters.
    real_length_m : real length of that rectangle (top -> bottom) in meters.
    Returns a 3x3 matrix H that maps image pixels -> meters.
    """
    src = np.array(image_pts, dtype=np.float32)
    dst = np.array([[0, 0],
                    [real_width_m, 0],
                    [real_width_m, real_length_m],
                    [0, real_length_m]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def to_world(H, point_xy):
    """Convert one image point (x, y) to world meters (X, Y)."""
    p = np.array([[point_xy]], dtype=np.float32)      # shape (1, 1, 2)
    out = cv2.perspectiveTransform(p, H)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def calibrate_interactively(frame):
    """Let the user click the 4 calibration points on the first frame."""
    pts = []
    clone = frame.copy()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    labels = ["1: top-left", "2: top-right", "3: bottom-right", "4: bottom-left"]
    while True:
        view = clone.copy()
        for i, p in enumerate(pts):
            cv2.circle(view, p, 5, (0, 0, 255), -1)
            cv2.putText(view, str(i + 1), (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if len(pts) == 4:
            cv2.polylines(view, [np.array(pts, np.int32)], True, (0, 255, 0), 2)
            msg = "Press ENTER to accept, R to redo"
        else:
            msg = "Click point " + labels[len(pts)]
        cv2.putText(view, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            pts.clear()
        elif key in (13, 10) and len(pts) == 4:
            break
        elif key == 27:
            cv2.destroyWindow(WINDOW)
            raise SystemExit("Calibration cancelled.")
    cv2.destroyWindow(WINDOW)
    return pts


def save_calibration(path, pts, width_m, length_m):
    with open(path, "w") as f:
        json.dump({"points": pts, "width_m": width_m, "length_m": length_m}, f, indent=2)


def load_calibration(path):
    with open(path) as f:
        d = json.load(f)
    return [tuple(p) for p in d["points"]], d["width_m"], d["length_m"]


def estimate_speed_kmh(history):
    """
    history : deque of (timestamp_seconds, world_x, world_y)
    Speed is measured between the oldest and newest points in the window,
    which smooths out single-frame jitter in the centroid.
    """
    if len(history) < 5:
        return None
    t0, x0, y0 = history[0]
    t1, x1, y1 = history[-1]
    dt = t1 - t0
    if dt <= 0:
        return None
    dist_m = float(np.hypot(x1 - x0, y1 - y0))
    return (dist_m / dt) * 3.6          # m/s -> km/h
