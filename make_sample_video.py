"""
make_sample_video.py
--------------------
Creates a synthetic traffic video (sample_input/traffic.mp4) so the project can be
tested and demonstrated without a webcam or real traffic.

Cars move along a road with KNOWN speeds, drawn with real perspective, so the
speed estimated by main.py can be compared with the ground truth.
Also writes calibration.json that matches the scene.
"""

import json
import os
import cv2
import numpy as np

W, H_IMG, FPS = 960, 540, 30
ROAD_W_M, ROAD_L_M = 6.0, 20.0           # calibrated road rectangle (meters)

# image corners of the road rectangle: TL, TR, BR, BL
IMG_PTS = np.float32([[330, 120], [630, 120], [880, 500], [80, 500]])
WORLD_PTS = np.float32([[0, 0], [ROAD_W_M, 0], [ROAD_W_M, ROAD_L_M], [0, ROAD_L_M]])
H_W2I = cv2.getPerspectiveTransform(WORLD_PTS, IMG_PTS)   # world(m) -> image(px)

# (start_time_s, lane_x_m, speed_kmh, BGR color)
CARS = [(1.0, 1.5, 25, (60, 60, 230)),      # red
        (3.0, 4.5, 40, (225, 225, 225)),    # white
        (7.0, 1.5, 55, (30, 200, 230)),     # yellow
        (9.5, 4.5, 30, (210, 200, 50))]     # teal
CAR_W, CAR_L = 1.8, 4.2


def world_to_img(pts):
    p = np.array([pts], dtype=np.float32)
    return cv2.perspectiveTransform(p, H_W2I)[0]


def background():
    bg = np.full((H_IMG, W, 3), (70, 140, 70), np.uint8)             # grass
    # a wide road polygon extending beyond the calibrated rectangle
    road = world_to_img([[-1.5, -12], [ROAD_W_M + 1.5, -12],
                         [ROAD_W_M + 1.5, 22], [-1.5, 22]]).astype(np.int32)
    cv2.fillConvexPoly(bg, road, (60, 60, 60))
    # dashed center line
    for y in np.arange(-12, 22, 3.0):
        seg = world_to_img([[3 - 0.08, y], [3 + 0.08, y], [3 + 0.08, y + 1.5], [3 - 0.08, y + 1.5]])
        cv2.fillConvexPoly(bg, seg.astype(np.int32), (220, 220, 220))
    return bg


def main():
    os.makedirs("sample_input", exist_ok=True)
    bg = background()
    rng = np.random.default_rng(1)
    out = cv2.VideoWriter("sample_input/traffic.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                          FPS, (W, H_IMG))
    total = int(15 * FPS)
    for f in range(total):
        t = f / FPS
        frame = bg.copy()
        for (t0, lx, kmh, col) in CARS:
            if t < t0:
                continue
            y_center = -6 + (kmh / 3.6) * (t - t0)       # meters along the road
            if y_center - CAR_L / 2 > 22:
                continue
            body = [[lx - CAR_W / 2, y_center - CAR_L / 2], [lx + CAR_W / 2, y_center - CAR_L / 2],
                    [lx + CAR_W / 2, y_center + CAR_L / 2], [lx - CAR_W / 2, y_center + CAR_L / 2]]
            roof = [[lx - CAR_W / 2 + .2, y_center - CAR_L / 4], [lx + CAR_W / 2 - .2, y_center - CAR_L / 4],
                    [lx + CAR_W / 2 - .2, y_center + CAR_L / 4], [lx - CAR_W / 2 + .2, y_center + CAR_L / 4]]
            cv2.fillConvexPoly(frame, world_to_img(body).astype(np.int32), col)
            cv2.fillConvexPoly(frame, world_to_img(roof).astype(np.int32),
                               tuple(int(c * 0.6) for c in col))
        # light sensor noise like a real webcam
        noise = rng.normal(0, 3, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        out.write(frame)
    out.release()

    with open("calibration.json", "w") as fh:
        json.dump({"points": IMG_PTS.astype(int).tolist(),
                   "width_m": ROAD_W_M, "length_m": ROAD_L_M}, fh, indent=2)
    print("Wrote sample_input/traffic.mp4 and calibration.json")
    print("Ground-truth speeds (km/h):", [c[2] for c in CARS])


if __name__ == "__main__":
    main()
