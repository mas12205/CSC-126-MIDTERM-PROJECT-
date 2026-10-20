"""
main.py  -  Vehicle Speed Tracker (CSC 126 Computer Vision Midterm Project)
===========================================================================
Estimates the speed of vehicles from a live webcam feed (or a recorded video)
using classical OpenCV techniques only - no deep learning.

OpenCV topics from the laboratory used in this project:
    a. Image Operations ............ grayscale, Gaussian blur, resizing,
                                      morphological opening/closing/dilation
    b. Image Segmentation .......... MOG2 background subtraction + thresholding
                                      to separate moving vehicles from the road
    d. Image Analysis & Transform .. contour extraction, bounding boxes, area
                                      filtering, perspective transform (pixels
                                      -> meters on the road plane)
    e. Motion & Object Tracking .... centroid tracker with unique IDs and
                                      speed computed from motion over time

Usage examples
--------------
    python main.py --source 0                       # live webcam
    python main.py --source sample_input/traffic.mp4 --calib calibration.json
    python main.py --source 0 --width-m 7 --length-m 12 --limit 30

Keys while running:  Q / ESC = quit,  P = pause,  S = save screenshot,
                     C = recalibrate
"""

import argparse
import os
import time

import cv2
import numpy as np

from tracker import CentroidTracker
import speed_utils as su


# ----------------------------------------------------------------------------
# Command-line arguments
# ----------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Vehicle speed tracker (OpenCV)")
    ap.add_argument("--source", default="0",
                    help="0 for webcam, or a path to a video file")
    ap.add_argument("--calib", default="calibration.json",
                    help="calibration file (created if missing)")
    ap.add_argument("--width-m", type=float, default=6.0,
                    help="real width (m) of the calibration rectangle")
    ap.add_argument("--length-m", type=float, default=10.0,
                    help="real length (m) of the calibration rectangle")
    ap.add_argument("--limit", type=float, default=30.0,
                    help="campus speed limit in km/h (vehicles above it turn red)")
    ap.add_argument("--min-area", type=int, default=1200,
                    help="minimum contour area (pixels) to count as a vehicle")
    ap.add_argument("--width", type=int, default=960,
                    help="frames are resized to this width for processing")
    ap.add_argument("--save-video", default=None,
                    help="optional path to save the processed video (.mp4)")
    ap.add_argument("--shots-dir", default="screenshots",
                    help="folder for screenshots")
    ap.add_argument("--headless", action="store_true",
                    help="do not open windows (for testing); needs --calib file")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="stop after N frames (0 = no limit)")
    ap.add_argument("--auto-shots", type=int, default=0,
                    help="in headless mode, save a screenshot every N frames")
    return ap.parse_args()


# ----------------------------------------------------------------------------
# Processing steps (each one is a small function so the pipeline is easy to read)
# ----------------------------------------------------------------------------
def preprocess(frame):
    """Topic a - Image Operations: grayscale + Gaussian blur to reduce noise."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def segment_motion(blurred, subtractor, kernel):
    """
    Topic b - Image Segmentation.
    MOG2 learns the static background; moving pixels become the foreground mask.
    Shadows are marked 127 by MOG2, so thresholding at 200 removes them.
    Morphology cleans the mask (Topic a again).
    """
    fg = subtractor.apply(blurred)
    _, mask = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)    # remove specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)   # fill holes
    mask = cv2.dilate(mask, kernel, iterations=2)                          # merge parts
    return mask


def find_vehicles(mask, min_area, roi_poly):
    """
    Topic d - Image Analysis.
    Extract contours from the mask, drop small ones (noise, people, leaves) and
    keep only blobs whose centroid is inside the calibrated road region.
    Returns a list of (centroid, bbox).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        # keep only objects inside the road rectangle that was calibrated
        if cv2.pointPolygonTest(roi_poly, (float(cx), float(cy)), False) < 0:
            continue
        detections.append(((cx, cy), (x, y, w, h)))
    return detections


def draw_overlay(frame, tracks, roi_poly, mask, stats, limit):
    """Draw the road region, boxes, IDs, speeds, mask inset and stats panel."""
    out = frame.copy()
    cv2.polylines(out, [roi_poly], True, (255, 200, 0), 2)

    for tr in tracks.values():
        if tr.missing > 0:
            continue
        x, y, w, h = tr.bbox
        speeding = tr.speed_kmh is not None and tr.speed_kmh > limit
        color = (0, 0, 255) if speeding else (0, 220, 0)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        label = f"ID {tr.id}"
        if tr.speed_kmh is not None:
            label += f"  {tr.speed_kmh:4.1f} km/h"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (x, y - th - 8), (x + tw + 6, y), color, -1)
        cv2.putText(out, label, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2)

    # Foreground-mask inset (shows the segmentation result)
    ih, iw = 120, 160
    inset = cv2.resize(mask, (iw, ih))
    inset = cv2.cvtColor(inset, cv2.COLOR_GRAY2BGR)
    out[10:10 + ih, out.shape[1] - iw - 10: out.shape[1] - 10] = inset
    cv2.rectangle(out, (out.shape[1] - iw - 10, 10),
                  (out.shape[1] - 10, 10 + ih), (255, 255, 255), 1)
    cv2.putText(out, "Motion mask", (out.shape[1] - iw - 5, 10 + ih + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Stats panel
    panel = [f"CSU Vehicle Speed Tracker",
             f"FPS: {stats['fps']:.1f}",
             f"Vehicles counted: {stats['count']}",
             f"Speeding (> {limit:.0f} km/h): {stats['speeding']}",
             f"Top speed: {stats['top']:.1f} km/h"]
    cv2.rectangle(out, (5, 5), (290, 10 + 22 * len(panel)), (0, 0, 0), -1)
    for i, line in enumerate(panel):
        cv2.putText(out, line, (12, 25 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1)
    return out


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    src = int(args.source) if args.source.isdigit() else args.source
    is_live = isinstance(src, int)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video source: {args.source}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps != video_fps or video_fps < 1:
        video_fps = 30.0   # webcams sometimes report 0

    ok, first = cap.read()
    if not ok:
        raise SystemExit("Could not read the first frame.")
    scale = args.width / first.shape[1]
    first = cv2.resize(first, (args.width, int(first.shape[0] * scale)))

    # ---- Calibration (pixels -> meters) ------------------------------------
    if os.path.exists(args.calib):
        pts, wm, lm = su.load_calibration(args.calib)
        print(f"Loaded calibration from {args.calib}  ({wm} m x {lm} m)")
    else:
        if args.headless:
            raise SystemExit("Headless mode needs an existing --calib file.")
        pts = su.calibrate_interactively(first)
        wm, lm = args.width_m, args.length_m
        su.save_calibration(args.calib, pts, wm, lm)
        print(f"Saved calibration to {args.calib}")
    H = su.build_homography(pts, wm, lm)
    roi_poly = np.array(pts, np.int32)

    # ---- Objects used every frame -----------------------------------------
    subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40,
                                                    detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tracker = CentroidTracker(max_distance=90, max_missing=8)
    os.makedirs(args.shots_dir, exist_ok=True)

    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_video, fourcc, video_fps,
                                 (first.shape[1], first.shape[0]))

    stats = {"fps": 0.0, "count": 0, "speeding": 0, "top": 0.0}
    speeders = set()
    speed_log = {}             # track id -> list of speed readings (for the final report)
    frame_idx = 0
    paused = False
    t_prev = time.time()
    pending = first            # first frame is processed before reading more

    while True:
        t_loop_start = time.time()

        
        if not paused:
            if pending is not None:
                frame, pending = pending, None
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.resize(frame, (args.width, int(frame.shape[0] * scale)))
            frame_idx += 1

            # Timestamp: real clock for webcam, frame number / fps for video files
            # (so a video file gives correct speeds even if processing is slow).
            now = time.time()
            ts = now if is_live else frame_idx / video_fps

            # 1) Image operations  2) Segmentation  3) Contour analysis
            blurred = preprocess(frame)
            mask = segment_motion(blurred, subtractor, kernel)
            detections = find_vehicles(mask, args.min_area, roi_poly)

            # 4) Tracking + speed estimation
            tracks = tracker.update(detections, frame_idx)
            for tr in tracks.values():
                if tr.missing > 0:
                    continue
                wx, wy = su.to_world(H, tr.centroid)
                tr.world_history.append((ts, wx, wy))
                v = su.estimate_speed_kmh(tr.world_history)
                if v is not None:
                    tr.speeds_kmh.append(v)
                    tr.speed_kmh = float(np.median(tr.speeds_kmh))   # median = robust
                    tr.max_speed_kmh = max(tr.max_speed_kmh, tr.speed_kmh)
                    speed_log.setdefault(tr.id, []).append(tr.speed_kmh)
                    stats["top"] = max(stats["top"], tr.speed_kmh)
                    if tr.speed_kmh > args.limit:
                        speeders.add(tr.id)
                # count a vehicle once it has been tracked for a few frames
                if tr.age >= 8 and not tr.counted:
                    tr.counted = True
                    stats["count"] += 1
            stats["speeding"] = len(speeders)
            stats["fps"] = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now

            out = draw_overlay(frame, tracks, roi_poly, mask, stats, args.limit)
            if writer:
                writer.write(out)
            if args.headless and args.auto_shots and frame_idx % args.auto_shots == 0:
                cv2.imwrite(os.path.join(args.shots_dir, f"frame_{frame_idx:04d}.png"), out)
            if args.max_frames and frame_idx >= args.max_frames:
                break

        if not args.headless:
            cv2.imshow("Vehicle Speed Tracker", out)
            key = cv2.waitKey(1) & 0xFF
            if is_live:
                delay = 1
            else:
                # wait long enough that the video plays at its real frame rate
                elapsed_ms = (time.time() - t_loop_start) * 1000
                delay = max(1, int(1000 / video_fps - elapsed_ms))
            key = cv2.waitKey(delay) & 0xFF

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Done. Frames: {frame_idx}, vehicles counted: {stats['count']}, "
          f"speeding: {stats['speeding']}, top speed: {stats['top']:.1f} km/h")
    # Per-vehicle summary: median of all readings for each tracked vehicle
    for vid, vals in speed_log.items():
        if len(vals) >= 10:
            print(f"  Vehicle ID {vid}: median speed {np.median(vals):.1f} km/h "
                  f"({len(vals)} readings)")

            


if __name__ == "__main__":
    main()
