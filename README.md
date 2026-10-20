# CSU Vehicle Speed Tracker

**CSC 126 – Computer Vision Midterm Project** · Caraga State University, College of Computing and Information Sciences

A classical OpenCV application that estimates the speed of vehicles from a **live webcam feed**
(or a recorded video) inside the Caraga State University campus. No deep learning is used.

## How it works

| Step | OpenCV technique | Lab topic |
|------|------------------|-----------|
| 1. Resize, grayscale, Gaussian blur | `cvtColor`, `GaussianBlur` | a. Image Operations |
| 2. Separate moving vehicles from the road | `createBackgroundSubtractorMOG2`, `threshold` | b. Image Segmentation |
| 3. Clean the mask | `morphologyEx` (open/close), `dilate` | a. Image Operations |
| 4. Find vehicles | `findContours`, `contourArea`, `boundingRect` | d. Image Analysis |
| 5. Pixels → meters | `getPerspectiveTransform`, `perspectiveTransform` | d. Transformation |
| 6. Track vehicles and compute speed | centroid tracker (`tracker.py`) | e. Motion & Object Tracking |

**Speed formula:** each vehicle's centroid is mapped to real-world meters on the road plane.
Speed = distance travelled ÷ elapsed time over a short window, converted to km/h, then
smoothed with a median of the last 10 readings. Vehicles above the speed limit are drawn in **red**.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```

## Run

**1. Try it with the included sample video (no webcam needed)**

```bash
python make_sample_video.py          # creates sample_input/traffic.mp4 + calibration.json
python main.py --source sample_input/traffic.mp4 --calib calibration.json
```

**2. Live webcam at the school**

```bash
python main.py --source 0 --width-m 6 --length-m 10 --limit 30
```

On the first run a window opens: **click 4 points** on the road forming a rectangle
(top-left, top-right, bottom-right, bottom-left), then press **ENTER**.
Use `--width-m` and `--length-m` for the **real size of that rectangle in meters**
(measure it with a tape measure or use a known lane width and the length between two
painted marks). The calibration is saved in `calibration.json`; press **C** to redo it.

### Keys
`Q`/`ESC` quit · `P` pause · `S` screenshot (saved in `screenshots/`) · `C` recalibrate

### Useful options
`--limit 30` speed limit (km/h) · `--min-area 1200` ignore small blobs ·
`--save-video out.mp4` record the result · `--headless --max-frames 200 --auto-shots 50` run without a window

## Camera placement tips (important for accuracy)
- Mount the camera high, looking down the road at an angle, and keep it **fixed**.
- Calibrate with a rectangle as large as possible, laid on the road surface.
- Good daylight, no heavy shadows; one vehicle per lane in the region works best.

## Project files
```
main.py               application (pipeline, overlay, UI)
tracker.py            centroid object tracker
speed_utils.py        calibration + perspective transform + speed formula
make_sample_video.py  creates the synthetic test video
requirements.txt      dependencies
sample_input/         sample video
screenshots/          output screenshots
```

## Limitations
Accuracy depends on calibration and camera angle; vehicles that overlap are merged into one
blob; strong shadows, rain or camera shake can create false detections.