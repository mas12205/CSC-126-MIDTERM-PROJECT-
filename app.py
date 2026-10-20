"""
app.py  -  Desktop launcher for the CSU Vehicle Speed Tracker
=============================================================
This is the entry point used to build the .exe. It shows a small window where
the user chooses the video source and settings, then starts the tracker
(main.py) with those settings.

    python app.py                      -> opens the launcher window
    python app.py --source 0 ...       -> skips the window, same options as main.py
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


def app_folder():
    """Folder of the .exe (when built) or of this script. Calibration and
    screenshots are saved here, so they are easy to find."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def build_args(use_webcam, webcam_index, video_path, width_m, length_m, limit, recalibrate):
    """Turn the launcher settings into the same command-line options main.py uses."""
    source = str(webcam_index) if use_webcam else video_path
    if recalibrate and os.path.exists("calibration.json"):
        os.remove("calibration.json")          # forces the 4-click calibration again
    return ["main.py", "--source", source,
            "--width-m", str(width_m), "--length-m", str(length_m),
            "--limit", str(limit)]


def run_tracker(argv):
    """Run the tracker in this process (the OpenCV window opens)."""
    import main                                # imported late so the launcher opens fast
    sys.argv = argv
    main.main()


class Launcher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CSU Vehicle Speed Tracker")
        self.resizable(False, False)
        pad = {"padx": 10, "pady": 4}

        ttk.Label(self, text="CSU Vehicle Speed Tracker", font=("Segoe UI", 14, "bold")
                  ).grid(row=0, column=0, columnspan=3, pady=(12, 2))
        ttk.Label(self, text="CSC 126 Computer Vision - OpenCV").grid(
            row=1, column=0, columnspan=3, pady=(0, 8))

        # --- source ---
        self.mode = tk.StringVar(value="webcam")
        ttk.Radiobutton(self, text="Live webcam, camera #", variable=self.mode,
                        value="webcam").grid(row=2, column=0, sticky="w", **pad)
        self.cam = tk.StringVar(value="0")
        ttk.Spinbox(self, from_=0, to=9, width=4, textvariable=self.cam
                    ).grid(row=2, column=1, sticky="w")
        ttk.Radiobutton(self, text="Video file", variable=self.mode,
                        value="video").grid(row=3, column=0, sticky="w", **pad)
        self.video = tk.StringVar()
        ttk.Entry(self, textvariable=self.video, width=34).grid(row=3, column=1, **pad)
        ttk.Button(self, text="Browse...", command=self.browse).grid(row=3, column=2, **pad)

        # --- settings ---
        ttk.Separator(self).grid(row=4, column=0, columnspan=3, sticky="ew", pady=6)
        self.width_m = tk.StringVar(value="6")
        self.length_m = tk.StringVar(value="10")
        self.limit = tk.StringVar(value="30")
        for r, (label, var) in enumerate([("Road rectangle width (m)", self.width_m),
                                          ("Road rectangle length (m)", self.length_m),
                                          ("Speed limit (km/h)", self.limit)], start=5):
            ttk.Label(self, text=label).grid(row=r, column=0, sticky="w", **pad)
            ttk.Entry(self, textvariable=var, width=8).grid(row=r, column=1, sticky="w", **pad)
        self.recal = tk.BooleanVar(value=not os.path.exists("calibration.json"))
        ttk.Checkbutton(self, text="Calibrate again (click 4 road corners, press ENTER)",
                        variable=self.recal).grid(row=8, column=0, columnspan=3,
                                                  sticky="w", **pad)

        ttk.Button(self, text="START", command=self.start).grid(
            row=9, column=0, columnspan=3, pady=(8, 4), ipadx=30, ipady=4)
        ttk.Label(self, text="In the video window:  Q = quit   P = pause   S = screenshot   C = recalibrate",
                  font=("Segoe UI", 8)).grid(row=10, column=0, columnspan=3, pady=(0, 10))

    def browse(self):
        path = filedialog.askopenfilename(
            title="Choose a video",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
        if path:
            self.video.set(path)
            self.mode.set("video")

    def start(self):
        try:
            width_m, length_m, limit = (float(self.width_m.get()),
                                        float(self.length_m.get()),
                                        float(self.limit.get()))
            if min(width_m, length_m, limit) <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid input", "Width, length and speed limit must be positive numbers.")
            return
        use_webcam = self.mode.get() == "webcam"
        if not use_webcam and not os.path.isfile(self.video.get()):
            messagebox.showerror("No video", "Please choose a video file.")
            return
        self.argv = build_args(use_webcam, self.cam.get(), self.video.get(),
                               width_m, length_m, limit, self.recal.get())
        self.destroy()                         # close the launcher, then run the tracker


def main():
    os.chdir(app_folder())                     # calibration.json / screenshots live next to the app
    if len(sys.argv) > 1:                      # command-line use: skip the window
        run_tracker(["main.py"] + sys.argv[1:])
        return
    win = Launcher()
    win.argv = None
    win.mainloop()
    if win.argv:
        try:
            run_tracker(win.argv)
        except SystemExit as e:                # main.py uses SystemExit for friendly errors
            if e.code not in (None, 0):
                _show_error(str(e.code))
        except Exception as e:                 # show crashes in a dialog (no console in the .exe)
            _show_error(f"{type(e).__name__}: {e}")


def _show_error(msg):
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Vehicle Speed Tracker", msg)
    root.destroy()


if __name__ == "__main__":
    main()
