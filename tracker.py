"""
tracker.py
----------
A simple centroid-based multi-object tracker (Topic e: Motion and Object Tracking).

Each detected vehicle is given a unique ID. On every frame we match the new
detections to the existing tracks by nearest-centroid distance. A track that
is not matched for `max_missing` frames is removed.
"""

from collections import deque
import math


class Track:
    """One tracked vehicle."""

    def __init__(self, track_id, centroid, bbox, frame_idx, history_len=30):
        self.id = track_id
        self.centroid = centroid          # (x, y) in image pixels
        self.bbox = bbox                  # (x, y, w, h)
        self.missing = 0                  # consecutive frames without a match
        self.age = 1                      # number of frames this track has lived
        # history of (frame_idx, world_x, world_y) used for speed estimation
        self.world_history = deque(maxlen=history_len)
        self.speeds_kmh = deque(maxlen=10)  # recent speed values for smoothing
        self.speed_kmh = None             # latest smoothed speed
        self.max_speed_kmh = 0.0
        self.counted = False              # already added to the vehicle count?
        self.first_frame = frame_idx


class CentroidTracker:
    def __init__(self, max_distance=80, max_missing=10):
        """
        max_distance : max pixel distance between a track and a detection
                       for them to be considered the same vehicle.
        max_missing  : frames a track may go unmatched before it is deleted.
        """
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.tracks = {}
        self._next_id = 1

    @staticmethod
    def _dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def update(self, detections, frame_idx):
        """
        detections : list of (centroid, bbox)
        Returns the dict of active tracks {id: Track}.
        """
        # Greedy nearest-neighbour matching (sorted by distance)
        pairs = []
        for tid, tr in self.tracks.items():
            for di, (c, _) in enumerate(detections):
                d = self._dist(tr.centroid, c)
                if d <= self.max_distance:
                    pairs.append((d, tid, di))
        pairs.sort()

        used_tracks, used_dets = set(), set()
        for d, tid, di in pairs:
            if tid in used_tracks or di in used_dets:
                continue
            c, bbox = detections[di]
            tr = self.tracks[tid]
            tr.centroid, tr.bbox = c, bbox
            tr.missing = 0
            tr.age += 1
            used_tracks.add(tid)
            used_dets.add(di)

        # Unmatched tracks -> mark missing, delete if too old
        for tid in list(self.tracks.keys()):
            if tid not in used_tracks:
                self.tracks[tid].missing += 1
                if self.tracks[tid].missing > self.max_missing:
                    del self.tracks[tid]

        # Unmatched detections -> new tracks
        for di, (c, bbox) in enumerate(detections):
            if di not in used_dets:
                self.tracks[self._next_id] = Track(self._next_id, c, bbox, frame_idx)
                self._next_id += 1

        return self.tracks
