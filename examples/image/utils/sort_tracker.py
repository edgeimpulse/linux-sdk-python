# To use this package, you will need to install filterpy and lap via `pip install filterpy lap`

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Track:
    def __init__(self, detection, track_id):
        self.track_id = track_id
        self.history = [detection]
        self.hits = 1
        self.no_losses = 0
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        self.kf.R *= 10.
        self.kf.Q[2:, 2:] *= 0.01
        self.kf.x[:2] = np.array(detection[:2]).reshape(2, 1)

class SORTTracker:
    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.1):
        """
        Args:
            max_age: Maximum number of frames to keep a track alive without detections.
            min_hits: Minimum number of detections to initialize a track.
            iou_threshold: IoU threshold for matching detections to tracks.
        """
        self.next_id = 1
        self.tracks = {}
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

    def update(self, detections):
        if not detections:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['no_losses'] += 1
                if self.tracks[track_id]['no_losses'] > self.max_age:
                    logger.debug(f"Removing track {track_id} due to max age.")
                    del self.tracks[track_id]
            return []

        # Predict
        for track_id in self.tracks:
            self.tracks[track_id]['kf'].predict()

        # Match detections to tracks using IoU
        cost_matrix = self._build_cost_matrix(detections)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_tracks = set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] > (1 - self.iou_threshold):
                continue
            track_id = list(self.tracks.keys())[r]
            self.tracks[track_id]['kf'].update(np.array(detections[c][:2]).reshape(2, 1))
            self.tracks[track_id]['history'].append(detections[c])
            self.tracks[track_id]['hits'] += 1
            self.tracks[track_id]['no_losses'] = 0
            assigned_tracks.add(track_id)

        # Create new tracks for unassigned detections
        for i, det in enumerate(detections):
            if i not in col_ind:
                self.tracks[self.next_id] = {
                    'kf': KalmanFilter(dim_x=4, dim_z=2),
                    'history': [det],
                    'hits': 1,
                    'no_losses': 0
                }
                self.tracks[self.next_id]['kf'].F = np.array([
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ])
                self.tracks[self.next_id]['kf'].H = np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0]
                ])
                self.tracks[self.next_id]['kf'].R *= 10.
                self.tracks[self.next_id]['kf'].Q[2:, 2:] *= 0.01
                self.tracks[self.next_id]['kf'].x[:2] = np.array(det[:2]).reshape(2, 1)
                self.next_id += 1

        # Remove lost tracks
        for track_id in list(self.tracks.keys()):
            if track_id not in assigned_tracks:
                self.tracks[track_id]['no_losses'] += 1
                if self.tracks[track_id]['no_losses'] > self.max_age:
                    logger.debug(f"Removing track {track_id} due to no detections.")
                    del self.tracks[track_id]

        return self._get_tracked_objects()

    def _build_cost_matrix(self, detections):
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        for i, track_id in enumerate(self.tracks):
            for j, det in enumerate(detections):
                iou = self._calculate_iou(self.tracks[track_id]['history'][-1], det)
                cost_matrix[i, j] = 1 - iou
        return cost_matrix

    def _calculate_iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        box1_coords = (x1 - w1/2, y1 - h1/2, x1 + w1/2, y1 + h1/2)
        box2_coords = (x2 - w2/2, y2 - h2/2, x2 + w2/2, y2 + h2/2)
        xi1, yi1 = max(box1_coords[0], box2_coords[0]), max(box1_coords[1], box2_coords[1])
        xi2, yi2 = min(box1_coords[2], box2_coords[2]), min(box1_coords[3], box2_coords[3])
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def _get_tracked_objects(self):
        tracked_objects = []
        for track_id, track in self.tracks.items():
            if track['hits'] >= self.min_hits:
                tracked_objects.append((track_id, track['history'][-1]))
        return tracked_objects
