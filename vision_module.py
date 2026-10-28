"""
Vision Module — Face detection, tracking & attendance marking
Part of: Multimodal Attendance & Engagement Agent
"""

import cv2
import numpy as np
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FaceTrack:
    track_id: int
    centroid: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    presence_duration: float = 0.0
    frames_present: int = 0
    attended: bool = False
    movement_history: List[Tuple[int, int]] = field(default_factory=list)

    def update(self, centroid, bbox, timestamp):
        interval = timestamp - self.last_seen
        if interval < 2.0:  # Only count if gap < 2s (not a new appearance)
            self.presence_duration += interval
        self.last_seen = timestamp
        self.centroid = centroid
        self.bbox = bbox
        self.frames_present += 1
        self.movement_history.append(centroid)
        if len(self.movement_history) > 50:
            self.movement_history.pop(0)


class CentroidTracker:
    """Simple centroid-based face tracker."""

    def __init__(self, max_disappeared=30, max_distance=80):
        self.next_id = 0
        self.tracks: Dict[int, FaceTrack] = {}
        self.disappeared: Dict[int, int] = defaultdict(int)
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _centroid(self, bbox):
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)

    def update(self, bboxes: List[Tuple], timestamp: float) -> Dict[int, FaceTrack]:
        if not bboxes:
            for tid in list(self.disappeared.keys()):
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    del self.tracks[tid]
                    del self.disappeared[tid]
            return self.tracks

        centroids = [self._centroid(b) for b in bboxes]

        if not self.tracks:
            for i, (cx, cy) in enumerate(centroids):
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = FaceTrack(
                    track_id=tid,
                    centroid=(cx, cy),
                    bbox=bboxes[i],
                    first_seen=timestamp,
                    last_seen=timestamp,
                )
                self.disappeared[tid] = 0
        else:
            track_ids = list(self.tracks.keys())
            track_centroids = [self.tracks[tid].centroid for tid in track_ids]

            # Compute distance matrix
            D = np.zeros((len(track_centroids), len(centroids)))
            for i, tc in enumerate(track_centroids):
                for j, ic in enumerate(centroids):
                    D[i, j] = np.sqrt((tc[0] - ic[0]) ** 2 + (tc[1] - ic[1]) ** 2)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                tid = track_ids[row]
                self.tracks[tid].update(centroids[col], bboxes[col], timestamp)
                self.disappeared[tid] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(len(track_ids))) - used_rows
            unused_cols = set(range(len(centroids))) - used_cols

            for row in unused_rows:
                tid = track_ids[row]
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    del self.tracks[tid]
                    del self.disappeared[tid]

            for col in unused_cols:
                tid = self.next_id
                self.next_id += 1
                cx, cy = centroids[col]
                self.tracks[tid] = FaceTrack(
                    track_id=tid,
                    centroid=(cx, cy),
                    bbox=bboxes[col],
                    first_seen=timestamp,
                    last_seen=timestamp,
                )
                self.disappeared[tid] = 0

        return self.tracks


class VisionModule:
    """
    Vision Module: Detects faces, tracks across frames, computes presence,
    and generates attendance records.
    """

    ATTENDANCE_THRESHOLD = 10  # seconds of presence to mark attended

    def __init__(self, source=0, attendance_threshold: float = None):
        """
        Args:
            source: 0 for webcam, or path to video file
            attendance_threshold: seconds of sustained presence to mark as attended
        """
        self.source = source
        self.attendance_threshold = attendance_threshold or self.ATTENDANCE_THRESHOLD
        self.tracker = CentroidTracker(max_disappeared=40, max_distance=100)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.frame_data: List[Dict] = []
        self.session_start = None
        self._running = False

    def detect_faces(self, frame) -> List[Tuple]:
        """Detect faces in a frame and return bounding boxes (x,y,w,h)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) == 0:
            return []
        return [tuple(f) for f in faces]

    def compute_movement(self, track: FaceTrack) -> float:
        """Compute average pixel movement of a tracked face."""
        history = track.movement_history
        if len(history) < 2:
            return 0.0
        movements = []
        for i in range(1, len(history)):
            dx = history[i][0] - history[i - 1][0]
            dy = history[i][1] - history[i - 1][1]
            movements.append(np.sqrt(dx ** 2 + dy ** 2))
        return float(np.mean(movements))

    def process_video(self, max_frames: int = None, display: bool = False) -> Dict:
        """
        Process video input and compute attendance + tracking data.

        Returns:
            Dict with attendance records and frame-level data
        """
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            logger.error(f"Cannot open video source: {self.source}")
            return {"attendance": {}, "frame_data": [], "error": "Cannot open source"}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.session_start = time.time()
        frame_count = 0
        self._running = True

        logger.info(f"Starting vision processing. Source: {self.source}, FPS: {fps}")

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    break
                if max_frames and frame_count >= max_frames:
                    break

                timestamp = self.session_start + frame_count / fps
                bboxes = self.detect_faces(frame)
                tracks = self.tracker.update(bboxes, timestamp)

                # Update attendance status
                for tid, track in tracks.items():
                    if (
                        not track.attended
                        and track.presence_duration >= self.attendance_threshold
                    ):
                        track.attended = True
                        logger.info(
                            f"Participant {tid} marked PRESENT "
                            f"(presence={track.presence_duration:.1f}s)"
                        )

                # Collect frame-level data
                frame_record = {
                    "frame": frame_count,
                    "timestamp": timestamp,
                    "faces_detected": len(bboxes),
                    "active_tracks": len(tracks),
                    "track_ids": list(tracks.keys()),
                }
                self.frame_data.append(frame_record)

                if display:
                    annotated = self._annotate_frame(frame, tracks)
                    cv2.imshow("Attendance Agent — Vision", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_count += 1

        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()

        return self.get_results()

    def _annotate_frame(self, frame, tracks: Dict[int, FaceTrack]):
        """Draw bounding boxes and track IDs on frame."""
        annotated = frame.copy()
        for tid, track in tracks.items():
            x, y, w, h = track.bbox
            color = (0, 255, 0) if track.attended else (0, 165, 255)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            label = f"ID:{tid} {'PRESENT' if track.attended else f'{track.presence_duration:.0f}s'}"
            cv2.putText(
                annotated, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
        cv2.putText(
            annotated,
            f"Faces: {len(tracks)} | Frame data collected",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        return annotated

    def get_results(self) -> Dict:
        """Return structured attendance and tracking results."""
        attendance_records = {}
        for tid, track in self.tracker.tracks.items():
            movement = self.compute_movement(track)
            attendance_records[f"participant_{tid}"] = {
                "track_id": tid,
                "attended": track.attended,
                "presence_duration": round(track.presence_duration, 2),
                "frames_present": track.frames_present,
                "first_seen": track.first_seen,
                "last_seen": track.last_seen,
                "avg_movement": round(movement, 2),
                "attendance_threshold": self.attendance_threshold,
            }

        return {
            "attendance": attendance_records,
            "frame_data": self.frame_data,
            "total_frames": len(self.frame_data),
            "session_start": self.session_start,
            "total_participants": len(attendance_records),
            "present_count": sum(
                1 for r in attendance_records.values() if r["attended"]
            ),
        }

    def stop(self):
        self._running = False


def generate_simulated_vision_data(
    num_participants: int = 8,
    session_duration: float = 60.0,
    fps: float = 1.0,
) -> Dict:
    """
    Generate simulated vision data for testing without camera.

    Args:
        num_participants: number of virtual participants
        session_duration: session length in seconds
        fps: frames per second (use 1.0 for lightweight simulation)
    Returns:
        Dict matching VisionModule.get_results() format
    """
    import random

    random.seed(42)
    np.random.seed(42)

    session_start = time.time() - session_duration  # simulate past session

    participants = {}
    frame_data = []
    total_frames = int(session_duration * fps)

    # Pre-generate attendance pattern per participant
    for pid in range(num_participants):
        name = f"participant_{pid}"
        # Each participant is present 50–100% of the session
        presence_ratio = random.uniform(0.5, 1.0)
        presence_duration = session_duration * presence_ratio
        attended = presence_duration >= 10.0  # threshold 10s
        movement = random.uniform(1.5, 12.0)
        participants[name] = {
            "track_id": pid,
            "attended": attended,
            "presence_duration": round(presence_duration, 2),
            "frames_present": int(presence_duration * fps),
            "first_seen": session_start + random.uniform(0, 5),
            "last_seen": session_start + presence_duration + random.uniform(0, 5),
            "avg_movement": round(movement, 2),
            "attendance_threshold": 10.0,
        }

    # Generate per-frame data
    for f in range(total_frames):
        ts = session_start + f / fps
        active = random.randint(max(1, num_participants - 2), num_participants)
        frame_data.append(
            {
                "frame": f,
                "timestamp": ts,
                "faces_detected": active,
                "active_tracks": active,
                "track_ids": list(range(active)),
            }
        )

    return {
        "attendance": participants,
        "frame_data": frame_data,
        "total_frames": total_frames,
        "session_start": session_start,
        "total_participants": num_participants,
        "present_count": sum(1 for p in participants.values() if p["attended"]),
    }


if __name__ == "__main__":
    # Quick smoke test with simulated data
    print("=== Vision Module — Simulated Test ===")
    results = generate_simulated_vision_data(num_participants=6, session_duration=60)
    print(f"Total participants: {results['total_participants']}")
    print(f"Present count: {results['present_count']}")
    for name, data in results["attendance"].items():
        status = "PRESENT" if data["attended"] else "ABSENT"
        print(f"  {name}: {status} ({data['presence_duration']:.1f}s)")
