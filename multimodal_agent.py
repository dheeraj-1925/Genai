"""
Multimodal Agent — Signal fusion, engagement scoring & trend analysis
Part of: Multimodal Attendance & Engagement Agent
"""

import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EngagementScore:
    timestamp: float
    interval_start: float
    interval_end: float
    score: float           # 0.0 to 1.0
    level: str             # "high", "medium", "low"
    visual_contribution: float
    audio_contribution: float
    face_count: int
    speech_detected: bool
    noise_level: str
    movement_intensity: float


class MultimodalAgent:
    """
    Multimodal Agent: Fuses vision & audio signals, computes engagement scores
    and generates temporal engagement profiles.
    """

    HIGH_ENGAGEMENT = 0.7
    MEDIUM_ENGAGEMENT = 0.4

    # Weights for multimodal fusion
    VISUAL_WEIGHT = 0.6
    AUDIO_WEIGHT = 0.4

    def __init__(self, interval_seconds: float = 10.0):
        """
        Args:
            interval_seconds: Time window for engagement scoring
        """
        self.interval_seconds = interval_seconds
        self.engagement_scores: List[EngagementScore] = []

    def _classify_level(self, score: float) -> str:
        if score >= self.HIGH_ENGAGEMENT:
            return "high"
        elif score >= self.MEDIUM_ENGAGEMENT:
            return "medium"
        return "low"

    def _visual_score_for_window(
        self,
        frame_data: List[Dict],
        attendance: Dict,
        window_start: float,
        window_end: float,
    ) -> Tuple[float, float, int]:
        """
        Compute visual engagement score for a time window.
        Returns (score, movement_intensity, avg_face_count)
        """
        window_frames = [
            f for f in frame_data
            if window_start <= f["timestamp"] < window_end
        ]
        if not window_frames:
            return 0.0, 0.0, 0

        face_counts = [f["faces_detected"] for f in window_frames]
        avg_faces = float(np.mean(face_counts))
        max_faces = max(
            (v["track_id"] for v in attendance.values()), default=1
        ) + 1  # approximate class size

        # Presence score: ratio of avg faces to expected participants
        presence_score = min(avg_faces / max(max_faces, 1), 1.0)

        # Movement intensity (0 = no movement, 1 = very active)
        # Collect avg_movement from attendance records
        movements = [v["avg_movement"] for v in attendance.values()]
        if movements:
            norm_movement = min(float(np.mean(movements)) / 15.0, 1.0)
        else:
            norm_movement = 0.0

        # Inactivity penalty: if std of face counts is very low + few faces
        inactivity_penalty = 0.0
        if np.std(face_counts) < 0.5 and avg_faces < 2:
            inactivity_penalty = 0.2

        visual_score = max(0.0, presence_score * 0.7 + norm_movement * 0.3 - inactivity_penalty)
        return round(min(visual_score, 1.0), 3), round(norm_movement, 3), int(avg_faces)

    def _audio_score_for_window(
        self,
        audio_segments: List[Dict],
        window_start: float,
        window_end: float,
    ) -> Tuple[float, bool, str]:
        """
        Compute audio engagement score for a time window.
        Returns (score, speech_detected, dominant_noise_level)
        """
        window_segs = [
            s for s in audio_segments
            if window_start <= s["start_time"] < window_end
        ]
        if not window_segs:
            return 0.3, False, "low"  # neutral default

        speech_count = sum(1 for s in window_segs if s["speech_detected"])
        speech_ratio = speech_count / len(window_segs)

        noise_counts = {"low": 0, "medium": 0, "high": 0}
        for s in window_segs:
            noise_counts[s["noise_level"]] += 1
        dominant_noise = max(noise_counts, key=noise_counts.get)

        # Speech boosts engagement; high noise reduces it slightly
        audio_score = speech_ratio * 0.8
        if dominant_noise == "high":
            audio_score = max(0.0, audio_score - 0.15)
        elif dominant_noise == "medium":
            audio_score += 0.1

        return round(min(audio_score, 1.0), 3), speech_count > 0, dominant_noise

    def fuse_and_score(
        self,
        vision_results: Dict,
        audio_results: Dict,
        session_start: float,
        session_duration: float,
    ) -> List[EngagementScore]:
        """
        Fuse multimodal signals and compute engagement scores per interval.

        Args:
            vision_results: Output from VisionModule.get_results()
            audio_results: Output from AudioModule.get_results()
            session_start: Unix timestamp of session start
            session_duration: Total session length in seconds

        Returns:
            List of EngagementScore objects for each interval
        """
        frame_data = vision_results.get("frame_data", [])
        attendance = vision_results.get("attendance", {})
        audio_segments = audio_results.get("segments", [])

        self.engagement_scores = []
        t = session_start

        while t < session_start + session_duration:
            window_end = min(t + self.interval_seconds, session_start + session_duration)

            v_score, movement, face_count = self._visual_score_for_window(
                frame_data, attendance, t, window_end
            )
            a_score, speech, noise_level = self._audio_score_for_window(
                audio_segments, t, window_end
            )

            # Weighted fusion
            fused = self.VISUAL_WEIGHT * v_score + self.AUDIO_WEIGHT * a_score
            level = self._classify_level(fused)

            self.engagement_scores.append(
                EngagementScore(
                    timestamp=t,
                    interval_start=t,
                    interval_end=window_end,
                    score=round(fused, 3),
                    level=level,
                    visual_contribution=v_score,
                    audio_contribution=a_score,
                    face_count=face_count,
                    speech_detected=speech,
                    noise_level=noise_level,
                    movement_intensity=movement,
                )
            )
            t += self.interval_seconds

        logger.info(f"Computed {len(self.engagement_scores)} engagement intervals")
        return self.engagement_scores

    def get_engagement_results(self) -> Dict:
        """Return structured engagement analysis results."""
        if not self.engagement_scores:
            return {"intervals": [], "summary": {}}

        scores = [e.score for e in self.engagement_scores]
        level_counts = {"high": 0, "medium": 0, "low": 0}
        for e in self.engagement_scores:
            level_counts[e.level] += 1

        intervals = []
        for e in self.engagement_scores:
            intervals.append({
                "interval_start": e.interval_start,
                "interval_end": e.interval_end,
                "score": e.score,
                "level": e.level,
                "visual_contribution": e.visual_contribution,
                "audio_contribution": e.audio_contribution,
                "face_count": e.face_count,
                "speech_detected": e.speech_detected,
                "noise_level": e.noise_level,
                "movement_intensity": e.movement_intensity,
            })

        return {
            "intervals": intervals,
            "summary": {
                "avg_engagement": round(float(np.mean(scores)), 3),
                "max_engagement": round(float(np.max(scores)), 3),
                "min_engagement": round(float(np.min(scores)), 3),
                "std_engagement": round(float(np.std(scores)), 3),
                "level_distribution": level_counts,
                "dominant_level": max(level_counts, key=level_counts.get),
                "total_intervals": len(self.engagement_scores),
                "high_engagement_ratio": round(
                    level_counts["high"] / len(self.engagement_scores), 3
                ),
            },
        }


def generate_simulated_engagement_data(
    session_duration: float = 60.0,
    interval_seconds: float = 10.0,
) -> Dict:
    """
    Generate simulated engagement data without running actual modules.
    """
    import random
    random.seed(42)
    np.random.seed(42)

    session_start = time.time() - session_duration
    num_intervals = int(session_duration / interval_seconds)
    intervals = []
    level_counts = {"high": 0, "medium": 0, "low": 0}

    for i in range(num_intervals):
        # Simulate realistic engagement curve: mid-session dip
        base = 0.7
        if i == num_intervals // 2:
            base = 0.4  # attention dip
        elif i == num_intervals - 1:
            base = 0.6
        score = round(min(1.0, max(0.0, base + random.uniform(-0.2, 0.2))), 3)
        level = "high" if score >= 0.7 else ("medium" if score >= 0.4 else "low")
        level_counts[level] += 1

        v_score = round(score + random.uniform(-0.1, 0.1), 3)
        a_score = round(score + random.uniform(-0.15, 0.15), 3)
        v_score = max(0.0, min(1.0, v_score))
        a_score = max(0.0, min(1.0, a_score))

        speech = a_score > 0.3
        noise = "high" if a_score > 0.7 else ("medium" if a_score > 0.4 else "low")

        intervals.append({
            "interval_start": session_start + i * interval_seconds,
            "interval_end": session_start + (i + 1) * interval_seconds,
            "score": score,
            "level": level,
            "visual_contribution": v_score,
            "audio_contribution": a_score,
            "face_count": random.randint(4, 8),
            "speech_detected": speech,
            "noise_level": noise,
            "movement_intensity": round(random.uniform(0.1, 0.9), 3),
        })

    scores = [iv["score"] for iv in intervals]
    return {
        "intervals": intervals,
        "summary": {
            "avg_engagement": round(float(np.mean(scores)), 3),
            "max_engagement": round(float(np.max(scores)), 3),
            "min_engagement": round(float(np.min(scores)), 3),
            "std_engagement": round(float(np.std(scores)), 3),
            "level_distribution": level_counts,
            "dominant_level": max(level_counts, key=level_counts.get),
            "total_intervals": num_intervals,
            "high_engagement_ratio": round(level_counts["high"] / num_intervals, 3),
        },
    }


if __name__ == "__main__":
    print("=== Multimodal Agent — Simulated Test ===")
    data = generate_simulated_engagement_data(session_duration=60, interval_seconds=10)
    s = data["summary"]
    print(f"Avg engagement: {s['avg_engagement']}")
    print(f"Dominant level: {s['dominant_level']}")
    print(f"Level distribution: {s['level_distribution']}")
    for iv in data["intervals"]:
        print(f"  [{iv['level'].upper():6s}] score={iv['score']} | speech={iv['speech_detected']} | noise={iv['noise_level']}")
