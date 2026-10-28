"""
Analytics Engine — Teaching suggestions & student session analytics
Part of: Multimodal Attendance & Engagement Agent
"""

import numpy as np
import time
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Teaching Suggestion Rules
# ─────────────────────────────────────────────

TEACHING_RULES = [
    {
        "id": "sustained_low_engagement",
        "condition": lambda intervals, i: (
            i >= 2
            and all(intervals[j]["score"] < 0.4 for j in range(max(0, i - 2), i + 1))
        ),
        "suggestion": "[HIGH] Extended low engagement detected. Consider introducing an interactive activity, quiz, or group discussion to re-energize participants.",
        "priority": "high",
    },
    {
        "id": "attention_drop",
        "condition": lambda intervals, i: (
            i >= 1
            and intervals[i]["score"] < 0.4
            and intervals[i - 1]["score"] >= 0.6
        ),
        "suggestion": "[MEDIUM] Sudden attention drop detected. A brief break, change of topic, or quick recap may help restore focus.",
        "priority": "medium",
    },
    {
        "id": "high_noise",
        "condition": lambda intervals, i: intervals[i]["noise_level"] == "high",
        "suggestion": "[MEDIUM] High ambient noise detected. Consider pausing for noise management or requesting quieter participation.",
        "priority": "medium",
    },
    {
        "id": "peak_engagement",
        "condition": lambda intervals, i: intervals[i]["score"] >= 0.8,
        "suggestion": "[INFO] Peak engagement moment! This is an excellent time to introduce complex concepts or important material.",
        "priority": "info",
    },
    {
        "id": "low_speech",
        "condition": lambda intervals, i: (
            i >= 2
            and all(not intervals[j]["speech_detected"] for j in range(max(0, i - 2), i + 1))
        ),
        "suggestion": "[MEDIUM] Low verbal interaction for an extended period. Encourage questions, pair discussions, or open-ended prompts.",
        "priority": "medium",
    },
    {
        "id": "recovery_trend",
        "condition": lambda intervals, i: (
            i >= 2
            and intervals[i]["score"] > intervals[i - 1]["score"] > intervals[i - 2]["score"]
            and intervals[i - 2]["score"] < 0.5
        ),
        "suggestion": "[INFO] Positive engagement recovery trend detected. Current teaching approach is working well - maintain the pace.",
        "priority": "info",
    },
    {
        "id": "low_presence",
        "condition": lambda intervals, i: intervals[i].get("face_count", 5) < 3,
        "suggestion": "[HIGH] Low participant presence detected. Verify attendance and consider ways to encourage re-engagement.",
        "priority": "high",
    },
]


class AnalyticsEngine:
    """
    Analytics Engine: Generates teaching suggestions and student analytics
    from multimodal engagement data.
    """

    def __init__(self):
        self.suggestions: List[Dict] = []
        self.student_analytics: List[Dict] = []

    def generate_teaching_suggestions(self, engagement_results: Dict) -> List[Dict]:
        """
        Apply rule-based logic to engagement intervals and generate suggestions.

        Args:
            engagement_results: Output from MultimodalAgent.get_engagement_results()

        Returns:
            List of teaching suggestion dicts
        """
        intervals = engagement_results.get("intervals", [])
        self.suggestions = []
        seen_ids = set()

        for i, interval in enumerate(intervals):
            for rule in TEACHING_RULES:
                try:
                    if rule["condition"](intervals, i):
                        # Avoid duplicate suggestions of same type in consecutive windows
                        key = f"{rule['id']}_{i}"
                        if rule["id"] not in seen_ids or rule["priority"] == "high":
                            self.suggestions.append({
                                "rule_id": rule["id"],
                                "interval_index": i,
                                "interval_start": interval["interval_start"],
                                "interval_end": interval["interval_end"],
                                "engagement_score": interval["score"],
                                "suggestion": rule["suggestion"],
                                "priority": rule["priority"],
                                "timestamp": interval["interval_start"],
                            })
                            seen_ids.add(rule["id"])
                        else:
                            seen_ids.discard(rule["id"])  # reset to allow re-trigger
                except Exception as e:
                    logger.warning(f"Rule {rule['id']} failed at interval {i}: {e}")

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "info": 2}
        self.suggestions.sort(key=lambda s: priority_order.get(s["priority"], 3))

        logger.info(f"Generated {len(self.suggestions)} teaching suggestions")
        return self.suggestions

    def generate_student_analytics(
        self,
        vision_results: Dict,
        engagement_results: Dict,
    ) -> List[Dict]:
        """
        Generate individual student analytics from vision and engagement data.

        Args:
            vision_results: Output from VisionModule.get_results()
            engagement_results: Output from MultimodalAgent.get_engagement_results()

        Returns:
            List of per-student analytics dicts
        """
        attendance = vision_results.get("attendance", {})
        intervals = engagement_results.get("intervals", [])
        session_start = vision_results.get("session_start", time.time())

        self.student_analytics = []

        for student_id, record in attendance.items():
            # Engagement intervals where student was present
            first_seen = record.get("first_seen", session_start)
            last_seen = record.get("last_seen", session_start)

            active_intervals = [
                iv for iv in intervals
                if first_seen <= iv["interval_end"] and last_seen >= iv["interval_start"]
            ]

            if active_intervals:
                engagement_scores = [iv["score"] for iv in active_intervals]
                avg_engagement = float(np.mean(engagement_scores))
                engagement_trend = self._compute_trend(engagement_scores)
                peak_interval = max(active_intervals, key=lambda x: x["score"])
                low_interval = min(active_intervals, key=lambda x: x["score"])
                high_count = sum(1 for s in engagement_scores if s >= 0.7)
                low_count = sum(1 for s in engagement_scores if s < 0.4)
                engagement_consistency = 1.0 - float(np.std(engagement_scores))
            else:
                avg_engagement = 0.0
                engagement_trend = "unavailable"
                peak_interval = None
                low_interval = None
                high_count = 0
                low_count = 0
                engagement_consistency = 0.0

            analytics = {
                "student_id": student_id,
                "track_id": record["track_id"],
                "attended": record["attended"],
                "presence_duration": record["presence_duration"],
                "avg_movement": record.get("avg_movement", 0.0),
                "avg_engagement": round(avg_engagement, 3),
                "engagement_trend": engagement_trend,
                "engagement_consistency": round(max(0.0, engagement_consistency), 3),
                "high_engagement_intervals": high_count,
                "low_engagement_intervals": low_count,
                "total_active_intervals": len(active_intervals),
                "peak_engagement_score": round(peak_interval["score"], 3) if peak_interval else 0.0,
                "peak_engagement_time": peak_interval["interval_start"] if peak_interval else None,
                "lowest_engagement_score": round(low_interval["score"], 3) if low_interval else 0.0,
                "reflective_summary": self._generate_reflective_summary(
                    record, avg_engagement, engagement_trend, high_count, low_count
                ),
            }
            self.student_analytics.append(analytics)

        # Sort attended first, then by engagement
        self.student_analytics.sort(
            key=lambda x: (-int(x["attended"]), -x["avg_engagement"])
        )

        logger.info(f"Generated analytics for {len(self.student_analytics)} students")
        return self.student_analytics

    def _compute_trend(self, scores: List[float]) -> str:
        """Compute simple linear trend direction."""
        if len(scores) < 2:
            return "stable"
        first_half = float(np.mean(scores[: len(scores) // 2]))
        second_half = float(np.mean(scores[len(scores) // 2 :]))
        diff = second_half - first_half
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "declining"
        return "stable"

    def _generate_reflective_summary(
        self,
        record: Dict,
        avg_engagement: float,
        trend: str,
        high_count: int,
        low_count: int,
    ) -> str:
        """Generate a reflective, non-evaluative summary for the student."""
        name = record.get("student_id", "You")
        attended = record.get("attended", False)
        duration = record.get("presence_duration", 0)

        if not attended:
            return (
                "Your session presence was below the attendance threshold. "
                "Aim to be present for a longer portion of the session."
            )

        lines = [f"You were present for {duration:.0f} seconds during this session."]

        if avg_engagement >= 0.7:
            lines.append("Your engagement level was consistently high — great focus!")
        elif avg_engagement >= 0.4:
            lines.append("Your engagement was moderate throughout the session.")
        else:
            lines.append(
                "Your engagement was lower than expected during this session. "
                "Consider reducing distractions in your environment."
            )

        if trend == "improving":
            lines.append("Your attention improved as the session progressed.")
        elif trend == "declining":
            lines.append(
                "Your engagement dipped toward the end of the session. "
                "Taking short mental breaks between sessions may help."
            )
        else:
            lines.append("Your engagement remained steady throughout the session.")

        if high_count > 0:
            lines.append(f"You had {high_count} high-focus interval(s) — well done!")
        if low_count > 2:
            lines.append(
                f"There were {low_count} low-engagement intervals. "
                "Identifying distractions at those times could be beneficial."
            )

        return " ".join(lines)

    def get_session_summary(self, engagement_results: Dict, vision_results: Dict) -> Dict:
        """Generate overall session summary."""
        summary = engagement_results.get("summary", {})
        attendance = vision_results.get("attendance", {})
        total = len(attendance)
        present = sum(1 for r in attendance.values() if r["attended"])

        return {
            "total_participants": total,
            "present_count": present,
            "absent_count": total - present,
            "attendance_rate": round(present / max(total, 1), 3),
            "avg_engagement": summary.get("avg_engagement", 0.0),
            "dominant_engagement_level": summary.get("dominant_level", "unknown"),
            "high_engagement_ratio": summary.get("high_engagement_ratio", 0.0),
            "level_distribution": summary.get("level_distribution", {}),
            "total_suggestions": len(self.suggestions),
            "high_priority_suggestions": sum(
                1 for s in self.suggestions if s["priority"] == "high"
            ),
        }


if __name__ == "__main__":
    from vision_module import generate_simulated_vision_data
    from multimodal_agent import generate_simulated_engagement_data

    print("=== Analytics Engine — Test ===")
    vision_data = generate_simulated_vision_data(num_participants=6, session_duration=60)
    engagement_data = generate_simulated_engagement_data(session_duration=60)

    engine = AnalyticsEngine()
    suggestions = engine.generate_teaching_suggestions(engagement_data)
    student_analytics = engine.generate_student_analytics(vision_data, engagement_data)
    session_summary = engine.get_session_summary(engagement_data, vision_data)

    print(f"\nSession Summary: {session_summary}")
    print(f"\nTeaching Suggestions ({len(suggestions)}):")
    for s in suggestions[:3]:
        print(f"  [{s['priority'].upper()}] {s['suggestion']}")
    print(f"\nStudent Analytics ({len(student_analytics)}):")
    for a in student_analytics[:3]:
        print(f"  {a['student_id']}: attended={a['attended']}, avg_engagement={a['avg_engagement']}")
