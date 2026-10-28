"""
Multimodal Attendance & Engagement Agent — Streamlit Dashboard
Main entry point: `streamlit run app.py`
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(__file__))

from vision_module import generate_simulated_vision_data
from audio_module import generate_simulated_audio_data
from multimodal_agent import MultimodalAgent, generate_simulated_engagement_data
from analytics_engine import AnalyticsEngine

# ─────────────────────────── Page Config ────────────────────────────
st.set_page_config(
    page_title="Attendance & Engagement Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── CSS Styling ────────────────────────────
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #2d6a4f);
        border-radius: 12px;
        padding: 16px 20px;
        color: white;
        text-align: center;
        margin: 4px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .metric-card h2 { margin: 0; font-size: 2rem; }
    .metric-card p  { margin: 4px 0 0; font-size: 0.9rem; opacity: 0.85; }

    .suggestion-high   { border-left: 5px solid #e74c3c; padding: 10px 14px; background: #fff5f5; border-radius: 6px; margin: 6px 0; color: #1a1a1a !important; }
    .suggestion-medium { border-left: 5px solid #f39c12; padding: 10px 14px; background: #fffbf0; border-radius: 6px; margin: 6px 0; color: #1a1a1a !important; }
    .suggestion-info   { border-left: 5px solid #27ae60; padding: 10px 14px; background: #f0fff4; border-radius: 6px; margin: 6px 0; color: #1a1a1a !important; }
    .suggestion-high strong, .suggestion-medium strong, .suggestion-info strong { color: #111 !important; }

    .student-card { background: #f8f9fa; border-radius: 10px; padding: 14px; margin: 8px 0; border: 1px solid #dee2e6; color: #1a1a1a !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────── Session State ───────────────────────────
def init_session_state():
    defaults = {
        "vision_data": None,
        "audio_data": None,
        "engagement_data": None,
        "suggestions": None,
        "student_analytics": None,
        "session_summary": None,
        "session_run": False,
        "num_participants": 8,
        "session_duration": 60,
        "speech_ratio": 0.65,
        "attendance_threshold": 10,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()


# ─────────────────────────── Sidebar ─────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/classroom.png",
        width=80,
    )
    st.title("🎓 Session Config")
    st.divider()

    st.subheader("📋 Simulation Settings")
    num_participants = st.slider("Number of Participants", 3, 20, 8, key="num_participants")
    session_duration = st.slider("Session Duration (seconds)", 30, 300, 60, step=10, key="session_duration")
    speech_ratio = st.slider("Speech Activity Ratio", 0.1, 1.0, 0.65, step=0.05, key="speech_ratio")
    attendance_threshold = st.slider("Attendance Threshold (seconds)", 5, 30, 10, key="attendance_threshold")

    st.divider()

    run_btn = st.button("▶️ Run Session Analysis", use_container_width=True, type="primary")
    reset_btn = st.button("🔄 Reset", use_container_width=True)

    if reset_btn:
        for key in ["vision_data", "audio_data", "engagement_data", "suggestions",
                    "student_analytics", "session_summary", "session_run"]:
            st.session_state[key] = None if key != "session_run" else False
        st.rerun()

    st.divider()
    st.caption("ℹ️ This dashboard runs in **simulation mode**. All data is generated from configurable parameters without requiring a camera or microphone.")


# ─────────────────────────── Run Analysis ────────────────────────────
if run_btn:
    with st.spinner("🔄 Processing session data..."):
        progress = st.progress(0, text="Generating vision data...")
        time.sleep(0.3)

        vision_data = generate_simulated_vision_data(
            num_participants=num_participants,
            session_duration=float(session_duration),
        )
        progress.progress(25, text="Analyzing audio signals...")
        time.sleep(0.2)

        audio_data = generate_simulated_audio_data(
            session_duration=float(session_duration),
            speech_ratio=speech_ratio,
        )
        progress.progress(50, text="Fusing multimodal signals...")
        time.sleep(0.2)

        # Real multimodal fusion
        agent = MultimodalAgent(interval_seconds=max(5.0, session_duration / 10))
        agent.fuse_and_score(
            vision_results=vision_data,
            audio_results=audio_data,
            session_start=vision_data["session_start"],
            session_duration=float(session_duration),
        )
        engagement_data = agent.get_engagement_results()
        progress.progress(75, text="Running analytics engine...")
        time.sleep(0.2)

        engine = AnalyticsEngine()
        suggestions = engine.generate_teaching_suggestions(engagement_data)
        student_analytics = engine.generate_student_analytics(vision_data, engagement_data)
        session_summary = engine.get_session_summary(engagement_data, vision_data)
        progress.progress(100, text="Done!")
        time.sleep(0.3)
        progress.empty()

        st.session_state.vision_data = vision_data
        st.session_state.audio_data = audio_data
        st.session_state.engagement_data = engagement_data
        st.session_state.suggestions = suggestions
        st.session_state.student_analytics = student_analytics
        st.session_state.session_summary = session_summary
        st.session_state.session_run = True

    st.success("✅ Session analysis complete!")


# ─────────────────────────── Main Content ────────────────────────────
st.title("🎓 Multimodal Attendance & Engagement Agent")
st.caption("AI-powered classroom intelligence — privacy-preserving, offline-capable")

if not st.session_state.session_run:
    st.info("👈 Configure session parameters in the sidebar and click **▶️ Run Session Analysis** to begin.")

    st.markdown("### 🚀 What this system does")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **📷 Vision Module**
        - Face detection & tracking
        - Presence duration calculation
        - Attendance marking
        - Movement analysis
        """)
    with col2:
        st.markdown("""
        **🎙️ Audio Module**
        - Noise level analysis
        - Speech activity detection
        - Audio energy profiling
        - Segment-level insights
        """)
    with col3:
        st.markdown("""
        **🧠 Multimodal Agent**
        - Signal fusion
        - Engagement scoring
        - Teaching suggestions
        - Student analytics
        """)
    st.stop()


# ─────────────── Tabs ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Session Overview",
    "📷 Attendance",
    "📈 Engagement",
    "💡 Teaching Suggestions",
    "👥 Student Analytics",
])


# ══════════════ TAB 1: Session Overview ══════════════════════════════
with tab1:
    st.header("📊 Session Overview")
    summary = st.session_state.session_summary

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Participants", summary["total_participants"])
    with c2:
        st.metric("Present", summary["present_count"], delta=f"{summary['attendance_rate']:.0%} rate")
    with c3:
        st.metric("Avg Engagement", f"{summary['avg_engagement']:.1%}")
    with c4:
        st.metric("Teaching Suggestions", summary["total_suggestions"])

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Attendance Breakdown")
        att_df = pd.DataFrame({
            "Status": ["Present", "Absent"],
            "Count": [summary["present_count"], summary["absent_count"]],
        })
        st.bar_chart(att_df.set_index("Status"))

    with col_b:
        st.subheader("Engagement Level Distribution")
        level_dist = summary.get("level_distribution", {})
        if level_dist:
            level_df = pd.DataFrame({
                "Level": list(level_dist.keys()),
                "Count": list(level_dist.values()),
            })
            st.bar_chart(level_df.set_index("Level"))

    st.subheader("Audio Summary")
    audio_sum = st.session_state.audio_data.get("summary", {})
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Speech Ratio", f"{audio_sum.get('speech_ratio', 0):.1%}")
    a2.metric("Avg Energy", f"{audio_sum.get('avg_energy', 0):.4f}")
    a3.metric("Dominant Noise", audio_sum.get("dominant_noise_level", "—").capitalize())
    a4.metric("Audio Segments", audio_sum.get("total_segments", 0))


# ══════════════ TAB 2: Attendance ════════════════════════════════════
with tab2:
    st.header("📷 Attendance Records")

    attendance = st.session_state.vision_data.get("attendance", {})
    if attendance:
        rows = []
        for student_id, rec in attendance.items():
            rows.append({
                "Student ID": student_id,
                "Status": "✅ Present" if rec["attended"] else "❌ Absent",
                "Presence (s)": rec["presence_duration"],
                "Frames Present": rec["frames_present"],
                "Avg Movement (px)": rec["avg_movement"],
                "Track ID": rec["track_id"],
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        present_pct = sum(1 for r in attendance.values() if r["attended"]) / len(attendance)
        st.progress(present_pct, text=f"Attendance Rate: {present_pct:.0%}")

        st.subheader("Presence Duration per Participant")
        presence_df = pd.DataFrame({
            "Student": list(attendance.keys()),
            "Presence (s)": [r["presence_duration"] for r in attendance.values()],
        })
        st.bar_chart(presence_df.set_index("Student"))


# ══════════════ TAB 3: Engagement ════════════════════════════════════
with tab3:
    st.header("📈 Engagement Trends")

    intervals = st.session_state.engagement_data.get("intervals", [])
    if intervals:
        df = pd.DataFrame(intervals)
        df["interval_label"] = [f"T{i+1}" for i in range(len(df))]
        df.set_index("interval_label", inplace=True)

        st.subheader("Engagement Score Over Time")
        st.line_chart(df[["score", "visual_contribution", "audio_contribution"]])

        st.subheader("Face Count & Movement Over Time")
        st.bar_chart(df[["face_count", "movement_intensity"]])

        st.subheader("Interval Detail Table")
        display_df = df[["score", "level", "face_count", "speech_detected", "noise_level", "movement_intensity"]].copy()
        display_df.columns = ["Score", "Level", "Faces", "Speech", "Noise", "Movement"]
        st.dataframe(display_df, use_container_width=True)

        eng_sum = st.session_state.engagement_data.get("summary", {})
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Avg Score", f"{eng_sum.get('avg_engagement', 0):.1%}")
        e2.metric("Max Score", f"{eng_sum.get('max_engagement', 0):.1%}")
        e3.metric("Min Score", f"{eng_sum.get('min_engagement', 0):.1%}")
        e4.metric("High Engagement Ratio", f"{eng_sum.get('high_engagement_ratio', 0):.1%}")


# ══════════════ TAB 4: Teaching Suggestions ═══════════════════════════
with tab4:
    st.header("💡 AI-Driven Teaching Suggestions")
    st.caption("These suggestions are generated from engagement patterns and are intended as decision-support tools only.")

    suggestions = st.session_state.suggestions or []
    if not suggestions:
        st.info("No suggestions generated for this session. Engagement was consistently good!")
    else:
        priority_icons = {"high": "🔴", "medium": "🟡", "info": "🟢"}

        # Group by priority
        for priority in ["high", "medium", "info"]:
            group = [s for s in suggestions if s["priority"] == priority]
            if group:
                st.subheader(f"{priority_icons[priority]} {priority.capitalize()} Priority")
                for s in group:
                    div_class = f"suggestion-{priority}"
                    interval_label = f"Interval {s['interval_index'] + 1}"
                    st.markdown(
                        f'<div class="{div_class}"><strong>{interval_label}</strong> | '
                        f'Score: {s["engagement_score"]:.1%}<br>{s["suggestion"]}</div>',
                        unsafe_allow_html=True,
                    )

        st.divider()
        st.subheader("All Suggestions Table")
        sug_df = pd.DataFrame(suggestions)[["rule_id", "interval_index", "priority", "engagement_score", "suggestion"]]
        sug_df.columns = ["Rule", "Interval", "Priority", "Eng. Score", "Suggestion"]
        st.dataframe(sug_df, use_container_width=True, hide_index=True)


# ══════════════ TAB 5: Student Analytics ══════════════════════════════
with tab5:
    st.header("👥 Student Session Analytics")
    st.caption("Reflective, privacy-preserving analytics — no peer comparisons or grades.")

    student_analytics = st.session_state.student_analytics or []
    if not student_analytics:
        st.info("No student analytics available.")
    else:
        # Summary metrics
        attended = [s for s in student_analytics if s["attended"]]
        if attended:
            avg_eng = float(np.mean([s["avg_engagement"] for s in attended]))
            avg_pres = float(np.mean([s["presence_duration"] for s in attended]))
        else:
            avg_eng = 0.0
            avg_pres = 0.0

        m1, m2, m3 = st.columns(3)
        m1.metric("Students Present", len(attended))
        m2.metric("Avg Session Engagement", f"{avg_eng:.1%}")
        m3.metric("Avg Presence Duration", f"{avg_pres:.0f}s")

        st.divider()

        # Search / filter
        filter_status = st.selectbox("Filter by status", ["All", "Present", "Absent"])
        filtered = student_analytics
        if filter_status == "Present":
            filtered = [s for s in student_analytics if s["attended"]]
        elif filter_status == "Absent":
            filtered = [s for s in student_analytics if not s["attended"]]

        for student in filtered:
            with st.expander(
                f"{'✅' if student['attended'] else '❌'} {student['student_id']} — "
                f"Engagement: {student['avg_engagement']:.1%} | "
                f"Trend: {student['engagement_trend'].capitalize()}"
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Presence Duration", f"{student['presence_duration']:.0f}s")
                    st.metric("Avg Engagement", f"{student['avg_engagement']:.1%}")
                    st.metric("Engagement Trend", student["engagement_trend"].capitalize())
                    st.metric("Consistency Score", f"{student['engagement_consistency']:.1%}")
                with col2:
                    st.metric("High Engagement Intervals", student["high_engagement_intervals"])
                    st.metric("Low Engagement Intervals", student["low_engagement_intervals"])
                    st.metric("Peak Score", f"{student['peak_engagement_score']:.1%}")
                    st.metric("Avg Movement", f"{student['avg_movement']:.1f} px/frame")

                st.markdown("**📝 Reflective Summary**")
                st.info(student["reflective_summary"])

        st.divider()
        st.subheader("Comparative Overview Table")
        table_df = pd.DataFrame([
            {
                "Student": s["student_id"],
                "Attended": "✅" if s["attended"] else "❌",
                "Presence (s)": s["presence_duration"],
                "Avg Engagement": f"{s['avg_engagement']:.1%}",
                "Trend": s["engagement_trend"].capitalize(),
                "Consistency": f"{s['engagement_consistency']:.1%}",
                "High Intervals": s["high_engagement_intervals"],
                "Low Intervals": s["low_engagement_intervals"],
            }
            for s in student_analytics
        ])
        st.dataframe(table_df, use_container_width=True, hide_index=True)
