<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=30&duration=3000&pause=1000&color=00D4FF&center=true&vCenter=true&width=700&lines=Multimodal+Attendance+%26+Engagement+Agent;AI-Powered+Classroom+Intelligence;Privacy-Preserving+%7C+Offline+%7C+Real-Time" alt="Typing SVG" />

<br/>

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-00897B?style=for-the-badge&logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-00D4FF?style=for-the-badge)

<br/>

> **An AI-powered, privacy-preserving classroom intelligence system** that automatically tracks attendance, analyzes student engagement, and generates real-time teaching suggestions — running entirely **offline** with no cloud dependency.

<br/>

[🚀 Quick Start](#-quick-start) · [📐 Architecture](#-architecture) · [✨ Features](#-features) · [📊 Dashboard](#-dashboard-preview) · [🛠 Tech Stack](#-tech-stack) · [📁 Structure](#-project-structure)

---

</div>

## 🎯 What Is This?

Traditional attendance systems are **manual, error-prone, and context-blind**. This project goes far beyond — combining **Computer Vision**, **Audio Signal Processing**, and **Agent-based AI Reasoning** to deliver:

- 🎓 **Automated, presence-based attendance** (not just single-frame detection)
- 📊 **Real-time engagement scoring** with multimodal signal fusion
- 💡 **AI-driven teaching suggestions** grounded in observable data
- 👥 **Per-student reflective analytics** that respect privacy
- 🔒 **100% offline** — no data ever leaves your machine

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 📷 Vision Intelligence
- Face detection using OpenCV Haar Cascades
- Centroid-based multi-face tracking with unique IDs
- Presence duration calculation via timestamps
- Movement intensity analysis per participant
- Attendance confirmed only after sustained presence threshold

</td>
<td width="50%">

### 🎙️ Audio Intelligence
- Audio signal segmentation & energy profiling
- Speech activity detection via RMS thresholding
- Noise level classification: `low` / `medium` / `high`
- Amplitude variance & frequency analysis
- Session-level audio context generation

</td>
</tr>
<tr>
<td width="50%">

### 🧠 Multimodal Fusion
- Timestamp-synchronized vision + audio fusion
- Weighted engagement scoring (60% visual / 40% audio)
- Temporal engagement profiles per configurable interval
- Attention drop & recovery trend detection
- Explainable, rule-based reasoning engine

</td>
<td width="50%">

### 📊 Analytics & Dashboard
- 5-tab interactive Streamlit dashboard
- 7 context-aware teaching suggestion rules
- Per-student non-evaluative reflective summaries
- Real-time charts: engagement timeline, attendance, noise
- Fully configurable via sidebar sliders

</td>
</tr>
</table>

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    VIDEO / AUDIO INPUT                          │
│              (Webcam / Video File / Microphone)                 │
└──────────────────────┬──────────────────┬───────────────────────┘
                       │                  │
              ┌────────▼──────┐  ┌────────▼──────┐
              │ Vision Module │  │  Audio Module │
              │               │  │               │
              │ • Face Detect │  │ • Energy Anal │
              │ • Tracking    │  │ • Speech Det  │
              │ • Attendance  │  │ • Noise Class │
              └────────┬──────┘  └────────┬──────┘
                       │                  │
              ┌────────▼──────────────────▼──────┐
              │         Multimodal Agent          │
              │                                  │
              │  • Signal Fusion (60/40 weight)  │
              │  • Engagement Scoring (0–1.0)    │
              │  • Temporal Profile Generation   │
              └─────────────────┬────────────────┘
                                │
              ┌─────────────────▼────────────────┐
              │         Analytics Engine          │
              │                                  │
              │  • 7 Rule-Based Suggestion Rules │
              │  • Per-Student Analytics         │
              │  • Session Summary               │
              └─────────────────┬────────────────┘
                                │
              ┌─────────────────▼────────────────┐
              │       Streamlit Dashboard         │
              │                                  │
              │  📊 Overview  📷 Attendance       │
              │  📈 Engagement  💡 Suggestions    │
              │  👥 Student Analytics             │
              └──────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Webcam (optional — simulation mode works without one)

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/dheeraj-1925/Genai.git
cd Genai/attendance_agent
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Launch the Dashboard

```bash
python -m streamlit run app.py
```

### 4️⃣ Open in Browser

```
http://localhost:8501
```

> 💡 **No camera? No problem.** The dashboard runs in **simulation mode** by default — configure parameters in the sidebar and click **▶️ Run Session Analysis**.

---

## 📊 Dashboard Preview

| Tab | Description |
|-----|-------------|
| 📊 **Session Overview** | KPI metrics, attendance rate, engagement distribution, audio summary |
| 📷 **Attendance** | Per-student status table, presence duration bar chart |
| 📈 **Engagement Trends** | Score timeline, visual vs audio contributions, face count chart |
| 💡 **Teaching Suggestions** | Priority-grouped AI suggestions (High / Medium / Info) |
| 👥 **Student Analytics** | Expandable cards with reflective summaries & engagement trend |

### ⚙️ Sidebar Configuration

| Parameter | Default | Range |
|-----------|---------|-------|
| Number of Participants | 8 | 3 – 20 |
| Session Duration | 60s | 30 – 300s |
| Speech Activity Ratio | 0.65 | 0.10 – 1.0 |
| Attendance Threshold | 10s | 5 – 30s |

---

## 🧠 Teaching Suggestion Rules

The Analytics Engine applies **7 rule-based triggers** to generate context-aware suggestions:

| Rule | Trigger Condition | Priority |
|------|-------------------|----------|
| `sustained_low_engagement` | 3+ consecutive low-score intervals | 🔴 High |
| `low_presence` | Face count drops below 3 | 🔴 High |
| `attention_drop` | Score drops sharply from ≥0.6 to <0.4 | 🟡 Medium |
| `high_noise` | Dominant noise level = `high` | 🟡 Medium |
| `low_speech` | No speech detected for 3+ intervals | 🟡 Medium |
| `peak_engagement` | Score reaches ≥ 0.8 | 🟢 Info |
| `recovery_trend` | 3 consecutive rising scores from low base | 🟢 Info |

---

## 📁 Project Structure

```
attendance_agent/
│
├── 📄 app.py                  →  Streamlit Dashboard (5-tab UI, main entry)
├── 📄 vision_module.py        →  Face detection, centroid tracking, attendance
├── 📄 audio_module.py         →  Audio analysis, speech detection, noise classification
├── 📄 multimodal_agent.py     →  Signal fusion, engagement scoring, temporal profiles
├── 📄 analytics_engine.py     →  Teaching suggestions, student analytics, summaries
├── 📄 requirements.txt        →  All Python dependencies
└── 📄 README.md               →  You are here
```

---

## 🛠 Tech Stack

<div align="center">

| Technology | Version | Purpose |
|------------|---------|---------|
| ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white) | 3.9+ | Core language |
| ![OpenCV](https://img.shields.io/badge/-OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white) | 4.9+ | Face detection & video processing |
| ![MediaPipe](https://img.shields.io/badge/-MediaPipe-00897B?style=flat&logo=google&logoColor=white) | 0.10+ | Face landmark analysis |
| ![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) | 1.32+ | Interactive dashboard UI |
| ![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat&logo=numpy&logoColor=white) | 1.24+ | Signal processing & math |
| ![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat&logo=pandas&logoColor=white) | 2.0+ | Data handling & analytics |
| ![Librosa](https://img.shields.io/badge/-Librosa-FF6B35?style=flat&logoColor=white) | 0.10+ | Audio feature extraction |

</div>

---

## 🔬 Run Individual Modules

Each module is independently testable with built-in simulation data:

```bash
# Test Vision Module (face tracking & attendance simulation)
python vision_module.py

# Test Audio Module (noise & speech analysis simulation)
python audio_module.py

# Test Multimodal Agent (engagement scoring simulation)
python multimodal_agent.py

# Test Analytics Engine (suggestions & student analytics)
python analytics_engine.py
```

---

## 🔒 Privacy & Ethics

This system is designed with **ethical AI principles** at its core:

- ✅ **No cloud uploads** — all processing happens locally
- ✅ **No biometric storage** — face data is not saved or stored
- ✅ **No peer comparisons** — student analytics are individual & reflective
- ✅ **No automated decisions** — all suggestions are advisory only
- ✅ **Transparent reasoning** — every suggestion is traceable to a rule

---

## 📌 Key Design Decisions

- **Presence-based attendance** (not single-frame) prevents false positives
- **Weighted multimodal fusion** (60% vision / 40% audio) balances reliability
- **Rule-based agent** ensures explainability over black-box ML
- **Simulation mode** makes the system testable without hardware
- **Modular architecture** allows each component to evolve independently

---

## 🗺️ Future Enhancements

- [ ] Real-time webcam & microphone integration
- [ ] PDF/CSV session report export
- [ ] Multi-session trend tracking
- [ ] Emotion recognition via MediaPipe face mesh
- [ ] LLM-powered suggestion generation
- [ ] Multi-classroom support

---

<div align="center">

---

**Built with ❤️ using AI, Computer Vision & Audio Signal Processing**

*Multimodal Attendance & Engagement Agent — Empowering Educators, Respecting Privacy*

</div>
