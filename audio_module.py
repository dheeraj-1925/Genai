"""
Audio Module — Audio capture, noise analysis & speech activity detection
Part of: Multimodal Attendance & Engagement Agent
"""

import numpy as np
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    start_time: float
    end_time: float
    energy: float
    rms: float
    speech_detected: bool
    noise_level: str  # "low", "medium", "high"
    amplitude_variance: float


class AudioModule:
    """
    Audio Module: Analyzes audio signals for noise levels and speech activity.
    Works with simulated or real audio data.
    """

    SPEECH_ENERGY_THRESHOLD = 0.02
    HIGH_NOISE_THRESHOLD = 0.08
    MEDIUM_NOISE_THRESHOLD = 0.03

    def __init__(self, segment_duration: float = 5.0, sample_rate: int = 16000):
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate
        self.segments: List[AudioSegment] = []

    def classify_noise_level(self, energy: float) -> str:
        if energy >= self.HIGH_NOISE_THRESHOLD:
            return "high"
        elif energy >= self.MEDIUM_NOISE_THRESHOLD:
            return "medium"
        return "low"

    def analyze_signal(self, signal: np.ndarray, start_time: float) -> AudioSegment:
        rms = float(np.sqrt(np.mean(signal ** 2)))
        energy = float(np.mean(np.abs(signal)))
        amplitude_variance = float(np.var(signal))
        speech_detected = energy >= self.SPEECH_ENERGY_THRESHOLD
        noise_level = self.classify_noise_level(energy)
        return AudioSegment(
            start_time=start_time,
            end_time=start_time + self.segment_duration,
            energy=round(energy, 5),
            rms=round(rms, 5),
            speech_detected=speech_detected,
            noise_level=noise_level,
            amplitude_variance=round(amplitude_variance, 8),
        )

    def process_audio_array(self, audio: np.ndarray, session_start: float) -> List[AudioSegment]:
        samples_per_segment = int(self.sample_rate * self.segment_duration)
        self.segments = []
        for i in range(0, len(audio), samples_per_segment):
            chunk = audio[i: i + samples_per_segment]
            if len(chunk) < samples_per_segment // 2:
                break
            start = session_start + i / self.sample_rate
            seg = self.analyze_signal(chunk.astype(np.float32) / 32768.0, start)
            self.segments.append(seg)
        logger.info(f"Processed {len(self.segments)} audio segments")
        return self.segments

    def get_results(self) -> Dict:
        if not self.segments:
            return {"segments": [], "summary": {}}
        energies = [s.energy for s in self.segments]
        speech_count = sum(1 for s in self.segments if s.speech_detected)
        noise_dist = {"low": 0, "medium": 0, "high": 0}
        for s in self.segments:
            noise_dist[s.noise_level] += 1

        segments_list = []
        for s in self.segments:
            segments_list.append({
                "start_time": s.start_time,
                "end_time": s.end_time,
                "energy": s.energy,
                "rms": s.rms,
                "speech_detected": s.speech_detected,
                "noise_level": s.noise_level,
                "amplitude_variance": s.amplitude_variance,
            })

        return {
            "segments": segments_list,
            "summary": {
                "total_segments": len(self.segments),
                "speech_segments": speech_count,
                "speech_ratio": round(speech_count / len(self.segments), 3),
                "avg_energy": round(float(np.mean(energies)), 5),
                "max_energy": round(float(np.max(energies)), 5),
                "min_energy": round(float(np.min(energies)), 5),
                "noise_distribution": noise_dist,
                "dominant_noise_level": max(noise_dist, key=noise_dist.get),
            },
        }


def generate_simulated_audio_data(
    session_duration: float = 60.0,
    segment_duration: float = 5.0,
    speech_ratio: float = 0.65,
) -> Dict:
    """
    Generate simulated audio analysis data for testing.
    """
    import random
    random.seed(42)
    np.random.seed(42)

    session_start = time.time() - session_duration
    num_segments = int(session_duration / segment_duration)
    segments = []

    for i in range(num_segments):
        is_speech = random.random() < speech_ratio
        if is_speech:
            energy = random.uniform(0.02, 0.12)
        else:
            energy = random.uniform(0.001, 0.025)

        rms = energy * random.uniform(0.9, 1.1)
        noise_level = "high" if energy >= 0.08 else ("medium" if energy >= 0.03 else "low")
        segments.append({
            "start_time": session_start + i * segment_duration,
            "end_time": session_start + (i + 1) * segment_duration,
            "energy": round(energy, 5),
            "rms": round(rms, 5),
            "speech_detected": is_speech,
            "noise_level": noise_level,
            "amplitude_variance": round(random.uniform(0.0001, 0.005), 8),
        })

    speech_count = sum(1 for s in segments if s["speech_detected"])
    noise_dist = {"low": 0, "medium": 0, "high": 0}
    for s in segments:
        noise_dist[s["noise_level"]] += 1

    return {
        "segments": segments,
        "summary": {
            "total_segments": num_segments,
            "speech_segments": speech_count,
            "speech_ratio": round(speech_count / num_segments, 3),
            "avg_energy": round(float(np.mean([s["energy"] for s in segments])), 5),
            "max_energy": round(float(np.max([s["energy"] for s in segments])), 5),
            "min_energy": round(float(np.min([s["energy"] for s in segments])), 5),
            "noise_distribution": noise_dist,
            "dominant_noise_level": max(noise_dist, key=noise_dist.get),
        },
    }


if __name__ == "__main__":
    print("=== Audio Module — Simulated Test ===")
    results = generate_simulated_audio_data(session_duration=60, speech_ratio=0.65)
    s = results["summary"]
    print(f"Total segments: {s['total_segments']}")
    print(f"Speech ratio: {s['speech_ratio']:.1%}")
    print(f"Avg energy: {s['avg_energy']}")
    print(f"Noise distribution: {s['noise_distribution']}")
