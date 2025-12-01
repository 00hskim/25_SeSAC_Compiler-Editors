# Audio_Features.py — 오디오 특징 추출

from __future__ import annotations
from typing import Dict, Optional

import numpy as np
import librosa

_EPS = 1e-12


def _safe_mean(x: np.ndarray) -> float:
    if x is None or len(x) == 0:
        return 0.0
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    return float(x.mean()) if x.size else 0.0


def _to_mono(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y.astype(np.float32, copy=False)
    # (channels, samples) 또는 (samples, channels) → 채널 평균
    return (
        y.mean(axis=0).astype(np.float32, copy=False)
        if y.shape[0] <= y.shape[1]
        else y.mean(axis=1).astype(np.float32, copy=False)
    )

def compute_basic_stats(
    y: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 512,
    vad_rel_thresh_db: float = -6.0,
) -> Dict[str, float]:

    y = _to_mono(y)
    if y.size == 0:
        return {"rms_db": -80.0, "zcr": 0.0, "vad_ratio": 0.0}

    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
    )[0]
    rms_db_frames = 20.0 * np.log10(np.maximum(rms, _EPS))
    rms_db = _safe_mean(rms_db_frames)

    zcr = librosa.feature.zero_crossing_rate(
        y,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]
    zcr_mean = _safe_mean(zcr)

    med_db = np.median(rms_db_frames) if rms_db_frames.size else -80.0
    active = rms_db_frames > (med_db + vad_rel_thresh_db)
    vad_ratio = float(active.mean()) if active.size else 0.0

    return {
        "rms_db": float(rms_db),
        "zcr": float(zcr_mean),
        "vad_ratio": float(vad_ratio),
    }


def compute_spectral(
    y: np.ndarray,
    sr: int,
    roll_percent: float = 0.85,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> Dict[str, float]:

    y = _to_mono(y)
    if y.size == 0:
        return {
            "centroid_hz": 0.0,
            "rolloff_hz": 0.0,
            "hf_ratio": 0.0,
            "flatness": 0.0,
            "slope_db_per_khz": 0.0,
            "contrast": 0.0,
        }

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    if S.size == 0:
        return {
            "centroid_hz": 0.0,
            "rolloff_hz": 0.0,
            "hf_ratio": 0.0,
            "flatness": 0.0,
            "slope_db_per_khz": 0.0,
            "contrast": 0.0,
        }

    freq = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    centroid = librosa.feature.spectral_centroid(S=S, freq=freq)[0]
    rolloff = librosa.feature.spectral_rolloff(
        S=S,
        sr=sr,
        roll_percent=roll_percent,
    )[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)[0]

    centroid_hz = _safe_mean(centroid)
    rolloff_hz = _safe_mean(rolloff)
    flatness_mean = _safe_mean(flatness)
    contrast_mean = _safe_mean(contrast)

    # HF 에너지 비율
    hf_cut = 4000.0
    hf_bin = np.searchsorted(freq, hf_cut)
    hf_energy = np.sum(S[hf_bin:, :])
    tot_energy = np.sum(S) + _EPS
    hf_ratio = float(hf_energy / tot_energy)

    # 스펙트럼 경사(선형회귀)
    mag_mean = np.maximum(S.mean(axis=1), _EPS)
    mag_db = 20.0 * np.log10(mag_mean)
    x = freq / 1000.0  # kHz
    mask = np.isfinite(mag_db) & np.isfinite(x)
    if mask.sum() > 2:
        slope = np.polyfit(x[mask], mag_db[mask], 1)[0]  # dB per kHz
    else:
        slope = 0.0

    return {
        "centroid_hz": float(centroid_hz),
        "rolloff_hz": float(rolloff_hz),
        "hf_ratio": float(hf_ratio),
        "flatness": float(flatness_mean),
        "slope_db_per_khz": float(slope),
        "contrast": float(contrast_mean),
    }


def estimate_tempo(y: np.ndarray, sr: int) -> Dict[str, float]:
    y = _to_mono(y)
    if y.size == 0:
        return {"bpm_est": 0.0}
    try:
        tempo = librosa.beat.tempo(y=y, sr=sr, aggregate=None)
        bpm = float(np.median(tempo)) if tempo is not None and tempo.size else 0.0
    except Exception:
        bpm = 0.0
    if bpm > 0:
        choices = np.array([60, 70, 80, 90, 100, 110, 120, 130])
        bpm = float(choices[np.argmin(np.abs(choices - bpm))])
    return {"bpm_est": bpm}


def bucketize_tone(
    spectral: Dict[str, float],
    sr: Optional[int] = None,
) -> Dict[str, str]:

    c = spectral.get("centroid_hz", 0.0)
    r = spectral.get("rolloff_hz", 0.0)
    hf = spectral.get("hf_ratio", 0.0)

    scale = 16000.0 if sr is None or sr <= 0 else float(sr)
    k1, k2, k3 = (
        1500.0 * (scale / 16000.0),
        2500.0 * (scale / 16000.0),
        3500.0 * (scale / 16000.0),
    )

    if c < k1:
        tone = "dark"
    elif c < k2:
        tone = "warm"
    elif c < k3:
        tone = "neutral"
    else:
        tone = "bright"

    if r < (3000.0 * (scale / 16000.0)) or hf < 0.12:
        tone = {"bright": "neutral", "neutral": "warm", "warm": "dark", "dark": "dark"}[
            tone
        ]
    if r > (5000.0 * (scale / 16000.0)) or hf > 0.25:
        tone = {"dark": "warm", "warm": "neutral", "neutral": "bright", "bright": "bright"}[
            tone
        ]

    return {"tone_tag": tone}


def bucketize_arrangement_density(
    basic: Dict[str, float],
    text_words_per_sec: Optional[float] = None,
    emotion_score: Optional[float] = None,
) -> Dict[str, str]:

    rms_db = basic.get("rms_db", -80.0)
    zcr = basic.get("zcr", 0.0)
    vad = basic.get("vad_ratio", 0.0)

    if rms_db <= -24:
        volume = "very gentle background"
    elif rms_db <= -20:
        volume = "gentle background"
    elif rms_db <= -16:
        volume = "moderate background"
    else:
        volume = "noticeable but controlled"

    if vad >= 0.7 or (text_words_per_sec is not None and text_words_per_sec >= 3.5):
        arrangement = "thin arrangement"
    elif vad >= 0.4:
        arrangement = "light arrangement"
    else:
        arrangement = "fuller arrangement"

    if text_words_per_sec is not None:
        if text_words_per_sec >= 3.5:
            rhythm = "minimal rhythm, low syncopation"
        elif text_words_per_sec >= 2.0:
            rhythm = "simple rhythm"
        else:
            rhythm = "moderate rhythm"
    else:
        if vad >= 0.6 or zcr <= 0.05:
            rhythm = "minimal rhythm, low syncopation"
        elif vad >= 0.3:
            rhythm = "simple rhythm"
        else:
            rhythm = "moderate rhythm"

    if emotion_score is not None:
        if emotion_score < 0.4:
            arrangement = "thin arrangement"
            volume = "very gentle background"
        elif emotion_score > 0.75 and arrangement != "fuller arrangement":
            arrangement = "light arrangement"

    return {
        "arrangement_tag": arrangement,
        "rhythm_tag": rhythm,
        "volume_hint": volume,
    }


def extract_audio_features(
    y: np.ndarray,
    sr: int,
    text_words_per_sec: Optional[float] = None,
    emotion_score: Optional[float] = None,
) -> Dict[str, float | str]:

    basic = compute_basic_stats(y, sr)
    spectral = compute_spectral(y, sr)
    tempo = estimate_tempo(y, sr)
    tone = bucketize_tone(spectral, sr=sr)
    arr = bucketize_arrangement_density(
        basic,
        text_words_per_sec=text_words_per_sec,
        emotion_score=emotion_score,
    )

    return {
        **basic,
        **spectral,
        **tempo,
        **tone,
        **arr,
    }
