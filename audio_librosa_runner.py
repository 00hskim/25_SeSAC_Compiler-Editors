# audio_librosa_runner.py
from __future__ import annotations
from typing import List, Dict, Any, Union
from pathlib import Path

import numpy as np
import librosa

from Audio_Features import extract_audio_features


def _words_per_sec(text: str, duration_s: float) -> float:
    if duration_s <= 0:
        return 0.0
    text = (text or "").strip()
    if not text:
        return 0.0
    wc = len(text.split())
    return float(wc) / float(duration_s)


def attach_audio_features_to_segments(
    wav_path: Union[str, Path],
    segments: List[Dict[str, Any]],
    target_sr: int = 16000,
    min_seg_sec: float = 0.3,
) -> List[Dict[str, Any]]:
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"[audio_librosa_runner] audio not found: {wav_path}")

    print("[audio_librosa_runner] Start audio feature extraction")
    print(f"[audio_librosa_runner]  - wav_path    : {wav_path}")
    print(f"[audio_librosa_runner]  - target_sr   : {target_sr}")
    print(f"[audio_librosa_runner]  - min_seg_sec : {min_seg_sec}")
    print(f"[audio_librosa_runner]  - num_segments: {len(segments)}")

    y, sr = librosa.load(str(wav_path), sr=target_sr)
    if y.size == 0:
        print("[audio_librosa_runner]  - WARNING: empty audio")
        return segments

    n_total = len(segments)
    n_used = 0
    min_samples = int(min_seg_sec * sr)

    for seg in segments:
        try:
            s = float(seg.get("start", 0.0))
            e = float(seg.get("end", 0.0))
        except Exception:
            continue

        if e <= s:
            continue

        start_idx = int(max(s * sr, 0))
        end_idx = int(min(e * sr, len(y)))
        if end_idx <= start_idx:
            continue

        seg_audio = y[start_idx:end_idx]
        if seg_audio.size < min_samples:
            continue

        duration = max(e - s, 1e-6)
        text = seg.get("text", "")
        wps = _words_per_sec(text, duration)

        emo_score = seg.get("emotion_score", None)
        if emo_score is not None:
            try:
                emo_score = float(emo_score)
            except Exception:
                emo_score = None

        feats = extract_audio_features(
            seg_audio,
            sr,
            text_words_per_sec=wps,
            emotion_score=emo_score,
        )

        seg["features"] = feats
        n_used += 1

    print(f"[audio_librosa_runner]  - segments(updated): {n_used}/{n_total}")
    print("[audio_librosa_runner] Audio feature extraction done")
    return segments
