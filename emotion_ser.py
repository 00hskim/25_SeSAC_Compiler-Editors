# emotion_ser.py

from __future__ import annotations
from typing import Dict, Any, List
import os

import numpy as np
import librosa
import torch
from transformers import AutoModelForAudioClassification

from emotion_preproc import build_emotion_processor, make_emotion_inputs

MODEL_NAME = "superb/wav2vec2-base-superb-er"


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_emotion_pipeline():
    device = _get_device()
    print(f"[emotion_ser] Using device: {device}")
    print(f"[emotion_ser] Loading emotion processor/model: {MODEL_NAME}")

    processor = build_emotion_processor(MODEL_NAME)

    # 🔧 핵심 수정: safetensors 자동 변환 끄기
    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        use_safetensors=False,
    )
    model.to(device)
    model.eval()

    print("[emotion_ser] Emotion model loaded")
    return processor, model, device


def _load_wav_mono_16k(wav_path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"[emotion_ser] wav not found: {wav_path}")
    y, sr = librosa.load(wav_path, sr=target_sr, mono=True)
    return y.astype(np.float32, copy=False), sr


def attach_emotions_to_segments(
    wav_path: str,
    segments: List[Dict[str, Any]],
    processor,
    model,
    device: str,
    target_sr: int = 16000,
    min_seg_sec: float = 0.3,
) -> List[Dict[str, Any]]:
    print(f"[emotion_ser] Attaching emotions to {len(segments)} segments")
    y, sr = _load_wav_mono_16k(wav_path, target_sr=target_sr)

    num_classes = getattr(model.config, "num_labels", None)
    id2label = getattr(model.config, "id2label", None)
    if id2label is None:
        id2label = {i: f"class_{i}" for i in range(num_classes or 0)}

    for seg in segments:
        try:
            s = float(seg.get("start", 0.0))
            e = float(seg.get("end", 0.0))
        except Exception:
            seg["emotion"] = None
            seg["emotion_score"] = None
            continue

        if e <= s:
            seg["emotion"] = None
            seg["emotion_score"] = None
            continue

        start_idx = int(s * sr)
        end_idx = int(e * sr)
        if end_idx <= start_idx:
            seg["emotion"] = None
            seg["emotion_score"] = None
            continue

        audio_seg = y[start_idx:end_idx]
        dur = (end_idx - start_idx) / float(sr)
        if dur < min_seg_sec:
            seg["emotion"] = None
            seg["emotion_score"] = None
            continue

        inputs = make_emotion_inputs(
            processor=processor,
            audio_array=audio_seg,
            sampling_rate=sr,
            device=device,
        )

        with torch.inference_mode():
            logits = model(**inputs).logits  # [1, C]
            probs = torch.softmax(logits, dim=-1)[0]  # [C]

        score, idx = torch.max(probs, dim=-1)
        label = id2label.get(idx.item(), str(idx.item()))
        label = str(label)

        # 너무 애매하면 neutral로 정리
        if label.lower() == "neutral":
            emo_label = "neutral"
        else:
            emo_label = label

        emo_score = float(score.item())

        seg["emotion"] = emo_label
        seg["emotion_score"] = emo_score

    print("[emotion_ser] Emotion attachment done")
    return segments
