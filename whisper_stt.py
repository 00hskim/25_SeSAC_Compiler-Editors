# whisper_stt.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path
import re

from transformers import pipeline

SKIP_HEAD_SEC = 2.0


def load_whisper_asr(
    device_index: Optional[int] = None,
    model_name: str = "openai/whisper-large-v3",
):
    print("[whisper_stt] Loading Whisper ASR pipeline...")
    if device_index is None:
        asr = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            return_timestamps=True,
        )
    else:
        asr = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device_index,
            return_timestamps=True,
        )
    print("[whisper_stt] Whisper pipeline loaded")
    return asr


def _split_korean_ends(
    segments: List[Dict[str, Any]],
    min_seg_sec: float,
) -> List[Dict[str, Any]]:
    new_segments: List[Dict[str, Any]] = []

    pattern = re.compile(r'(.*?(?:입니다|합니다)[\.!\?]?)', re.DOTALL)

    for seg in segments:
        text = (seg.get("text") or "").strip()
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", 0.0))
        dur = max(e - s, 0.0)

        if dur <= 0 or len(text) == 0:
            continue

        parts: List[str] = []
        last_end = 0
        for m in pattern.finditer(text):
            part = m.group(1)
            if part:
                parts.append(part.strip())
            last_end = m.end()

        if last_end < len(text):
            tail = text[last_end:].strip()
            if tail:
                parts.append(tail)

        if len(parts) <= 1:
            new_segments.append(seg)
            continue

        total_chars = sum(len(p) for p in parts) or 1
        cur_t = s

        temp_children: List[Dict[str, Any]] = []
        for p in parts:
            ratio = len(p) / total_chars
            part_dur = dur * ratio
            part_end = cur_t + part_dur

            child = {
                "start": float(cur_t),
                "end": float(part_end),
                "text": p,
                "language": seg.get("language", ""),
            }
            temp_children.append(child)
            cur_t = part_end

        if all((c["end"] - c["start"]) < min_seg_sec * 0.4 for c in temp_children):
            new_segments.append(seg)
        else:
            new_segments.extend(temp_children)

    return new_segments


def run_whisper(
    wav_path: str,
    asr_pipeline,
    min_seg_sec: float = 0.5,
) -> Dict[str, Any]:
    audio_path = Path(wav_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"[whisper_stt] audio not found: {audio_path}")

    print("[whisper_stt] Start transcription")
    print(f"[whisper_stt]  - wav_path    : {audio_path}")
    print(f"[whisper_stt]  - min_seg_sec : {min_seg_sec}")

    result = asr_pipeline(str(audio_path), return_timestamps=True)
    text_full = result.get("text", "").strip()
    language = result.get("language", "")

    chunks: List[Dict[str, Any]] = []
    if isinstance(result, dict):
        if "chunks" in result and isinstance(result["chunks"], list):
            chunks = result["chunks"]
        elif "segments" in result and isinstance(result["segments"], list):
            chunks = result["segments"]

    segments: List[Dict[str, Any]] = []
    for idx, ch in enumerate(chunks):
        ts = ch.get("timestamp") or ch.get("timestamps")
        if ts is None:
            continue

        if isinstance(ts, (list, tuple)) and len(ts) >= 2:
            start, end = ts[0], ts[1]
        else:
            continue

        try:
            s = float(start)
            e = float(end)
        except Exception:
            continue

        if e <= s:
            continue

        if e <= SKIP_HEAD_SEC:
            continue
        if s < SKIP_HEAD_SEC:
            s = SKIP_HEAD_SEC

        if (e - s) < min_seg_sec:
            continue

        text = (ch.get("text") or "").strip()
        if not text:
            continue

        seg = {
            "id": idx,
            "start": float(s),
            "end": float(e),
            "text": text,
            "language": language,
        }
        segments.append(seg)

    segments = _split_korean_ends(segments, min_seg_sec)

    for new_id, seg in enumerate(segments):
        seg["id"] = new_id

    print(f"[whisper_stt]  - segments: {len(segments)}")
    return {
        "text": text_full,
        "segments": segments,
        "language": language,
    }
