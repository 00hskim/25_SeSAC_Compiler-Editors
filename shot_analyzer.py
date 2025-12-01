# shot_analyzer.py
from __future__ import annotations
from typing import Dict, Any
import os
import json

import torch
from transnetv2_pytorch import TransNetV2


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def analyze_shots(
    video_path: str,
    threshold: float = 0.2,
) -> Dict[str, Any]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"[shot_analyzer] video not found: {video_path}")

    print(f"[shot_analyzer] Start shot analysis")
    print(f"[shot_analyzer]  - video_path : {video_path}")
    print(f"[shot_analyzer]  - threshold  : {threshold}")

    device = _get_device()
    print(f"[shot_analyzer]  - device     : {device}")

    # 1) 모델 로드
    print("[shot_analyzer] Loading TransNetV2 model...")
    model = TransNetV2(device=device)
    model.eval()
    print("[shot_analyzer] Model loaded")

    # 2) 전체 메타 정보(fps, duration)
    print("[shot_analyzer] Running analyze_video(...)")
    results = model.analyze_video(video_path)
    fps = float(results.get("fps", 0.0))
    scenes_full = results.get("scenes", [])
    if scenes_full:
        last_end_time = float(scenes_full[-1].get("end_time", 0.0))
    else:
        last_end_time = 0.0
    print(f"[shot_analyzer]  - fps      : {fps}")
    print(f"[shot_analyzer]  - duration : {last_end_time:.3f} sec")

    # 3) threshold 기준 샷 분할
    print("[shot_analyzer] Running detect_scenes(...)")
    scenes = model.detect_scenes(video_path, threshold=threshold)
    print(f"[shot_analyzer]  - num_scenes(th={threshold}) : {len(scenes)}")

    # 4) JSON 구조
    shots_dict: Dict[str, Any] = {}
    for idx, s in enumerate(scenes, start=1):
        key = f"shot_{idx}"

        start_time = float(s.get("start_time", 0.0))
        end_time = float(s.get("end_time", 0.0))
        start_frame = int(s.get("start_frame", 0))
        end_frame = int(s.get("end_frame", 0))
        prob = float(s.get("probability", 0.0))

        print(
            f"[shot_analyzer]    - {key}: "
            f"{start_time:.3f} ~ {end_time:.3f} sec "
            f"(frames {start_frame}-{end_frame}, prob={prob:.3f})"
        )

        shots_dict[key] = [
            {
                "start": start_time,
                "end": end_time,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "probability": prob,
                "segments": [],
            }
        ]

    meta = {
        "video": {
            "path": os.path.abspath(video_path),
            "duration": last_end_time,
            "fps": fps,
        },
        "shots": shots_dict,
        "transnet": {
            "threshold": threshold,
            "model_device": device,
        },
    }

    print("[shot_analyzer] Shot analysis done")
    return meta


def save_shot_json(data: Dict[str, Any], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[shot_analyzer] Saved shot analysis JSON → {out_path}")
