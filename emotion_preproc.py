# emotion_preproc.py
from __future__ import annotations
from typing import Any, Dict, Union

import numpy as np
import torch
from transformers import AutoFeatureExtractor


def build_emotion_processor(model_name: str) -> Any:
    print(f"[emotion_preproc] Loading feature extractor: {model_name}")
    processor = AutoFeatureExtractor.from_pretrained(model_name)
    print("[emotion_preproc] Feature extractor loaded")
    return processor


def _to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x.astype(np.float32, copy=False)
    if x.ndim == 2:
        # 보통 (T, C) 또는 (C, T) → 길이 큰 축을 time으로 보고 평균
        if x.shape[0] <= x.shape[1]:
            return x.mean(axis=0).astype(np.float32, copy=False)
        return x.mean(axis=1).astype(np.float32, copy=False)
    return x.reshape(-1).astype(np.float32, copy=False)


def make_emotion_inputs(
    processor: Any,
    audio_array: Union[np.ndarray, list, torch.Tensor],
    sampling_rate: int,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, torch.Tensor]:
    if isinstance(audio_array, torch.Tensor):
        arr = audio_array.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        arr = np.asarray(audio_array, dtype=np.float32)

    arr = _to_mono(arr)

    inputs = processor(
        arr,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    )

    tensor_inputs: Dict[str, torch.Tensor] = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            tensor_inputs[k] = v.to(device)
        else:
            tensor_inputs[k] = torch.as_tensor(v, device=device)

    return tensor_inputs
