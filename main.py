# main.py

import os
# from __future__ import annotations
from typing import Dict, Any, List
import sys
import subprocess
from pathlib import Path

import torch

from config import init_env
from shot_analyzer import analyze_shots, save_shot_json
from whisper_stt import load_whisper_asr, run_whisper
from emotion_ser import load_emotion_pipeline, attach_emotions_to_segments
from audio_librosa_runner import attach_audio_features_to_segments
from hardcut import main as hardcut_main
from subtitle import main as subtitle_main
from bgm_create import main as bgm_main
from final_mapping import main as final_mapping_main

HF_TOKEN = os.getenv("HF_TOKEN")
STABLE_AUDIO_API_KEY = os.getenv("STABLE_AUDIO_API_KEY")

INPUT_VIDEO  = Path("./Input/Sample.mp4")
OUTPUT_VIDEO = Path("./Output/Sample_Result.mp4")
WORK_DIR     = Path("./Work")
WORK_WAV     = WORK_DIR / "Sample_16k.wav"
ANALYSIS_JSON = WORK_DIR / "Sample_analysis.json"

SHOT_THRESHOLD     = 0.2
MIN_SEG_SEC_STT    = 0.5
MIN_SEG_SEC_AUDIO  = 0.3
TARGET_SR          = 16000


def ffmpeg_to_wav_16k_mono(src_path: Path, dst_path: Path) -> bool:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_path),
        "-vn",
        "-ac", "1",
        "-ar", str(TARGET_SR),
        str(dst_path),
    ]
    try:
        print(f"[ffmpeg] {src_path} -> {dst_path} (16k mono)")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception as e:
        print(f"[ffmpeg] 변환 실패: {e}")
        return False


def _merge_segments_into_shots(
    shot_meta: Dict[str, Any],
    segments: List[Dict[str, Any]],
    full_text: str,
    language: str,
) -> Dict[str, Any]:
    shots = shot_meta.get("shots", {})
    if not shots:
        print("[merge] shots 비어 있음 → segments만 별도로 유지")
    else:
        print(f"[merge] shots 개수: {len(shots)} / segments: {len(segments)}")

    for seg in segments:
        try:
            s = float(seg.get("start", 0.0))
            e = float(seg.get("end", 0.0))
        except Exception:
            continue
        if e <= s:
            continue
        mid = 0.5 * (s + e)

        attached = False
        for shot_key, shot_list in shots.items():
            if not shot_list:
                continue
            shot = shot_list[0]
            st = float(shot.get("start", 0.0))
            ed = float(shot.get("end", 0.0))
            if st <= mid <= ed:
                seg_entry = {
                    "start": f"{s:.2f}",
                    "end": f"{e:.2f}",
                    "text": seg.get("text", ""),
                    "language": seg.get("language", language),
                    "emotion": seg.get("emotion", None),
                    "emotion_score": seg.get("emotion_score", None),
                    "features": seg.get("features", {}),
                }
                shot.setdefault("segments", []).append(seg_entry)
                attached = True
                break

        if not attached:
            # 어떤 샷에도 안 붙은 세그먼트는 일단 버림(경계에 걸린 구간 등)
            pass

    shot_meta["whisper"] = {
        "full_text": full_text,
        "language": language,
        "num_segments": len(segments),
    }
    return shot_meta


def main() -> None:
    print("\n=== Edit-Click: 분석 파이프라인 시작 ===")

    # 1) 환경 점검
    init_env()

    # 2) 입력 파일 확인
    if not INPUT_VIDEO.exists():
        print(f"[main] 입력 파일 없음: {INPUT_VIDEO.resolve()}")
        sys.exit(1)

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    # 3) mp4 -> wav(16k mono)
    ok = ffmpeg_to_wav_16k_mono(INPUT_VIDEO, WORK_WAV)
    if not ok or not WORK_WAV.exists():
        print("[main] 오디오 변환 실패, 종료")
        sys.exit(1)

    # 4) 샷 분석 (TransNetV2)
    shot_meta = analyze_shots(str(INPUT_VIDEO), threshold=SHOT_THRESHOLD)

    # 5) Whisper 로딩 + STT
    print("\n[main] Whisper ASR 로딩 중...")
    device_index = 0 if torch.cuda.is_available() else -1
    asr_pipeline = load_whisper_asr(device_index=device_index)
    stt_result = run_whisper(str(WORK_WAV), asr_pipeline, min_seg_sec=MIN_SEG_SEC_STT)

    segments = stt_result.get("segments", [])
    if not segments:
        print("[main] STT 세그먼트 없음, 종료")
        sys.exit(1)

    # 6) 감정 분석(Wav2Vec2 SER)
    print("\n[main] 감정 분석 파이프라인 로딩 중...")
    emo_proc, emo_model, emo_device = load_emotion_pipeline()
    segments = attach_emotions_to_segments(
        wav_path=str(WORK_WAV),
        segments=segments,
        processor=emo_proc,
        model=emo_model,
        device=emo_device,
        target_sr=TARGET_SR,
        min_seg_sec=MIN_SEG_SEC_AUDIO,
    )

    # 7) librosa 기반 오디오 피처 추출
    print("\n[main] 오디오 피처(librosa) 추출 중...")
    segments = attach_audio_features_to_segments(
        wav_path=str(WORK_WAV),
        segments=segments,
        target_sr=TARGET_SR,
        min_seg_sec=MIN_SEG_SEC_AUDIO,
    )

    # 8) 샷 구조에 세그먼트 결합 + 최종 JSON 작성
    print("\n[main] 샷 구조에 세그먼트 머지 중...")
    merged = _merge_segments_into_shots(
        shot_meta=shot_meta,
        segments=segments,
        full_text=stt_result.get("text", ""),
        language=stt_result.get("language", ""),
    )

    print("[main] 최종 분석 JSON 저장 중...")
    save_shot_json(merged, str(ANALYSIS_JSON))

    print("\n✅ 완료: 분석 JSON 생성")
    print(f"   - 입력 영상 : {INPUT_VIDEO.resolve()}")
    print(f"   - 분석 JSON : {ANALYSIS_JSON.resolve()}")
    print("   - Output 폴더에는 추후 최종 편집 영상만 저장 예정")

    # 9) hardcut 자동 컷 편집 실행 (ffmpeg + JSON 리맵)
    print("\n[main] Hardcut 자동 컷 편집 실행 중...")
    hardcut_main()

    # 10) subtitle(drawtext) 적용
    print("\n[main] Subtitle(drawtext) 적용 중...")
    subtitle_main()

    # 11) BGM 생성 (Stable Audio 호출)
    print("\n[main] BGM 생성 파이프라인 실행 중...")
    bgm_main()

    # 12) 최종 영상 + BGM 믹스
    print("\n[main] 최종 BGM 믹스 파이프라인 실행 중...")
    final_mapping_main()


if __name__ == "__main__":
    main()
