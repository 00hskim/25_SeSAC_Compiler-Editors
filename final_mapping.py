# final_mapping.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import json
import subprocess
from pathlib import Path

# ===== 경로 설정 =====
HARDCUT_JSON      = Path("./Work/Sample_analysis_hardcut.json")
SUBTITLED_VIDEO   = Path("./Work/hardcut_subtitled.mp4")
BGM_DIR           = Path("./Work/BGM")
BGM_MIX_DIR       = Path("./Work/BGM_mix")
FULL_BGM_AUDIO    = BGM_MIX_DIR / "bgm_full.wav"
FINAL_OUTPUT      = Path("./Output/Sample_Result.mp4")

MIN_BGM_SEC       = 15.0
BGM_VOLUME_RATIO  = 0.35


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"[final] JSON 없음: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_shots(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    shots_dict = data.get("shots", {})
    shots: List[Dict[str, Any]] = []

    for _, lst in shots_dict.items():
        if not lst:
            continue
        shots.append(lst[0])

    shots.sort(key=lambda s: float(s.get("start", 0.0)))
    return shots


def compute_bgm_groups(data: Dict[str, Any], min_bgm_sec: float = MIN_BGM_SEC) -> List[Tuple[float, float]]:
    video_meta = data.get("video", {})
    duration = float(video_meta.get("duration", 0.0))
    if duration <= 0:
        raise ValueError("[final] video.duration 이 0초 이하임")

    shots = _collect_shots(data)
    if not shots:
        return [(0.0, duration)]

    boundaries: List[float] = []
    for s in shots[1:]:
        try:
            t = float(s.get("start", 0.0))
        except Exception:
            continue
        if 0.0 < t < duration:
            boundaries.append(t)

    boundaries.sort()
    groups: List[Tuple[float, float]] = []
    cur_start = 0.0

    for t in boundaries:
        front = t - cur_start
        back = duration - t
        if front >= min_bgm_sec and back >= min_bgm_sec:
            groups.append((cur_start, t))
            cur_start = t

    if duration - cur_start > 0.1:
        groups.append((cur_start, duration))

    return groups


def build_full_bgm_track(
    groups: List[Tuple[float, float]],
    bgm_files: List[Path],
    out_audio: Path,
    volume_ratio: float = BGM_VOLUME_RATIO,
) -> bool:
    if not bgm_files:
        print("[final] BGM 파일이 없음")
        return False

    n = min(len(groups), len(bgm_files))
    if n == 0:
        print("[final] 사용할 그룹/BGM 쌍이 없음")
        return False

    BGM_MIX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[final] BGM 그룹 수: {len(groups)} / BGM 파일 수: {len(bgm_files)} → 사용 쌍: {n}")

    padded_files: List[Path] = []

    for i in range(n):
        g_start, g_end = groups[i]
        group_len = max(g_end - g_start, 0.0)
        if group_len <= 0.1:
            continue

        src_bgm = bgm_files[i]
        if not src_bgm.exists():
            print(f"[final] BGM 파일 없음 (skip): {src_bgm}")
            continue

        out_i = BGM_MIX_DIR / f"bgm_pad_{i+1:02d}.wav"

        # 🔥 여기만 수정: pad_dur/whole_dur 옵션 다 빼고, 안전하게 apad + atrim만 사용
        afilter = f"volume={volume_ratio},apad,atrim=0:{group_len}"

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(src_bgm),
            "-af", afilter,
            "-ar", "48000",
            "-ac", "2",
            str(out_i),
        ]
        print(f"[final] 그룹#{i+1}: {g_start:.2f}~{g_end:.2f} ({group_len:.2f}s) → {out_i.name}")
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as ex:
            print(f"[final] BGM 패딩/트림 실패 (#{i+1}): {ex}")
            return False

        padded_files.append(out_i)

    if not padded_files:
        print("[final] 패딩된 BGM 파일이 없음")
        return False

    concat_list = BGM_MIX_DIR / "concat.txt"
    with open(concat_list, "w", encoding="utf-8") as f:
        for p in padded_files:
            f.write(f"file '{p.name}'\n")

    cmd_concat = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", "concat.txt",
        "-c:a", "pcm_s16le",
        str(out_audio.name),
    ]
    print(f"[final] BGM 클립 concat → {out_audio}")
    try:
        subprocess.run(cmd_concat, check=True, cwd=str(BGM_MIX_DIR), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as ex:
        print(f"[final] BGM concat 실패: {ex}")
        return False

    return True


def mix_video_and_bgm(
    video_path: Path,
    bgm_path: Path,
    out_path: Path,
) -> bool:
    if not video_path.exists():
        print(f"[final] 영상 없음: {video_path}")
        return False
    if not bgm_path.exists():
        print(f"[final] BGM 트랙 없음: {bgm_path}")
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)

    filter_complex = "[0:a][1:a]amix=inputs=2:weights=1 1:normalize=1:duration=longest[aout]"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-i", str(bgm_path),
        "-filter_complex", filter_complex,
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(out_path),
    ]

    print(f"[final] 영상 + BGM 믹스 → {out_path}")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as ex:
        print(f"[final] 최종 믹스 실패: {ex}")
        return False

    return True


def main() -> None:
    print("\n=== Edit-Click: 최종 BGM 믹스 파이프라인 시작 (final_mapping.py) ===")

    if not HARDCUT_JSON.exists():
        print(f"[final] HARDCUT_JSON 없음: {HARDCUT_JSON.resolve()}")
        return
    if not SUBTITLED_VIDEO.exists():
        print(f"[final] 자막 영상 없음: {SUBTITLED_VIDEO.resolve()}")
        return
    if not BGM_DIR.exists():
        print(f"[final] BGM 폴더 없음: {BGM_DIR.resolve()}")
        return

    data = _load_json(HARDCUT_JSON)

    groups = compute_bgm_groups(data, MIN_BGM_SEC)
    if not groups:
        print("[final] BGM 그룹이 없음 → BGM 없이 종료")
        return

    print(f"[final] BGM 그룹 개수: {len(groups)}")
    for i, (s, e) in enumerate(groups, start=1):
        print(f"  - 그룹#{i}: {s:.2f} ~ {e:.2f} ({e - s:.2f}s)")

    bgm_files = sorted(BGM_DIR.glob("bgm_*.mp3"))
    if not bgm_files:
        print(f"[final] BGM mp3 파일 없음: {BGM_DIR.resolve()}")
        return

    ok = build_full_bgm_track(groups, bgm_files, FULL_BGM_AUDIO, BGM_VOLUME_RATIO)
    if not ok:
        print("[final] BGM 트랙 생성 실패")
        return

    ok = mix_video_and_bgm(SUBTITLED_VIDEO, FULL_BGM_AUDIO, FINAL_OUTPUT)
    if not ok:
        print("[final] 최종 영상 믹스 실패")
        return

    print("\n✅ 최종 믹스 완료")
    print(f"   - 자막 영상 : {SUBTITLED_VIDEO.resolve()}")
    print(f"   - BGM 트랙 : {FULL_BGM_AUDIO.resolve()}")
    print(f"   - 최종 결과 : {FINAL_OUTPUT.resolve()}")
