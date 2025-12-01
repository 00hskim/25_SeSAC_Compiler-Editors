# hardcut.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import json
import subprocess
from pathlib import Path

INPUT_VIDEO   = Path("./Input/Sample.mp4")
RAW_JSON      = Path("./Work/Sample_analysis.json")
HARDCUT_VIDEO = Path("./Work/hardcut.mp4")
HARDCUT_JSON  = Path("./Work/Sample_analysis_hardcut.json")
CLIP_DIR      = Path("./Work/hardcut_clips")

TRIM_HEAD_SEC       = 2.0
TRIM_TAIL_SEC       = 2.0
SHOT_MARGIN_SEC     = 3.5
LONG_SEG_THRESH_SEC = 15.0
SEG_TAIL_CUT_SEC    = 2.0
EPS                 = 1e-6


def _load_analysis(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"[hardcut] 분석 JSON 없음: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_shots(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    shots = []
    for _, lst in data.get("shots", {}).items():
        if not lst:
            continue
        shots.append(lst[0])
    shots.sort(key=lambda s: float(s.get("start", 0.0)))
    return shots


def _collect_segments(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    segs: List[Dict[str, Any]] = []
    for _, lst in data.get("shots", {}).items():
        if not lst:
            continue
        shot = lst[0]
        for seg in shot.get("segments", []):
            segs.append(seg)
    return segs


def _add_interval(intervals: List[Tuple[float, float]], s: float, e: float, duration: float) -> None:
    s = max(0.0, min(float(s), duration))
    e = max(0.0, min(float(e), duration))
    if e - s > EPS:
        intervals.append((s, e))


def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    intervals = [(min(a, b), max(a, b)) for a, b in intervals if abs(b - a) > EPS]
    intervals.sort(key=lambda x: x[0])

    merged: List[Tuple[float, float]] = []
    cur_s, cur_e = intervals[0]

    for s, e in intervals[1:]:
        if s <= cur_e + EPS:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def _build_keeps(cuts: List[Tuple[float, float]], duration: float) -> List[Tuple[float, float]]:
    keeps: List[Tuple[float, float]] = []
    cur = 0.0
    for s, e in cuts:
        if s - cur > EPS:
            keeps.append((cur, s))
        cur = max(cur, e)
    if duration - cur > EPS:
        keeps.append((cur, duration))
    return keeps


def _compute_offsets(keeps: List[Tuple[float, float]]) -> Tuple[List[float], float]:
    offsets: List[float] = []
    acc = 0.0
    for s, e in keeps:
        offsets.append(acc)
        acc += (e - s)
    return offsets, acc


def compute_cut_and_keep_intervals(
    data: Dict[str, Any],
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], float]:
    video_meta = data.get("video", {})
    duration = float(video_meta.get("duration", 0.0))
    if duration <= 0:
        raise ValueError("[hardcut] video.duration 이 0초 이하임")

    shots = _collect_shots(data)
    segments = _collect_segments(data)

    print(f"[hardcut] duration: {duration:.3f} sec")
    print(f"[hardcut] shots   : {len(shots)} 개")
    print(f"[hardcut] segments: {len(segments)} 개")

    cuts: List[Tuple[float, float]] = []

    # 전체 앞/뒤 컷
    _add_interval(cuts, 0.0, TRIM_HEAD_SEC, duration)
    _add_interval(cuts, duration - TRIM_TAIL_SEC, duration, duration)

    # 샷 전환 앞/뒤 컷
    for i in range(1, len(shots)):
        t = float(shots[i].get("start", 0.0))
        _add_interval(cuts, t - SHOT_MARGIN_SEC, t + SHOT_MARGIN_SEC, duration)

    # 긴 세그먼트 끝부분 컷
    for seg in segments:
        try:
            s = float(seg.get("start", 0.0))
            e = float(seg.get("end", 0.0))
        except Exception:
            continue
        if e <= s:
            continue
        length = e - s
        if length >= LONG_SEG_THRESH_SEC:
            tail_s = e - SEG_TAIL_CUT_SEC
            _add_interval(cuts, tail_s, e, duration)

    print(f"[hardcut] raw cut intervals: {len(cuts)} 개")
    cuts_merged = _merge_intervals(cuts)
    print(f"[hardcut] merged cut intervals: {len(cuts_merged)} 개")
    for i, (s, e) in enumerate(cuts_merged, start=1):
        print(f"  - cut#{i}: {s:.3f} ~ {e:.3f} ({e - s:.3f}s)")

    keeps = _build_keeps(cuts_merged, duration)
    print(f"[hardcut] keep intervals: {len(keeps)} 개")
    for i, (s, e) in enumerate(keeps, start=1):
        print(f"  - keep#{i}: {s:.3f} ~ {e:.3f} ({e - s:.3f}s)")

    return cuts_merged, keeps, duration


def rebuild_json_with_shots(
    data: Dict[str, Any],
    keeps: List[Tuple[float, float]],
    hardcut_duration: float,
) -> Dict[str, Any]:
    shots_dict = data.get("shots", {})
    fps = float(data.get("video", {}).get("fps", 30.0))

    # 1) 원본 세그먼트 → 순서대로 평탄화 + 연속 타임라인 재배치
    flat: List[Dict[str, Any]] = []

    def _shot_start(k: str) -> float:
        lst = shots_dict.get(k, [])
        if lst and isinstance(lst, list):
            try:
                return float(lst[0].get("start", 0.0))
            except Exception:
                return 0.0
        return 0.0

    for shot_key in sorted(shots_dict.keys(), key=_shot_start):
        lst = shots_dict.get(shot_key, [])
        if not lst:
            continue
        shot = lst[0]
        for seg in shot.get("segments", []):
            s_val = seg.get("start", 0.0)
            e_val = seg.get("end", 0.0)
            try:
                s_float = float(s_val)
            except Exception:
                s_float = 0.0
            try:
                e_float = float(e_val)
            except Exception:
                e_float = s_float
            seg_copy = dict(seg)
            seg_copy["_orig_seg_start"] = s_float
            seg_copy["_orig_seg_end"] = e_float
            flat.append(seg_copy)

    new_segments: List[Dict[str, Any]] = []
    t = 0.0
    for seg in flat:
        s_orig = seg.pop("_orig_seg_start")
        e_orig = seg.pop("_orig_seg_end")
        length = max(e_orig - s_orig, 0.0)
        if length <= 0.01:
            continue
        seg["orig_start"] = round(s_orig, 3)
        seg["orig_end"] = round(e_orig, 3)
        seg["start"] = round(t, 2)
        seg["end"] = round(t + length, 2)
        t += length
        new_segments.append(seg)

    logical_duration = t

    # 필요 시 세그먼트 전체 길이를 hardcut_duration에 맞게 스케일링
    if hardcut_duration > 0 and logical_duration > hardcut_duration + 1e-3:
        ratio = hardcut_duration / logical_duration
        print(f"[hardcut] 세그먼트 타임라인이 영상보다 김 → 비율 {ratio:.6f} 로 스케일링")
        t = 0.0
        for seg in new_segments:
            length = seg["end"] - seg["start"]
            new_len = length * ratio
            seg["start"] = round(t, 2)
            seg["end"] = round(t + new_len, 2)
            t += new_len
        logical_duration = t

    # 2) keeps + 원본 샷을 이용해, hardcut 이후 샷 구간 계산
    orig_shots: List[Dict[str, Any]] = _collect_shots(data)
    shot_infos: List[Dict[str, Any]] = []

    for shot in orig_shots:
        s0 = float(shot.get("start", 0.0))
        e0 = float(shot.get("end", 0.0))
        kept_len = 0.0
        for ks, ke in keeps:
            inter_s = max(s0, ks)
            inter_e = min(e0, ke)
            if inter_e - inter_s > EPS:
                kept_len += (inter_e - inter_s)
        shot_infos.append(
            {
                "orig_start": s0,
                "orig_end": e0,
                "probability": float(shot.get("probability", 1.0)),
                "kept_len": kept_len,
            }
        )

    new_shots_list: List[Dict[str, Any]] = []
    cur = 0.0
    for info in shot_infos:
        kept_len = info["kept_len"]
        if kept_len <= EPS:
            continue
        s_new = cur
        e_new = cur + kept_len
        cur = e_new
        new_shots_list.append(
            {
                "start": round(s_new, 3),
                "end": round(e_new, 3),
                "start_frame": int(round(s_new * fps)),
                "end_frame": int(round(e_new * fps)),
                "probability": info["probability"],
                "segments": [],
            }
        )

    # 3) 세그먼트를 샷 구간에 다시 배정 (mid 기준)
    if not new_shots_list:
        # 샷이 전부 잘려 나갔다면, 전체를 shot_1로 묶기
        new_shots_list = [
            {
                "start": 0.0,
                "end": round(logical_duration, 3),
                "start_frame": 0,
                "end_frame": int(round(logical_duration * fps)),
                "probability": 1.0,
                "segments": [],
            }
        ]

    for seg in new_segments:
        s = float(seg["start"])
        e = float(seg["end"])
        mid = 0.5 * (s + e)
        attached = False
        for shot in new_shots_list:
            st = float(shot["start"])
            ed = float(shot["end"])
            if st - 1e-6 <= mid <= ed + 1e-6:
                shot["segments"].append(seg)
                attached = True
                break
        if not attached:
            new_shots_list[-1]["segments"].append(seg)

    # 4) 최종 메타 구성
    video_meta = dict(data.get("video", {}))
    orig_duration = float(video_meta.get("duration", hardcut_duration))
    video_meta["orig_duration"] = round(orig_duration, 3)
    video_meta["duration"] = round(hardcut_duration, 3)
    video_meta["logical_duration"] = round(logical_duration, 3)

    whisper = dict(data.get("whisper", {}))
    whisper["num_segments"] = len(new_segments)

    new_shots_dict: Dict[str, List[Dict[str, Any]]] = {}
    for idx, shot in enumerate(new_shots_list, start=1):
        key = f"shot_{idx}"
        new_shots_dict[key] = [shot]

    new_data = dict(data)
    new_data["video"] = video_meta
    new_data["shots"] = new_shots_dict
    new_data["whisper"] = whisper

    return new_data


def ffmpeg_cut_and_concat(
    src_video: Path,
    keeps: List[Tuple[float, float]],
    out_video: Path,
) -> bool:
    if not src_video.exists():
        print(f"[hardcut] 원본 영상 없음: {src_video}")
        return False
    if not keeps:
        print("[hardcut] keep 구간이 없음 → 편집 불가")
        return False

    CLIP_DIR.mkdir(parents=True, exist_ok=True)

    clip_paths: List[Path] = []

    for i, (s, e) in enumerate(keeps, start=1):
        clip_path = CLIP_DIR / f"part_{i:03d}.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", f"{s:.3f}",
            "-to", f"{e:.3f}",
            "-i", str(src_video),
            "-c", "copy",
            str(clip_path),
        ]
        print(f"[ffmpeg] clip#{i}: {s:.3f} ~ {e:.3f} → {clip_path.name}")
        try:
            subprocess.run(cmd, check=True)
        except Exception as ex:
            print(f"[ffmpeg] 클립 생성 실패 (clip#{i}): {ex}")
            return False
        clip_paths.append(clip_path)

    concat_file = CLIP_DIR / "concat.txt"
    with open(concat_file, "w", encoding="utf-8") as f:
        for p in clip_paths:
            f.write(f"file '{p.name}'\n")

    out_video.parent.mkdir(parents=True, exist_ok=True)
    cmd_concat = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", "concat.txt",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(out_video.resolve()),
    ]
    print(f"[ffmpeg] concat → {out_video}")
    try:
        subprocess.run(cmd_concat, check=True, cwd=str(CLIP_DIR))
    except Exception as ex:
        print(f"[ffmpeg] concat 실패: {ex}")
        return False

    return True


def main() -> None:
    print("\n=== Edit-Click: hardcut 편집 시작 ===")

    if not INPUT_VIDEO.exists():
        print(f"[hardcut] INPUT_VIDEO 없음: {INPUT_VIDEO.resolve()}")
        return
    if not RAW_JSON.exists():
        print(f"[hardcut] RAW_JSON 없음: {RAW_JSON.resolve()}")
        return

    data = _load_analysis(RAW_JSON)

    cuts, keeps, duration = compute_cut_and_keep_intervals(data)
    if not keeps:
        print("[hardcut] keep 구간이 없어 종료합니다.")
        return

    _, hardcut_duration = _compute_offsets(keeps)
    print(f"[hardcut] new_duration (after cut): {hardcut_duration:.3f} sec")

    new_data = rebuild_json_with_shots(data, keeps, hardcut_duration)
    new_data["hardcut"] = {
        "cuts": [[round(s, 3), round(e, 3)] for (s, e) in cuts],
        "keeps": [[round(s, 3), round(e, 3)] for (s, e) in keeps],
        "total_duration": round(hardcut_duration, 3),
    }

    HARDCUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(HARDCUT_JSON, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    print(f"[hardcut] 하드컷 JSON 저장 완료 → {HARDCUT_JSON}")

    ok = ffmpeg_cut_and_concat(INPUT_VIDEO, keeps, HARDCUT_VIDEO)
    if not ok:
        print("[hardcut] 영상 편집 실패")
        return

    print("\n✅ hardcut 완료")
    print(f"   - 입력 영상 : {INPUT_VIDEO.resolve()}")
    print(f"   - 편집 영상 : {HARDCUT_VIDEO.resolve()}")
    print(f"   - 분석 JSON : {HARDCUT_JSON.resolve()}")
