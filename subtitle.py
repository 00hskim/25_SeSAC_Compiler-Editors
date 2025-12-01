# subtitle.py
from __future__ import annotations
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

HARDCUT_JSON   = Path("./Work/Sample_analysis_hardcut.json")
INPUT_VIDEO    = Path("./Work/hardcut.mp4")
OUTPUT_VIDEO   = Path("./Work/hardcut_subtitled.mp4")
FONT_FILE      = Path("./fonts/NotoSansKR-Regular.otf")
DRAWFILTER_TXT = Path("./Work/drawtext_filter.txt")

SUB_START_MARGIN  = 0.08
SUB_END_MARGIN    = 0.08
SUB_MIN_DURATION  = 0.30
SUB_GLOBAL_DELAY  = 2.5


def _load_segments(path: Path) -> List[Dict[str, Any]]:
    print(f"[load] HARDCUT_JSON path       : {path.resolve()}")
    if not path.exists():
        raise FileNotFoundError(f"[subtitle] JSON 없음: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    shots = data.get("shots", {})
    print(f"[load] shots keys               : {list(shots.keys())}")
    segments: List[Dict[str, Any]] = []

    for shot_key, lst in shots.items():
        if not lst:
            print(f"[load] shot '{shot_key}' is empty, skip")
            continue
        shot = lst[0]
        segs = shot.get("segments", [])
        print(f"[load] shot '{shot_key}' segments count: {len(segs)}")
        for seg in segs:
            segments.append(seg)

    print(f"[load] total segments collected : {len(segments)}")
    segments.sort(key=lambda s: float(s.get("start", 0.0)))
    return segments


def _ffmpeg_escape_text(text: str) -> str:
    if text is None:
        return ""
    t = str(text)
    t = t.replace("\\", "\\\\")
    t = t.replace("\"", "\\\"")
    t = t.replace(":", "\\:")
    t = t.replace(",", "\\,")
    t = t.replace("'", "")
    t = t.replace("\n", " ")
    return t


def _decide_style(seg: Dict[str, Any]) -> Tuple[str, int]:
    color = "white"
    size = 42

    emo = (seg.get("emotion") or "").lower()
    features = seg.get("features") or {}
    try:
        rms_db = float(features.get("rms_db"))
    except Exception:
        rms_db = None
    tone_tag = (features.get("tone_tag") or "").lower()

    if emo in ("ang", "anger"):
        color = "red"
        size = 54
    elif emo in ("sad", "sadness"):
        color = "deepskyblue"
        size = 48
    elif emo in ("hap", "joy"):
        color = "yellow"
        size = 50

    if rms_db is not None:
        if rms_db > -25:
            size += 6
        elif rms_db < -35:
            size -= 4

    if tone_tag:
        if "bright" in tone_tag:
            color = "yellow"
        elif "warm" in tone_tag and emo in ("sad", "sadness"):
            color = "lightblue"
        elif "dark" in tone_tag and emo in ("angry", "anger"):
            color = "orangered"

    size = max(32, min(size, 64))
    return color, size


def _build_drawtext_filters(segments: List[Dict[str, Any]]) -> str:
    filters: List[str] = []

    font_path_resolved = FONT_FILE.resolve()
    font_posix = font_path_resolved.as_posix()
    print(f"[build] FONT_FILE resolved      : {font_path_resolved}")
    print(f"[build] FONT_FILE exists        : {font_path_resolved.exists()}")

    for idx, seg in enumerate(segments):
        try:
            s = float(seg.get("start", 0.0))
            e = float(seg.get("end", 0.0))
        except Exception as ex:
            print(f"[build] seg[{idx}] invalid time: {ex} -> {seg}")
            continue

        if e <= s or (e - s) < 0.05:
            print(f"[build] seg[{idx}] skip (too short or invalid) start={s:.3f}, end={e:.3f}")
            continue

        vis_s = s + SUB_START_MARGIN
        vis_e = e - SUB_END_MARGIN
        if vis_e - vis_s < SUB_MIN_DURATION:
            vis_s, vis_e = s, e

        vis_s += SUB_GLOBAL_DELAY
        vis_e += SUB_GLOBAL_DELAY

        if vis_e <= vis_s:
            print(f"[build] seg[{idx}] skip (vis_e<=vis_s) vis_s={vis_s:.3f}, vis_e={vis_e:.3f}")
            continue

        raw_text = seg.get("text", "") or ""
        stripped = raw_text.strip()
        safe_text = _ffmpeg_escape_text(stripped)
        if not safe_text:
            print(f"[build] seg[{idx}] skip (empty text)")
            continue

        color, fontsize = _decide_style(seg)

        f = (
            f"drawtext=fontfile='{font_posix}'"
            f":text=\"{safe_text}\""
            f":fontcolor={color}"
            f":fontsize={fontsize}"
            f":borderw=2"
            f":x=(w-text_w)/2"
            f":y=h-140"
            f":enable='between(t,{vis_s:.3f},{vis_e:.3f})'"
        )
        filters.append(f)

        print(
            f"[build] seg[{idx}] "
            f"time={s:.3f}-{e:.3f} vis={vis_s:.3f}-{vis_e:.3f} "
            f"len={len(stripped)} color={color} size={fontsize} "
            f"text='{stripped[:30]}'"
        )

    print(f"[build] total drawtext filters  : {len(filters)}")

    if not filters:
        raise RuntimeError("[subtitle] 유효한 세그먼트가 없음 (필터 0개)")

    vf = ",".join(filters)
    print(f"[build] vf_filter length        : {len(vf)}")
    print(f"[build] vf_filter preview(400)  : {vf[:400]}")
    return vf


def _run_ffmpeg_drawtext(vf_filter: str) -> bool:
    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    DRAWFILTER_TXT.parent.mkdir(parents=True, exist_ok=True)

    with open(DRAWFILTER_TXT, "w", encoding="utf-8") as f:
        f.write(vf_filter)
    print(f"[run] drawtext filter saved     : {DRAWFILTER_TXT.resolve()}")

    preview = vf_filter[:200].replace("\n", " ")
    print(f"[run] -vf preview(200)          : {preview}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(INPUT_VIDEO),
        "-vf",
        vf_filter,
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(OUTPUT_VIDEO),
    ]

    print(f"[run] ffmpeg command:")
    print("      " + " ".join(str(c) for c in cmd))

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    print(f"[run] ffmpeg returncode         : {proc.returncode}")
    print("----- ffmpeg stdout (first 400) -----")
    print(proc.stdout[:400])
    print("----- end stdout -----")
    print("----- ffmpeg stderr (first 2000) -----")
    print(proc.stderr[:2000])
    print("----- end stderr -----")

    if proc.returncode != 0:
        print("[run] ffmpeg failed, no subtitle applied")
        return False

    out_exists = OUTPUT_VIDEO.exists()
    out_size = OUTPUT_VIDEO.stat().st_size if out_exists else 0
    print(f"[run] output video exists       : {out_exists}")
    print(f"[run] output video size (bytes) : {out_size}")

    return True


def main() -> None:
    print("\n=== Edit-Click: subtitle 파이프라인 시작 (drawtext, single-pass) ===")
    print(f"[main] INPUT_VIDEO path         : {INPUT_VIDEO.resolve()}")
    print(f"[main] HARDCUT_JSON path        : {HARDCUT_JSON.resolve()}")
    print(f"[main] OUTPUT_VIDEO path        : {OUTPUT_VIDEO.resolve()}")
    print(f"[main] FONT_FILE path           : {FONT_FILE.resolve()}")

    if not INPUT_VIDEO.exists():
        print(f"[main] ERROR: INPUT_VIDEO 없음  : {INPUT_VIDEO.resolve()}")
        return
    if not HARDCUT_JSON.exists():
        print(f"[main] ERROR: HARDCUT_JSON 없음: {HARDCUT_JSON.resolve()}")
        return

    segments = _load_segments(HARDCUT_JSON)
    print(f"[main] 세그먼트 개수            : {len(segments)}")

    try:
        vf_filter = _build_drawtext_filters(segments)
    except Exception as ex:
        print(f"[main] ERROR in _build_drawtext_filters: {ex}")
        return

    ok = _run_ffmpeg_drawtext(vf_filter)
    if not ok:
        print("[main] 자막 입히기 실패")
        return

    print("\n[main] ✅ subtitle 완료")
    print(f"[main] 입력 영상 : {INPUT_VIDEO.resolve()}")
    print(f"[main] 출력 영상 : {OUTPUT_VIDEO.resolve()}")
    print(f"[main] 필터 로그 : {DRAWFILTER_TXT.resolve()}")
