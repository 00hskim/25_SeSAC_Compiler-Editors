# bgm_create.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import json
from pathlib import Path
import math
import requests

# ===== 설정 =====
HARDCUT_JSON_PATH = Path("./Work/Sample_analysis_hardcut.json")
BGM_DIR = Path("./Work/BGM")

STABLE_AUDIO_API_KEY = os.getenv("STABLE_AUDIO_API_KEY")

MIN_BGM_WINDOW_SEC = 15.0   # 한 BGM 구간 최소 길이
BGM_MARGIN_SEC = 1.5        # 영상 길이보다 BGM을 이만큼 짧게 생성

STABLE_AUDIO_ENDPOINT = "https://api.stability.ai/v2beta/audio/stable-audio-2/text-to-audio"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"[bgm] JSON 없음: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_shots(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    shots_raw = data.get("shots", {})
    shots: List[Dict[str, Any]] = []
    for key, lst in shots_raw.items():
        if not lst:
            continue
        shot = lst[0]
        shots.append(
            {
                "key": key,
                "start": float(shot.get("start", 0.0)),
                "end": float(shot.get("end", 0.0)),
                "segments": shot.get("segments", []),
            }
        )
    shots.sort(key=lambda s: s["start"])
    return shots


def _build_bgm_groups(shots: List[Dict[str, Any]], min_window: float) -> List[Tuple[float, float]]:
    if not shots:
        return []

    groups: List[Tuple[float, float]] = []
    cur_start = shots[0]["start"]
    cur_end = shots[0]["end"]

    for shot in shots[1:]:
        s = shot["start"]
        e = shot["end"]

        if s > cur_end:
            cur_end = s

        cur_len = cur_end - cur_start
        if cur_len < min_window:
            cur_end = e
        else:
            groups.append((cur_start, cur_end))
            cur_start, cur_end = s, e

    groups.append((cur_start, cur_end))

    if len(groups) >= 2:
        last_s, last_e = groups[-1]
        if (last_e - last_s) < min_window:
            prev_s, prev_e = groups[-2]
            groups[-2] = (prev_s, last_e)
            groups.pop()

    return groups


def _collect_segments_for_group(
    shots: List[Dict[str, Any]],
    g_start: float,
    g_end: float,
) -> List[Tuple[Dict[str, Any], float]]:
    segs_with_weight: List[Tuple[Dict[str, Any], float]] = []

    for shot in shots:
        for seg in shot.get("segments", []):
            try:
                s = float(seg.get("start", 0.0))
                e = float(seg.get("end", 0.0))
            except Exception:
                continue
            if e <= g_start or s >= g_end:
                continue
            overlap = min(e, g_end) - max(s, g_start)
            if overlap <= 0:
                continue
            segs_with_weight.append((seg, overlap))

    return segs_with_weight


def _analyze_mood(segs_with_weight: List[Tuple[Dict[str, Any], float]]) -> Dict[str, Any]:
    emo_dur = {"hap": 0.0, "ang": 0.0, "sad": 0.0, "neu": 0.0, "other": 0.0}
    bpm_sum = 0.0
    bpm_w = 0.0
    arr_counts: Dict[str, float] = {}
    rhythm_counts: Dict[str, float] = {}

    for seg, w in segs_with_weight:
        label = str(seg.get("emotion") or "").lower()
        if label.startswith("hap"):
            key = "hap"
        elif label.startswith("ang"):
            key = "ang"
        elif label.startswith("sad"):
            key = "sad"
        elif label.startswith("neu"):
            key = "neu"
        else:
            key = "other"
        emo_dur[key] += w

        feats = seg.get("features") or {}
        bpm = feats.get("bpm_est")
        if isinstance(bpm, (int, float)) and bpm > 0:
            bpm_sum += bpm * w
            bpm_w += w

        for tag_key, counter in [
            ("arrangement_tag", arr_counts),
            ("rhythm_tag", rhythm_counts),
        ]:
            tag_val = feats.get(tag_key)
            if isinstance(tag_val, str) and tag_val:
                counter[tag_val] = counter.get(tag_val, 0.0) + w

    total_dur = sum(emo_dur.values()) or 1.0
    ratios = {k: v / total_dur for k, v in emo_dur.items()}

    anger_ratio = ratios["ang"]
    sad_ratio = ratios["sad"]
    happy_ratio = ratios["hap"]
    neu_ratio = ratios["neu"]

    primary = max(emo_dur.items(), key=lambda x: x[1])[0]
    secondary = max(
        (item for item in emo_dur.items() if item[0] != primary),
        key=lambda x: x[1],
        default=("other", 0.0),
    )[0]

    bright_ratio = happy_ratio + neu_ratio * 0.7
    neg_strength = max(anger_ratio, sad_ratio)

    if bright_ratio >= 0.65:
        mood_core = "soft, positive and lightly uplifting, with a relaxed atmosphere"
    elif bright_ratio >= 0.45:
        mood_core = "calm, warm and slightly hopeful"
    elif bright_ratio >= 0.25:
        mood_core = "neutral and steady, with a gentle tone"
    else:
        mood_core = "neutral and unobtrusive, slightly on the serious side"

    if neg_strength >= 0.6:
        mood_core = "calm but clearly serious and somewhat heavy in tone"
    elif neg_strength >= 0.4 and bright_ratio < 0.5:
        mood_core = "calm and a bit tense or somber, but still understated"

    nuance_parts: List[str] = []
    if anger_ratio > 0.15 and primary != "ang":
        if anger_ratio >= 0.3:
            nuance_parts.append("with a gentle sense of background tension")
        else:
            nuance_parts.append("with a subtle, almost imperceptible sense of tension")
    if happy_ratio > 0.15 and primary != "hap":
        if happy_ratio >= 0.3:
            nuance_parts.append("with some quiet warmth and optimism")
        else:
            nuance_parts.append("with a light, soft positive undertone")
    if sad_ratio > 0.15 and primary != "sad":
        if sad_ratio >= 0.3:
            nuance_parts.append("with a soft, reflective feeling")
        else:
            nuance_parts.append("with a gentle touch of melancholy")

    mood_nuance = ""
    if nuance_parts:
        mood_nuance = " " + " ".join(nuance_parts)

    if bpm_w > 0:
        avg_bpm = bpm_sum / bpm_w
    else:
        avg_bpm = 100.0

    if bright_ratio >= 0.55:
        tempo_phrase = "a light, moderately upbeat tempo around 95–105 BPM"
    elif bright_ratio >= 0.35:
        tempo_phrase = "a relaxed mid-tempo pace around 90–100 BPM"
    else:
        tempo_phrase = "a calm, slightly slower tempo around 80–90 BPM"

    def _pick_top_label(counter: Dict[str, float]) -> str | None:
        if not counter:
            return None
        return max(counter.items(), key=lambda x: x[1])[0]

    arrangement = _pick_top_label(arr_counts)
    rhythm = _pick_top_label(rhythm_counts)

    texture_parts: List[str] = []
    if arrangement:
        texture_parts.append(f"arrangement that feels like \"{arrangement}\"")
    if rhythm:
        texture_parts.append(f"rhythmic feel similar to \"{rhythm}\"")

    texture_phrase = ""
    if texture_parts:
        texture_phrase = "The texture should have " + ", ".join(texture_parts) + "."

    return {
        "mood_core": mood_core,
        "mood_nuance": mood_nuance,
        "tempo_phrase": tempo_phrase,
        "avg_bpm": avg_bpm,
        "texture_phrase": texture_phrase,
    }


def _build_prompt_for_group(
    idx: int,
    g_start: float,
    g_end: float,
    mood_info: Dict[str, Any],
) -> str:
    length_sec = g_end - g_start
    approx_len = max(length_sec - BGM_MARGIN_SEC, 4.0)

    mood_core = mood_info["mood_core"]
    mood_nuance = mood_info["mood_nuance"]
    tempo_phrase = mood_info["tempo_phrase"]
    texture_phrase = mood_info["texture_phrase"]

    prompt = (
        "Soft, modern background music for a talking-head YouTube or documentary-style video. "
        "The track should feel unobtrusive and slightly positive by default. "
        "No vocals, no lyrics, no lead melody that competes with speech. "
        "The music should stay in the background and leave space for voices.\n\n"
        f"Scene description: this is BGM segment #{idx}, covering roughly "
        f"{length_sec:.1f} seconds of the final edited video, starting around "
        f"{g_start:.1f} seconds. The cue should feel {mood_core}{mood_nuance}. "
        f"Use {tempo_phrase}.\n\n"
        "Instrumentation: focus on light and bright sounds such as soft piano, "
        "gentle acoustic guitar, subtle plucks or mallet instruments, airy synth pads, "
        "and light percussion (shakers, snaps, soft kick). Avoid heavy drums, "
        "avoid dark drones, and avoid overly boomy low-end. Prefer a clear and fresh tone.\n\n"
        "Mixing: keep the overall volume gentle so that dialogue can sit clearly on top. "
        "Keep the character more bright and airy than dark or dramatic. "
        "No abrupt endings; fade out smoothly at the end.\n\n"
        f"Target duration is about {approx_len:.1f} seconds. "
    )

    if texture_phrase:
        prompt += "\n" + texture_phrase

    return prompt


def _generate_bgm_segment(
    idx: int,
    prompt: str,
    duration_sec: float,
    output_dir: Path,
    steps: int = 30,
    output_format: str = "mp3",
) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"bgm_{idx:02d}.{output_format}"

    print(f"[bgm] Stable Audio 요청 #{idx}: duration={duration_sec:.1f}s")
    try:
        resp = requests.post(
            STABLE_AUDIO_ENDPOINT,
            headers={
                "authorization": f"Bearer {STABLE_AUDIO_API_KEY}",
                "accept": "audio/*",
            },
            files={"none": ""},
            data={
                "prompt": prompt,
                "output_format": output_format,
                "duration": float(f"{duration_sec:.2f}"),
                "steps": steps,
            },
            timeout=180,
        )
    except Exception as e:
        print(f"[bgm] Stable Audio 요청 실패(네트워크 오류): {e}")
        return None

    if resp.status_code == 200:
        with open(out_path, "wb") as f:
            f.write(resp.content)
        print(f"[bgm] 생성 완료 → {out_path}")
        return out_path
    else:
        try:
            err = resp.json()
        except Exception:
            err = resp.text
        print(f"[bgm] Stable Audio 응답 에러: status={resp.status_code}, msg={err}")
        return None


def main() -> None:
    print("\n=== Edit-Click: BGM 생성 파이프라인 시작 (bgm_create.py) ===")

    data = _load_json(HARDCUT_JSON_PATH)

    video_meta = data.get("video", {})
    duration = float(video_meta.get("duration", 0.0))
    print(f"[bgm] 하드컷 이후 영상 길이: {duration:.3f} sec")

    shots = _extract_shots(data)
    print(f"[bgm] 샷 개수: {len(shots)}")

    if not shots:
        print("[bgm] 샷 정보가 없어 BGM 그룹을 만들 수 없음. 종료.")
        return

    groups = _build_bgm_groups(shots, MIN_BGM_WINDOW_SEC)
    print(f"[bgm] BGM 그룹 개수: {len(groups)}")
    for i, (gs, ge) in enumerate(groups, start=1):
        print(f"  - 그룹#{i}: {gs:.2f} ~ {ge:.2f} ({ge - gs:.2f}s)")

    if not groups:
        print("[bgm] BGM 그룹이 없음. 영상이 너무 짧거나 설정이 잘못된 듯.")
        return

    for idx, (gs, ge) in enumerate(groups, start=1):
        segs_with_w = _collect_segments_for_group(shots, gs, ge)
        if not segs_with_w:
            print(f"[bgm] 그룹#{idx}: 세그먼트가 없어 기본 neutral BGM으로 처리")
            dummy_seg = {
                "emotion": "neu",
                "features": {},
            }
            segs_with_w = [(dummy_seg, ge - gs)]

        mood_info = _analyze_mood(segs_with_w)
        prompt = _build_prompt_for_group(idx, gs, ge, mood_info)
        target_len = max((ge - gs) - BGM_MARGIN_SEC, 4.0)

        _generate_bgm_segment(
            idx=idx,
            prompt=prompt,
            duration_sec=target_len,
            output_dir=BGM_DIR,
        )

    print("\n✅ BGM 생성 파이프라인 종료")
    print(f"   - JSON 기준: {HARDCUT_JSON_PATH.resolve()}")
    print(f"   - BGM 출력 폴더: {BGM_DIR.resolve()}")
