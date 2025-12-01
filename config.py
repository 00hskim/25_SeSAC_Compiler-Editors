# config.py

from __future__ import annotations
import os
import shutil
from typing import Dict, List


# 메인에서 부를 공개 API
def init_env() -> None:
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("TRANSFORMERS_IMAGE_TRANSFORMS_USE_PIL_ONLY", "1")

    _print_header("환경 점검")
    ok = _check_dependencies()
    _check_ffmpeg()

    if not ok:
        _print_install_hint()


# ===== 내부 유틸 =====

_REQ_ORDER = [
    "numpy",
    "torch",
    "torchaudio",
    "transformers",
    "accelerate",
    "soundfile",
    "librosa",
    "pyannote.audio",
    "pandas",
    "requests",
]


def _try_import(name: str):
    try:
        import importlib
        return importlib.import_module(name)
    except Exception:
        return None


def _check_dependencies() -> bool:
    found_versions: Dict[str, str | None] = {}
    missing: List[str] = []

    for pkg in _REQ_ORDER:
        mod = _try_import(pkg)
        ver = getattr(mod, "__version__", None) if mod is not None else None
        found_versions[pkg] = ver
        if mod is None:
            missing.append(pkg)

    _print_dep_table(found_versions, missing)

    required = {
        "numpy",
        "torch",
        "torchaudio",
        "transformers",
        "soundfile",
        "librosa",
        "pandas",
        "requests",
    }
    hard_missing = [m for m in missing if m in required]
    return len(hard_missing) == 0


def _check_ffmpeg() -> None:
    if shutil.which("ffmpeg"):
        print("ffmpeg: OK")
    else:
        print("⚠️  ffmpeg 미설치. 영상/오디오 처리 실패 가능")


def _print_dep_table(found: Dict[str, str | None], missing: List[str]) -> None:
    print("의존성 상태:")
    for k in _REQ_ORDER:
        v = found.get(k)
        mark = "MISSING" if k in missing else "OK"
        ver = v or ""
        print(f" - {k:17s} : {mark} {ver}")


def _print_install_hint() -> None:
    print("\n[안내] 일부 필수 패키지가 없습니다. 예시 설치 명령:")
    print(
        'pip install "numpy==1.26.4" "torch==2.2.2" "torchaudio==2.2.2" '
        '"transformers>=4.40" accelerate soundfile librosa pandas requests'
    )
    print("※ torch/torchaudio는 CUDA 버전에 맞게 선택 설치 필요.")


def _print_header(msg: str) -> None:
    print(f"\n=== {msg} ===")
