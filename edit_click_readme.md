# Edit-Click 🎬  
**음성·장면 분석 기반 End-to-End AI 영상 편집 보조 도구**

한 번 클릭으로 **컷 · 자막 · BGM 초안을 자동 생성**하는 AI 편집 자동화 엔진입니다.  
사용자가 원본 영상을 업로드하면 자동으로 다음 프로세스가 실행됩니다:

- 📊 음성 내용 + 감정 + 오디오 리듬 분석  
- ✂️ 샷(Shot) 정보 기반 하드컷 자동 편집  
- 🎵 감정/리듬 기반 BGM 자동 생성  
- 📝 타임라인 정합 자막 생성 및 합성  

CLI 기준으로  
`Input/Sample.mp4 → Output/Sample_Result.mp4`  
까지 **완전 자동화된 End-To-End 파이프라인** 구성.

---

# 1. 전체 파이프라인 👁️

```
[1] 영상 업로드 (Input/Sample.mp4) 
 ↓
[2] 오디오 추출 (ffmpeg, 16kHz mono WAV)     
 ↓ 
[3] 장면(샷) 분석 (TransNet V2)
    → shot_1, shot_2, ... + start/end 타임
 ↓ 
[4] Whisper Large-v3 STT 🎙️
    → 문장 단위 타임스탬프 + 전체 스크립트
 ↓ 
[5] 감정/리듬 분석 (Wav2Vec2 SER + librosa) 💓
    → 감정 레이블, BPM, tone_tag, rhythm_tag
 ↓ 
[6] 통합 분석 JSON 생성 (Sample_analysis.json)
    → 샷 + 세그먼트 + 감정 + 오디오 피처 통합
 ↓ 
[7] 하드컷 편집 (hardcut.py) ✂️
    → ffmpeg cut/concat → Work/hardcut.mp4
    → 타임라인 기준 JSON 재생성
 ↓ 
[8] 자막 합성 (subtitle.py) 💬
    → drawtext 기반 하드렌더링
 ↓ 
[9] BGM 생성 (bgm_create.py, Stable Audio 2) 🎹
    → 감정 시퀀스 기반 BGM 생성
 ↓ 
[10] 최종 믹스 (final_mapping.py) 💿
    → amix로 원본 대사 + BGM 믹싱
    → Output/Sample_Result.mp4
```

---

# 2. 로드맵 (Roadmap) 🚀

## 2-1. 사용자 피드백 기반 편집 루프 🔄
- 컷/자막/BGM에 대한 사용자 반영  
- 세미 오토 모드
  - 컷만 자동  
  - 자막만 자동  
  - BGM만 자동  

---

## 2-2. 편집 스타일 템플릿 🎨
- 콘텐츠별 스타일
  - Vlog / 강의 / 게임 / 쇼츠 / 인터뷰  
- 사용자 프로필 저장  
- “내 채널 스타일 유지”

---

## 2-3. 시각 정보 기반 하드컷 고도화 👁️
- 얼굴 인식  
- 클로즈업/와이드  
- 카메라 움직임  
- 밝기/노출 변화  

---

## 2-4. 동적 자막 (Kinetic Typography) ✨
- 감정 기반 자막 애니메이션  
- 강조 단어 하이라이트  
- 스타일 프리셋:
  - minimal-clean  
  - highlight-words  
  - meme-style  
  - lecture-keypoint  

---

## 2-5. BGM 엔진 개선 🎼
- 씬 간 BGM 키/BPM 연결성  
- 다층 구조:
  - 기본 BGM + 클라이맥스용 Layer  
- 자동 볼륨 커브  

---

## 2-6. 프로젝트 / 협업 워크플로우 🤝
- 프로젝트 단위 관리  
- JSON/로그 버전 관리  
- 결과 Export:
  - Premiere EDL  
  - DaVinci XML  
  - FinalCut JSON  

---

# 3. 실행 방법 💻

## 3-1. 요구환경 ⚙️
- Python 3.10+
- NVIDIA GPU (VRAM 12GB 권장)
- ffmpeg 설치
- Stable Audio 2 API Key

```bash
ffmpeg -version
```

---

## 3-2. 설치 📦

```bash
git clone https://github.com/_____   # 실제 URL
cd Edit-Click
pip install -r requirements.txt
```

---

# 4. 문의 / 기여 / 협업 🙌
- Issue → 질문/제안  
- Pull Request → 환영  
- 연구/프로젝트 협업 가능  

---

# 5. License 📄  
(예: MIT / Apache 2.0 / GPL 등)
