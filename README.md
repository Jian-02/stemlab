# StemLab

영상·음원에서 소리를 뽑아 파트별로 해체하고, 다시 조립하는 로컬 오디오 작업대.

MP4에서 오디오를 추출하고, 곡을 악기별 스템으로 분리하고, "보컬만" 트랙에서
리드 보컬과 백코러스/하모니를 다시 나누고, 보컬 멜로디를 악보(MIDI/MusicXML)로
자동 채보하고, 영상의 오디오를 다른 음원으로 갈아끼우는 것까지 한 곳에서 처리합니다.

전부 **내 컴퓨터에서만** 도는 도구입니다. 외부로 파일을 올리지 않고, 웹앱도
`127.0.0.1`(로컬호스트)에서만 열립니다.

---

## 주요 기능

| 기능 | 스크립트 | 사용하는 엔진 |
|------|----------|---------------|
| MP4 등 영상 → MP3 추출 | `extract_mp3.py` | ffmpeg |
| 악기별 스템 분리 (보컬/드럼/베이스/기타/피아노/그 외) | `separate_stems.py` | [Demucs](https://github.com/facebookresearch/demucs) v4 |
| 리드 보컬 ↔ 백코러스/하모니 분리 | `split_backing_vocal.py` | [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) (UVR 가라오케 계열 모델) |
| 보컬 멜로디 자동 채보 → MIDI / MusicXML | `transcribe_vocal.py` | [Basic Pitch](https://github.com/spotify/basic-pitch) (ONNX) + music21 |
| 영상 오디오 트랙 교체 (화질 무손실) | `replace_audio.py` | ffmpeg |
| 위 기능들을 묶은 로컬 웹앱 (업로드·진행률·재생·다운로드) | `app.py` | Flask |

---

## 요구사항

- **Python 3.10+** (개발·테스트는 Windows / Python 3.12 기준)
- **ffmpeg / ffprobe** — 시스템 PATH에 등록되어 있어야 합니다
  - Windows: `winget install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`
- GPU는 없어도 됩니다. Demucs·백코러스 분리는 CPU로도 동작하지만 느립니다
  (3~4분짜리 곡 기준 수 분). NVIDIA GPU가 있으면 훨씬 빠릅니다.

---

## 설치

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

이걸로 `extract_mp3.py` / `separate_stems.py` / `split_backing_vocal.py` /
`replace_audio.py` / `app.py` 가 전부 동작합니다.

### 자동 채보(`transcribe_vocal.py`)만 추가 설치가 필요합니다

Basic Pitch를 그냥 설치하면 기본 의존성으로 `tensorflow<2.15.1` 을 끌고 오는데,
이게 이 프로젝트의 다른 기능이 요구하는 `numpy>=2` 와 충돌하고 Windows/Python 3.12용
배포판도 없어서 설치가 깨집니다. 다행히 Basic Pitch 안에 ONNX 버전 모델이 이미
들어있어서, TensorFlow 없이 설치할 수 있습니다 — **아래 순서 그대로** 두 번 실행하세요:

```bash
pip install -r requirements-transcribe.txt
pip install --no-deps basic-pitch
```

`--no-deps` 가 핵심입니다. 빼면 TensorFlow까지 같이 끌려와 충돌이 재현됩니다.

---

## 실행 전 준비 (매번)

새 터미널을 열 때마다 가상환경을 먼저 활성화합니다.

```bash
# Windows (PowerShell)
venv\Scripts\Activate.ps1
# Windows (cmd)
venv\Scripts\activate.bat
# macOS / Linux
source venv/bin/activate
```

> 활성화가 번거로우면 가상환경의 파이썬을 직접 지정해서 실행해도 됩니다.
> 특히 여러 파이썬이 깔린 Windows에서 "패키지를 분명히 설치했는데 import가 안 된다"
> 싶을 때는 이 방식이 확실합니다:
> ```bash
> venv\Scripts\python.exe app.py
> venv\Scripts\python.exe separate_stems.py song.mp3
> ```

---

## 웹앱 사용법 (권장)

```bash
python app.py
```

실행 후 브라우저에서 **http://127.0.0.1:5000** 접속. 종료는 터미널에서 `Ctrl+C`.

- 파일(mp3/mp4/wav 등, 최대 2GB)을 업로드하고 분리 모드를 고르면 큐에 들어갑니다.
- 작업은 한 번에 하나씩 순서대로 처리합니다 (동시에 여러 개 돌리면 CPU/GPU가
  감당하기 힘들어서 큐로 관리).
- 진행률·재생·다운로드를 화면에서 바로 할 수 있습니다.
- 분리가 끝난 "보컬" 스템은 다시 업로드할 필요 없이 그 자리에서 바로
  **백코러스 분리** 나 **악보 만들기** 작업으로 넘길 수 있습니다.
- 작업 히스토리는 `jobs.json` 에 저장되어 앱을 껐다 켜도 남습니다.

### 분리 모드

| 모드 | 설명 |
|------|------|
| 4-스템 | 보컬 / 드럼 / 베이스 / 그 외 악기 (`htdemucs`) |
| 4-스템 고품질 | `htdemucs_ft` — 약 4배 느리지만 더 정교함 |
| 6-스템 | 위 + 기타 / 피아노 (`htdemucs_6s`, 피아노는 부정확할 수 있음) |
| 보컬만 분리 | 반주 제거, 가장 빠름 |
| 백코러스/하모니 분리 (빠름/고품질) | **이미 보컬만 있는 파일 전용**. 리드 보컬 ↔ 백코러스 |
| 보컬 멜로디 악보로 만들기 | **이미 보컬만 있는 파일 전용**. MIDI/MusicXML 자동 채보 |

---

## CLI 스크립트 사용법

각 스크립트는 웹앱 없이 단독으로도 쓸 수 있습니다. `-h` 로 전체 옵션을 볼 수 있습니다.

### MP4 → MP3 추출

```bash
python extract_mp3.py input.mp4                    # 320kbps CBR
python extract_mp3.py input.mp4 --vbr              # 최고 품질 VBR
python extract_mp3.py input.mp4 --copy -o out.m4a  # 재인코딩 없이 무손실 추출
python extract_mp3.py --batch ./videos --outdir ./mp3s   # 폴더 일괄 변환
```

### 악기별 스템 분리

```bash
python separate_stems.py song.mp3                  # 4-스템, wav 출력
python separate_stems.py song.mp3 --mp3            # mp3 출력
python separate_stems.py song.mp3 --six-stem       # 6-스템 (기타/피아노)
python separate_stems.py song.mp3 --two-stem vocals   # 보컬 + 반주
```

### 리드 보컬 ↔ 백코러스/하모니 분리

`separate_stems.py --two-stem vocals` 등으로 **먼저 보컬만 뽑아둔 파일**을 넣습니다.

```bash
python split_backing_vocal.py vocals.wav                  # 빠른 모델 (MDX-Net)
python split_backing_vocal.py vocals.wav --preset quality # 고품질 (Mel-Roformer)
python split_backing_vocal.py vocals.wav --mp3
```

`(Vocals)` 출력 = 리드 보컬, `(Instrumental)` 출력 = 백코러스/하모니.
리드와 백코러스가 유니즌으로 겹치는 구간은 결과가 완벽하지 않을 수 있습니다.

### 보컬 멜로디 자동 채보

```bash
python transcribe_vocal.py vocals.mp3                          # 보컬에서 템포 추정
python transcribe_vocal.py vocals.mp3 --tempo-source original.mp3  # 원곡으로 템포 추정 (추천)
python transcribe_vocal.py vocals.mp3 --tempo 96              # 템포 직접 지정
```

자동 채보는 초안입니다 — 화음·랩·미분음에 약하고, 결과 악보는 사람이 다듬어야 합니다.

### 영상 오디오 갈아끼우기

```bash
python replace_audio.py video.mp4 new_audio.mp3               # 비디오 화질 그대로, 오디오 AAC 320k
python replace_audio.py video.mp4 new_audio.mp3 --shortest    # 길이 다르면 짧은 쪽에 맞춤
python replace_audio.py video.mp4 new_audio.mp3 -o result.mp4
```

---

## 폴더 구조

```
stemlab/
├── app.py                      # 로컬 웹앱 (Flask)
├── extract_mp3.py              # MP4 → MP3
├── separate_stems.py           # 악기별 스템 분리 (Demucs)
├── split_backing_vocal.py      # 리드/백코러스 분리 (audio-separator)
├── transcribe_vocal.py         # 보컬 자동 채보 (Basic Pitch + music21)
├── replace_audio.py            # 영상 오디오 교체
├── requirements.txt
├── requirements-transcribe.txt # 자동 채보용 (basic-pitch 제외)
├── templates/index.html        # 웹앱 프론트엔드
├── backing_models/             # 백코러스 분리 모델 가중치 캐시 (자동 다운로드)
├── uploads/<작업ID>/           # 웹앱 업로드 원본
├── separated/<작업ID>/         # 웹앱 분리 결과
└── jobs.json                   # 웹앱 작업 히스토리
```

> 한글·특수문자 파일명이 ffmpeg·Demucs·Windows 경로 처리에서 인코딩 문제를
> 일으키지 않도록, 웹앱은 디스크에 작업ID 기반 이름으로 저장하고 화면에는
> 원래 파일명을 보여줍니다.

---

## 참고 / 한계

- **Demucs 6-스템의 피아노·기타 분리는 Demucs 자체의 알려진 한계**로 정확도가
  낮은 경우가 많습니다 (Meta 공식 문서에도 명시). 일반 악기 분리 품질이 더 좋은
  `htdemucs_ft` 옵션을 대안으로 제공합니다.
- 백코러스 분리는 리드와 백코러스가 같은 사람 목소리라 주파수 대역이 크게 겹쳐,
  보컬-반주 분리보다 훨씬 어렵습니다.
- 자동 채보는 단선율(멜로디) 전용입니다. 화음이 섞이면 정확도가 크게 떨어집니다.
- 모델 가중치(수십~수백 MB)는 각 엔진이 처음 실행될 때 자동으로 내려받습니다.

## 모델 출처

- [Demucs](https://github.com/facebookresearch/demucs) — Meta Research, MIT License
- [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) / UVR 커뮤니티 모델
- [Basic Pitch](https://github.com/spotify/basic-pitch) — Spotify, Apache-2.0
- [music21](https://github.com/cuthbertLab/music21) — MIT / BSD
