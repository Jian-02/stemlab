#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
separate_stems.py
mp3(또는 다른 오디오) 파일을 악기별 트랙(스템, stem)으로 분리하는 스크립트.

내부적으로 Meta(Facebook) Research가 만든 오픈소스 딥러닝 모델
Demucs(v4, htdemucs)를 사용합니다. 현재 시점에서 무료로 구할 수 있는
음원 분리 도구 중 품질이 가장 좋은 축에 속합니다.

기본 4-스템 분리: vocals(보컬) / drums(드럼) / bass(베이스) / other(그 외 악기)
6-스템 모델 사용 시: 위 4개 + guitar(기타) / piano(피아노)

사전 준비 (최초 1회):
    pip install demucs
    (PyTorch가 자동으로 함께 설치됩니다. GPU 없어도 CPU로 동작하지만
     느립니다. 3~4분짜리 곡 기준 CPU에서 수 분 정도 걸릴 수 있습니다.)

    ffmpeg도 필요합니다 (mp3 등 인코딩/디코딩용):
    - Windows: winget install ffmpeg
    - macOS:   brew install ffmpeg
    - Linux:   sudo apt install ffmpeg

사용법:
    # 기본 4-스템 분리 (vocals/drums/bass/other), 결과는 wav로 저장
    python separate_stems.py song.mp3

    # 결과를 mp3로 저장 (용량 절약, 약간의 음질 손실 있음)
    python separate_stems.py song.mp3 --mp3

    # 6-스템 분리 (기타/피아노까지 분리, 더 느림)
    python separate_stems.py song.mp3 --six-stem

    # 보컬만 뽑고 싶을 때 (반주 제거 = "가라오케" 모드)
    python separate_stems.py song.mp3 --two-stem vocals

    # 출력 폴더 지정
    python separate_stems.py song.mp3 -o ./separated

    # GPU가 있을 경우 강제로 GPU 사용 (cuda) / 없으면 자동으로 cpu 사용
    python separate_stems.py song.mp3 --device cuda
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def check_dependency(module_name: str, pip_name: str = None):
    pip_name = pip_name or module_name
    try:
        __import__(module_name)
        return True
    except ImportError:
        print(f"[안내] '{pip_name}' 패키지가 설치되어 있지 않습니다.")
        answer = input(f"지금 설치할까요? (pip install {pip_name}) [Y/n]: ").strip().lower()
        if answer in ("", "y", "yes"):
            subprocess.run([sys.executable, "-m", "pip", "install", pip_name], check=True)
            return True
        return False


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        sys.exit(
            "[오류] ffmpeg를 찾을 수 없습니다. 먼저 설치해 주세요.\n"
            "  - Windows: winget install ffmpeg\n"
            "  - macOS:   brew install ffmpeg\n"
            "  - Linux:   sudo apt install ffmpeg"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Demucs를 이용해 음악 파일을 악기별 스템으로 분리합니다."
    )
    parser.add_argument("input", help="분리할 오디오 파일 경로 (mp3, wav 등)")
    parser.add_argument("-o", "--outdir", default="separated",
                         help="결과를 저장할 폴더 (기본: ./separated)")
    parser.add_argument("--six-stem", action="store_true",
                         help="6-스템 모델 사용 (vocals/drums/bass/guitar/piano/other). 더 느림.")
    parser.add_argument("--two-stem", choices=["vocals", "drums", "bass", "other"],
                         help="지정한 트랙과 나머지(no_<트랙>) 두 개로만 분리 (예: vocals -> "
                              "vocals.wav + no_vocals.wav = 반주). 더 빠름.")
    parser.add_argument("--model", default=None,
                         help="Demucs 모델 이름 직접 지정 (기본: htdemucs, "
                              "6-stem 지정 시 htdemucs_6s). 고품질이 필요하면 "
                              "htdemucs_ft(느리지만 더 정교함) 추천.")
    parser.add_argument("--mp3", action="store_true",
                         help="결과 파일을 mp3로 저장 (기본은 wav, 무손실이지만 용량이 큼)")
    parser.add_argument("--mp3-bitrate", default="320",
                         help="--mp3 사용 시 비트레이트(kbps), 기본 320 (최대 음질)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None,
                         help="연산 장치 지정. 생략하면 자동 감지(GPU 있으면 cuda 사용)")
    parser.add_argument("--jobs", type=int, default=0,
                         help="CPU 병렬 처리 개수 (0=자동). CPU로 돌릴 때 속도에 영향")

    args = parser.parse_args()

    src = Path(args.input)
    if not src.is_file():
        sys.exit(f"[오류] 파일을 찾을 수 없습니다: {src}")

    check_ffmpeg()
    if not check_dependency("demucs"):
        sys.exit("[오류] demucs 없이는 진행할 수 없습니다.")
    try:
        import torch  # noqa: F401
    except ImportError:
        pass  # demucs 설치 시 함께 깔림

    model = args.model
    if not model:
        model = "htdemucs_6s" if args.six_stem else "htdemucs"

    cmd = [sys.executable, "-m", "demucs", "-n", model, "-o", args.outdir]

    if args.two_stem:
        cmd += ["--two-stems", args.two_stem]

    if args.mp3:
        cmd += ["--mp3", "--mp3-bitrate", str(args.mp3_bitrate)]

    if args.device:
        cmd += ["-d", args.device]

    if args.jobs:
        cmd += ["-j", str(args.jobs)]

    cmd += [str(src)]

    print("실행 명령어:", " ".join(cmd))
    print("모델을 처음 사용하는 경우 가중치 파일(수백MB)을 자동으로 내려받습니다. "
          "잠시 기다려 주세요.\n")

    # Windows에서 시스템 코드페이지(cp949)와 demucs 내부의 UTF-8 문자열(경로에
    # 한글이 섞여 있을 때 등)이 충돌해 UnicodeDecodeError가 나는 것을 막기 위해
    # 자식 파이썬 프로세스의 입출력을 UTF-8로 강제합니다.
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"          # Python 3.7+ : UTF-8 모드 강제
    env["PYTHONIOENCODING"] = "utf-8"  # 구버전 호환용

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        sys.exit(f"[실패] demucs 실행 중 오류가 발생했습니다 (code {result.returncode})")

    out_folder = Path(args.outdir) / model / src.stem
    print(f"\n[완료] 분리된 트랙은 다음 폴더에 저장되었습니다:\n  {out_folder}")


if __name__ == "__main__":
    main()