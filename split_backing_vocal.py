#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_backing_vocal.py
이미 "보컬만" 분리되어 있는 오디오 파일에서, 리드 보컬과 백코러스/하모니를
다시 한 번 나누는 스크립트.

원리:
    Ultimate Vocal Remover(UVR) 커뮤니티가 만든 "가라오케(Karaoke)" 계열
    모델은 원래 노래방 반주를 만들기 위해, 리드 보컬만 쏙 빼내고 나머지
    (반주 + 백코러스/하모니)를 남기도록 학습되었습니다.

    여기에 이미 보컬만 있는 파일(반주가 없는 파일)을 넣으면:
      - "Vocals" 출력  -> 리드 보컬만 깨끗하게 남습니다.
      - "Instrumental" 출력 -> 반주가 없으니 백코러스/하모니만 남게 됩니다.

    다만 보컬-반주 분리보다 훨씬 어려운 작업입니다 (리드 보컬과 백코러스가
    같은 사람 목소리라 음색·주파수 대역이 크게 겹칩니다). 곡에 따라, 특히
    백코러스가 리드와 유니즌(같은 음)으로 겹치는 구간에서는 결과가 완벽하지
    않을 수 있습니다.

사전 준비 (최초 1회):
    pip install "audio-separator[cpu]"
    (NVIDIA GPU가 있다면 "audio-separator[gpu]" 로 설치하면 더 빠릅니다.)

    ffmpeg도 필요합니다 (mp3 출력 시 pydub이 사용).

사용법:
    # 빠른 모델(MDX-Net, 기본)
    python split_backing_vocal.py vocals.wav

    # 고품질 모델(Mel-Roformer, 더 느리고 다운로드 용량이 큼)
    python split_backing_vocal.py vocals.wav --preset quality

    # 결과를 mp3로 저장
    python split_backing_vocal.py vocals.wav --mp3

    # 모델 파일명을 직접 지정하고 싶을 때 (--preset보다 우선)
    python split_backing_vocal.py vocals.wav --model UVR_MDXNET_KARA_2.onnx
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# preset 이름 -> 실제 모델 파일명
# fast: MDX-Net 기반, 가볍고 빠름 (약 50MB 다운로드)
# quality: Mel-Band Roformer 기반, 최신 아키텍처라 품질이 더 좋지만 느리고
#          모델 파일도 훨씬 큼 (수백 MB)
MODEL_PRESETS = {
    "fast": "UVR_MDXNET_KARA_2.onnx",
    "quality": "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
}


def main():
    parser = argparse.ArgumentParser(
        description="이미 보컬만 있는 오디오 파일에서 리드 보컬 / 백코러스(하모니)를 나눕니다."
    )
    parser.add_argument("input", help="보컬만 있는 오디오 파일 경로 (반주가 섞여 있으면 품질이 떨어집니다)")
    parser.add_argument("-o", "--outdir", default="separated_backing",
                         help="결과를 저장할 폴더 (기본: ./separated_backing)")
    parser.add_argument("--preset", choices=list(MODEL_PRESETS.keys()), default="fast",
                         help="fast=빠름(기본) / quality=고품질(느림, 모델 다운로드 큼)")
    parser.add_argument("--model", default=None,
                         help="모델 파일명을 직접 지정 (지정하면 --preset은 무시됩니다)")
    parser.add_argument("--mp3", action="store_true",
                         help="결과 파일을 mp3(320kbps)로 저장 (기본은 wav, 무손실)")
    parser.add_argument("--model-dir", default=str(BASE_DIR / "backing_models"),
                         help="모델 가중치 파일을 캐시해둘 폴더 (기본: 이 스크립트 옆의 backing_models/)")
    args = parser.parse_args()

    src = Path(args.input)
    if not src.is_file():
        sys.exit(f"[오류] 파일을 찾을 수 없습니다: {src}")

    try:
        from audio_separator.separator import Separator
    except ImportError as exc:
        # 실제로는 audio-separator 자체가 아니라 그 하위 의존성(onnxruntime,
        # torch, librosa 등)이 "이 파이썬 환경"에는 없어서 나는 경우가 많다.
        # 어떤 파이썬/환경에서 실행됐는지와 실제 원인을 그대로 보여준다.
        sys.exit(
            "[오류] audio-separator를 import할 수 없습니다.\n"
            f"  실행 중인 파이썬: {sys.executable}\n"
            f"  실제 오류: {exc.__class__.__name__}: {exc}\n"
            '  이 파이썬 환경에 설치해 주세요: "이 경로의 python.exe" -m pip install "audio-separator[cpu]"\n'
            f'  예) "{sys.executable}" -m pip install "audio-separator[cpu]"'
        )

    model_filename = args.model or MODEL_PRESETS[args.preset]

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    kwargs = {
        "output_dir": args.outdir,
        "model_file_dir": args.model_dir,
    }
    if args.mp3:
        kwargs["output_format"] = "MP3"
        kwargs["output_bitrate"] = "320k"
    else:
        kwargs["output_format"] = "WAV"

    print(f"모델: {model_filename}")
    print("처음 사용하는 모델이면 가중치 파일을 자동으로 내려받습니다. 잠시 기다려 주세요.\n")

    sep = Separator(**kwargs)
    sep.load_model(model_filename=model_filename)
    out_files = sep.separate(str(src))

    print("\n[완료] 결과 파일:")
    for f in out_files:
        p = Path(f)
        full = p if p.is_absolute() else Path(args.outdir) / p
        print(" -", full)


if __name__ == "__main__":
    main()
