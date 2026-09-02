#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_mp3.py
mp4 등 동영상 파일에서 오디오만 뽑아 mp3로 저장하는 스크립트.
내부적으로 ffmpeg를 사용하며, 원본 오디오 트랙을 재인코딩 없이
최고 품질(320kbps CBR 또는 VBR 최고 품질)로 변환합니다.

사전 준비:
    ffmpeg가 설치되어 있어야 합니다.
    - Windows: https://ffmpeg.org/download.html 에서 받아 PATH에 추가
      (또는 `winget install ffmpeg`, `choco install ffmpeg`)
    - macOS: `brew install ffmpeg`
    - Linux: `sudo apt install ffmpeg` (배포판에 따라 다름)

사용법:
    # 파일 하나 변환 (기본: 320kbps CBR, 최고 음질)
    python extract_mp3.py input.mp4

    # 출력 파일명 지정
    python extract_mp3.py input.mp4 -o output.mp3

    # VBR 최고 품질(-q:a 0, 대략 220~260kbps 가변)로 변환
    python extract_mp3.py input.mp4 --vbr

    # 비트레이트 직접 지정 (예: 192k)
    python extract_mp3.py input.mp4 -b 192k

    # 폴더 안의 모든 동영상 파일 일괄 변환
    python extract_mp3.py --batch ./videos --outdir ./mp3s

    # 원본이 이미 mp3/AAC 등이라 손실 없이 오디오 스트림만 뽑고 싶을 때
    # (컨테이너만 바꾸는 것이라 재인코딩 없음 -> 화질/음질 완전 무손실)
    python extract_mp3.py input.mp4 --copy -o output.m4a
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

VIDEO_EXTS = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv",
    ".wmv", ".m4v", ".ts", ".mpg", ".mpeg",
}


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        sys.exit(
            "[오류] ffmpeg를 찾을 수 없습니다. 먼저 ffmpeg를 설치하고 "
            "PATH에 등록해 주세요.\n"
            "  - Windows: winget install ffmpeg (또는 공식 사이트에서 다운로드)\n"
            "  - macOS:   brew install ffmpeg\n"
            "  - Linux:   sudo apt install ffmpeg"
        )


def build_command(src: Path, dst: Path, bitrate: str, vbr: bool, copy: bool):
    cmd = ["ffmpeg", "-y", "-i", str(src), "-vn"]  # -vn: 비디오 스트림 제거

    if copy:
        # 재인코딩 없이 오디오 스트림을 그대로 컨테이너만 바꿔서 추출.
        # 원본 오디오 코덱이 mp3가 아니면 mp3 컨테이너에 못 넣을 수 있으니
        # 이 경우 출력 확장자는 원본 코덱에 맞는 것(m4a 등)을 권장.
        cmd += ["-acodec", "copy"]
    elif vbr:
        # libmp3lame VBR 품질 스케일: 0(최고, ~220-260kbps) ~ 9(최저)
        cmd += ["-codec:a", "libmp3lame", "-q:a", "0"]
    else:
        # CBR 고정 비트레이트. 320k는 mp3 규격상 최고 비트레이트.
        cmd += ["-codec:a", "libmp3lame", "-b:a", bitrate, "-ar", "44100"]

    cmd += [str(dst)]
    return cmd


def convert_one(src: Path, dst: Path, bitrate: str, vbr: bool, copy: bool, overwrite: bool = True):
    if dst.exists() and not overwrite:
        print(f"[건너뜀] 이미 존재함: {dst}")
        return True

    cmd = build_command(src, dst, bitrate, vbr, copy)
    print(f"[변환 중] {src.name} -> {dst.name}")
    # Windows에서 시스템 코드페이지(cp949)와 ffmpeg의 UTF-8 출력이 충돌해
    # UnicodeDecodeError가 나는 것을 방지하기 위해 인코딩을 명시적으로 지정합니다.
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )

    if result.returncode != 0:
        print(f"[실패] {src.name}")
        print(result.stderr[-2000:])  # ffmpeg 에러 로그 뒷부분만 출력
        return False

    print(f"[완료] {dst}")
    return True


def main():
    parser = argparse.ArgumentParser(description="동영상 파일에서 오디오(mp3)만 추출합니다.")
    parser.add_argument("input", nargs="?", help="변환할 입력 파일 경로")
    parser.add_argument("-o", "--output", help="출력 파일 경로 (기본: 입력 파일명.mp3)")
    parser.add_argument("-b", "--bitrate", default="320k",
                         help="mp3 비트레이트 (기본: 320k, mp3 규격상 최대치)")
    parser.add_argument("--vbr", action="store_true",
                         help="고정 비트레이트 대신 최고 품질 VBR(-q:a 0) 사용")
    parser.add_argument("--copy", action="store_true",
                         help="재인코딩 없이 오디오 스트림만 그대로 추출 (완전 무손실, "
                              "출력 확장자는 원본 오디오 코덱에 맞춰 지정 권장 예: .m4a)")
    parser.add_argument("--batch", metavar="DIR", help="폴더 내 모든 동영상 파일을 일괄 변환")
    parser.add_argument("--outdir", metavar="DIR", help="--batch 사용 시 결과를 저장할 폴더")
    parser.add_argument("--no-overwrite", action="store_true",
                         help="출력 파일이 이미 있으면 건너뛰기")

    args = parser.parse_args()
    check_ffmpeg()

    ext = "m4a" if args.copy and not args.output else "mp3"

    if args.batch:
        src_dir = Path(args.batch)
        if not src_dir.is_dir():
            sys.exit(f"[오류] 폴더를 찾을 수 없습니다: {src_dir}")

        out_dir = Path(args.outdir) if args.outdir else src_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        files = [p for p in sorted(src_dir.iterdir()) if p.suffix.lower() in VIDEO_EXTS]
        if not files:
            sys.exit(f"[오류] 폴더에 동영상 파일이 없습니다: {src_dir}")

        print(f"총 {len(files)}개 파일 변환을 시작합니다.")
        ok = fail = 0
        for f in files:
            dst = out_dir / f.with_suffix(f".{ext}").name
            success = convert_one(f, dst, args.bitrate, args.vbr, args.copy,
                                   overwrite=not args.no_overwrite)
            ok += success
            fail += not success
        print(f"\n완료: {ok}개 성공, {fail}개 실패")
        return

    if not args.input:
        parser.error("입력 파일 경로를 지정하거나 --batch 옵션을 사용하세요.")

    src = Path(args.input)
    if not src.is_file():
        sys.exit(f"[오류] 파일을 찾을 수 없습니다: {src}")

    dst = Path(args.output) if args.output else src.with_suffix(f".{ext}")
    convert_one(src, dst, args.bitrate, args.vbr, args.copy,
                overwrite=not args.no_overwrite)


if __name__ == "__main__":
    main()