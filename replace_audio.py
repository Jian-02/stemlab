#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
replace_audio.py
mp4 등 영상 파일의 오디오 트랙을 다른 mp3(또는 다른 오디오 파일)로
갈아끼우는 스크립트. 비디오는 재인코딩 없이 그대로 복사하므로
화질 손실이 없습니다.

사전 준비:
    ffmpeg가 설치되어 있어야 합니다 (extract_mp3.py와 동일).

사용법:
    # 기본: video.mp4의 오디오를 new_audio.mp3로 교체 (비디오 화질 그대로,
    # 오디오는 320kbps AAC로 인코딩 -> output.mp4)
    python replace_audio.py video.mp4 new_audio.mp3

    # 출력 파일명 지정
    python replace_audio.py video.mp4 new_audio.mp3 -o result.mp4

    # 영상 길이와 오디오 길이가 다를 때, 더 짧은 쪽 길이에 맞춰 자르기
    python replace_audio.py video.mp4 new_audio.mp3 --shortest

    # 오디오를 재인코딩하지 않고 원본 코덱 그대로 넣기 (더 빠르고 무손실이지만
    # 일부 플레이어에서 호환성 문제가 있을 수 있음, 특히 mp3를 mp4에 그대로 넣을 때)
    python replace_audio.py video.mp4 new_audio.mp3 --audio-codec copy

    # 오디오 비트레이트 직접 지정
    python replace_audio.py video.mp4 new_audio.mp3 -b 256k

    # 비디오도 재인코딩하고 싶을 때 (특별한 이유가 없다면 비추천: 느리고 화질 저하)
    python replace_audio.py video.mp4 new_audio.mp3 --video-codec libx264 --crf 18
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def check_ffmpeg():
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        sys.exit(
            f"[오류] {', '.join(missing)}를 찾을 수 없습니다. ffmpeg를 설치하고 "
            "PATH에 등록해 주세요.\n"
            "  - Windows: winget install ffmpeg\n"
            "  - macOS:   brew install ffmpeg\n"
            "  - Linux:   sudo apt install ffmpeg"
        )


def get_duration(path: Path):
    """ffprobe로 길이(초)를 가져온다. 실패하면 None."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


def main():
    parser = argparse.ArgumentParser(
        description="영상 파일의 오디오 트랙을 다른 오디오 파일로 교체합니다."
    )
    parser.add_argument("video", help="원본 영상 파일 (mp4 등)")
    parser.add_argument("audio", help="새로 넣을 오디오 파일 (mp3 등)")
    parser.add_argument("-o", "--output", help="출력 파일 경로 (기본: <원본이름>_replaced.mp4)")
    parser.add_argument("--video-codec", default="copy",
                         help="비디오 코덱 (기본: copy = 재인코딩 없음, 화질 그대로)")
    parser.add_argument("--crf", default="18",
                         help="--video-codec을 copy가 아닌 값으로 바꿨을 때 화질 지정 "
                              "(0=무손실~51=저화질, 기본 18=고화질)")
    parser.add_argument("--audio-codec", default="aac",
                         choices=["aac", "copy", "mp3"],
                         help="오디오 코덱. aac(기본, mp4 표준/호환성 좋음), "
                              "copy(재인코딩 없이 원본 그대로, 무손실이지만 호환성 이슈 가능), "
                              "mp3(mp4 컨테이너에 mp3 그대로 삽입)")
    parser.add_argument("-b", "--audio-bitrate", default="320k",
                         help="오디오 비트레이트 (기본: 320k, aac/mp3 재인코딩 시 적용)")
    parser.add_argument("--shortest", action="store_true",
                         help="영상/오디오 길이가 다를 때 더 짧은 쪽에 맞춰서 자르기")

    args = parser.parse_args()
    check_ffmpeg()

    video = Path(args.video)
    audio = Path(args.audio)
    if not video.is_file():
        sys.exit(f"[오류] 영상 파일을 찾을 수 없습니다: {video}")
    if not audio.is_file():
        sys.exit(f"[오류] 오디오 파일을 찾을 수 없습니다: {audio}")

    output = Path(args.output) if args.output else video.with_name(f"{video.stem}_replaced.mp4")

    # 길이 비교 (참고용 경고 메시지)
    v_dur = get_duration(video)
    a_dur = get_duration(audio)
    if v_dur is not None and a_dur is not None:
        diff = abs(v_dur - a_dur)
        if diff > 1.0:
            longer = "오디오" if a_dur > v_dur else "영상"
            print(f"[안내] 영상 길이 {v_dur:.1f}초, 오디오 길이 {a_dur:.1f}초로 "
                  f"{diff:.1f}초 차이가 있습니다 ({longer}가 더 김).")
            if not args.shortest:
                print("       --shortest 옵션 없이 진행하면 더 긴 쪽 끝부분에는 "
                      "영상만(무음) 또는 소리만(검은 화면 아님, 마지막 프레임 유지) 남을 수 있습니다.")

    cmd = ["ffmpeg", "-y", "-i", str(video), "-i", str(audio)]
    cmd += ["-map", "0:v:0", "-map", "1:a:0"]
    cmd += ["-c:v", args.video_codec]
    if args.video_codec != "copy":
        cmd += ["-crf", str(args.crf)]

    if args.audio_codec == "copy":
        cmd += ["-c:a", "copy"]
    else:
        codec = "libmp3lame" if args.audio_codec == "mp3" else "aac"
        cmd += ["-c:a", codec, "-b:a", args.audio_bitrate]

    if args.shortest:
        cmd += ["-shortest"]

    cmd += [str(output)]

    print("실행 명령어:", " ".join(cmd))
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )

    if result.returncode != 0:
        print("[실패] ffmpeg 실행 중 오류가 발생했습니다.")
        print(result.stderr[-2000:])
        sys.exit(1)

    print(f"[완료] {output}")


if __name__ == "__main__":
    main()
