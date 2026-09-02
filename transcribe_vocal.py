#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transcribe_vocal.py
보컬(또는 다른 단선율) 오디오 파일을 자동으로 채보해서 오선보(MusicXML)와
MIDI 파일로 만들어주는 스크립트.

원리:
    1) Spotify가 만든 오픈소스 자동 채보(AMT) 모델 "Basic Pitch"로 오디오에서
       음표(피치/시작/끝 시간)를 뽑는다. TensorFlow 대신 ONNX Runtime으로
       돌린다 (아래 "사전 준비" 참고 — 같은 가상환경에서 다른 기능과
       충돌 없이 쓰기 위한 선택이다).
    2) 비브라토 등으로 원래 하나였던 음이 여러 개의 짧은 음표로 쪼개져 나오는
       경우가 많아서, 같은 음높이의 인접한 음표를 하나로 합친다.
    3) 배음/포먼트를 잘못 잡아서 실제로는 하나의 멜로디인데 다른 음높이가
       같은 시간대에 겹쳐 나오는 "유령 음"들을 정리해서 항상 단선율이 되게
       만든다.
    4) 원곡(반주가 있는 원본 파일)에서 실제 박(비트)이 찍힌 "시각들"을
       직접 뽑아낸다(하나의 평균 BPM만 구하는 게 아니라, 곡 전체에 걸쳐
       실제 박이 어디어디에 있는지를 그대로 사용한다 — 사람이 부르는
       노래는 기계처럼 완벽하게 일정한 속도가 아니라서, 평균 BPM 하나로
       계산하면 곡 뒷부분으로 갈수록 박자가 밀리기 쉽다). 음표들의 시작/끝
       시각을 이 실제 박 위치 기준으로 "몇 번째 박(4분음표)인지"로 바꿔서,
       템포가 살짝 흔들려도 각 음표가 그 순간의 실제 박에 맞게 배치되도록
       한다.
    5) music21으로 그 결과를 박자(4분음표를 4등분/3등분한 격자)에 맞게
       양자화(quantize)하고, 마디를 새로 나눠서 MIDI와 MusicXML로 내보낸다.

한계 (중요):
    - 자동 채보는 완벽하지 않다. 특히 미분음, 매우 빠른 패시지, 말하듯
      부르는 창법(랩 등)에서 음정/박자가 부정확하게 잡힐 수 있다.
    - 화음(여러 음이 동시에 울리는 것)은 잘 못 잡는다 — 원래 단선율(멜로디)
      악보용으로 만들어졌다. 피아노/기타 반주를 채보하면 훨씬 부정확하다.
    - 박 추적(beat tracking)도 휴리스틱이다. 이상하게 나오면 --tempo 로
      직접 BPM을 지정해보자(이 경우 일정한 템포로 가정하고 만든다).
    - 결과 악보는 "초안"이라고 생각하고, 실제로 쓰려면 사람이 다듬어야 한다.

사전 준비 (최초 1회):
    basic-pitch가 기본으로 요구하는 tensorflow<2.15.1 은 numpy 2 이상과
    호환이 안 되고, Windows/Python 3.12용 배포판(wheel) 자체가 없어서
    설치가 아예 깨진다. 이 프로젝트의 다른 기능(백코러스 분리)이 쓰는
    audio-separator는 numpy>=2 가 필요하므로, tensorflow 없이 basic-pitch를
    설치해야 한다 — 다행히 basic-pitch 안에 ONNX 버전 모델이 이미 들어있어서
    tensorflow 없이 ONNX Runtime만으로도 완전히 동작한다.

    그래서 설치는 두 단계로 나눠서 한다 (순서 상관 없음):

        pip install -r requirements-transcribe.txt
        pip install --no-deps basic-pitch

    "--no-deps"가 중요하다 — 이게 없으면 pip이 basic-pitch의 기본
    tensorflow 의존성까지 끌고 오려다가 위 충돌이 그대로 재현된다.
    실제 필요한 라이브러리(librosa, music21, pretty_midi, onnxruntime 등)는
    requirements-transcribe.txt에 이미 다 들어있다.

사용법:
    # 보컬 파일만 있을 때 (템포는 보컬 자체에서 추정 시도)
    python transcribe_vocal.py vocals.mp3

    # 원곡(드럼 있는 원본)을 템포 추정용으로 같이 지정 (추천 — 훨씬 정확함)
    python transcribe_vocal.py vocals.mp3 --tempo-source original.mp3

    # 템포를 직접 지정하고 싶을 때 (자동 보정도 하지 않음)
    python transcribe_vocal.py vocals.mp3 --tempo 96

    # 출력 폴더 지정
    python transcribe_vocal.py vocals.mp3 -o ./transcribed
"""

import argparse
import sys
from pathlib import Path


def get_beat_grid(audio_path):
    """오디오에서 실제 박(비트)이 찍힌 시각들을 뽑아낸다. 평균 BPM 하나만
    구해서 처음부터 끝까지 일정한 속도라고 가정하면, 사람이 부르는(기계처럼
    완벽하게 정박이 아닌) 노래는 곡 뒷부분으로 갈수록 계산이 실제와 어긋나기
    쉽다. 대신 실제로 감지된 박 위치들을 그대로 격자로 써서, 음표 시각을
    "몇 번째 박인지"로 변환할 때 그때그때의 실제 박자에 맞춰지도록 한다.

    반환값: (박 시각 배열(초 단위, np.ndarray), 참고용 평균 BPM) — 박을
    하나도 못 찾으면 (None, None)."""
    import librosa
    import numpy as np

    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    if len(beat_times) < 2:
        return None, None

    avg_bpm = 60.0 / float(np.median(np.diff(beat_times)))
    # 흔한 옥타브 오차(박 추적이 실제 박의 절반/두 배 속도로 잡히는 것)를
    # 상식적인 범위로 보정한다 — 박 시각 배열 자체를 보정해야 하므로,
    # 부족하면 박 사이 중점을 끼워넣고(2배로 촘촘하게), 과하면 하나씩
    # 걸러낸다(절반으로 성기게).
    while avg_bpm < 70 and len(beat_times) > 1:
        midpoints = (beat_times[:-1] + beat_times[1:]) / 2.0
        beat_times = np.sort(np.concatenate([beat_times, midpoints]))
        avg_bpm *= 2
    while avg_bpm > 180 and len(beat_times) > 2:
        beat_times = beat_times[::2]
        avg_bpm /= 2

    return beat_times, avg_bpm


def time_to_beats(t, beat_times):
    """초 단위 시각 t를, 실제 감지된 박(beat_times) 기준으로 "몇 번째 박
    (=4분음표 몇 개 분량)"인지로 바꾼다. 박과 박 사이는 선형 보간하고, 곡의
    맨 처음 박 이전/마지막 박 이후는 그 근처 박 간격을 그대로 이어서
    바깥쪽으로 연장한다."""
    import numpy as np

    beat_times = np.asarray(beat_times)
    idx = np.arange(len(beat_times))
    if t <= beat_times[0]:
        step = beat_times[1] - beat_times[0]
        return float((t - beat_times[0]) / step)
    if t >= beat_times[-1]:
        step = beat_times[-1] - beat_times[-2]
        return float((len(beat_times) - 1) + (t - beat_times[-1]) / step)
    return float(np.interp(t, beat_times, idx))


def merge_close_notes(note_events, gap_threshold=0.05):
    """Basic Pitch는 비브라토나 음량 흔들림 때문에, 실제로는 하나로 이어지는
    음을 여러 개의 짧은 음표로 쪼개서 내놓는 경우가 흔하다. 같은 음높이의
    음표가 아주 짧은 간격(기본 0.05초)을 두고 바로 이어지면 하나로 합쳐서,
    악보에 자잘한 음표가 덕지덕지 붙는 걸 줄인다."""
    if not note_events:
        return note_events
    events = sorted(note_events, key=lambda e: e[0])
    merged = [list(events[0])]
    for ev in events[1:]:
        start, end, pitch, amp, bend = ev
        last = merged[-1]
        if pitch == last[2] and start - last[1] <= gap_threshold:
            last[1] = max(last[1], end)
            last[3] = max(last[3], amp)
        else:
            merged.append(list(ev))
    return [tuple(e) for e in merged]


def resolve_overlaps(note_events):
    """Basic Pitch는 원래 다성(여러 음이 동시에 울리는 것)까지 검출할 수
    있는 모델이라, 배음(하모닉스)이나 포먼트를 잘못 잡아서 실제로는
    하나의 멜로디 음인데 다른 음높이의 '유령 음'이 같은 시간대에 겹쳐서
    나오는 경우가 꽤 있다. 우리는 단선율(보컬 멜로디) 악보만 원하므로,
    시간이 겹치는 음이 나오면 진폭(세기)이 더 큰 쪽만 남기고, 겹치는
    부분은 잘라내거나(약한 음이 더 늦게 시작하면 그 시작을 뒤로 미룸)
    아예 통째로 버려서(약한 음이 강한 음 안에 완전히 파묻히면), 항상 그
    순간에는 최대 한 음만 울리도록 만든다.

    이 처리를 하지 않으면 겹친 음들이 마디에 실제 박자보다 훨씬 많은
    시간이 들어간 것처럼 계산되어(겹치는 두 음이 각각 온전히 카운트되므로)
    마디 길이가 실제 박자의 2배 가까이로 뻥튀기되는 경우가 흔하고, 이런
    악보 파일은 MuseScore 같은 프로그램이 열다가 멎거나 뻗는 원인이 된다."""
    if not note_events:
        return note_events
    events = sorted(note_events, key=lambda e: e[0])
    result = []
    for ev in events:
        start, end, pitch, amp, bend = ev
        if end <= start:
            continue
        while result:
            prev = result[-1]
            if prev[1] <= start:
                break  # 더 이상 겹치지 않음
            if prev[3] >= amp:
                # 이전 음이 더 세다 -> 새 음의 시작을 이전 음이 끝난 뒤로 미룸
                start = prev[1]
                if start >= end:
                    start = end  # 새 음이 완전히 파묻힘 -> 버림
                    break
            else:
                # 새 음이 더 세다 -> 이전 음을 겹치는 부분만큼 줄이거나 통째로 버림
                if prev[0] >= start:
                    result.pop()
                    continue
                else:
                    result[-1] = (prev[0], start, prev[2], prev[3], prev[4])
                break
        if end > start:
            result.append((start, end, pitch, amp, bend))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="보컬(단선율) 오디오를 자동 채보해서 MIDI/MusicXML 악보로 만듭니다."
    )
    parser.add_argument("input", help="채보할 오디오 파일 (보컬만 있는 파일 권장)")
    parser.add_argument("-o", "--outdir", default="transcribed", help="결과 저장 폴더")
    parser.add_argument("--tempo-source", default=None,
                         help="템포(BPM) 추정에 사용할 다른 오디오 파일 (원곡/드럼 등 리듬이 뚜렷한 파일 권장)")
    parser.add_argument("--tempo", type=float, default=None,
                         help="템포(BPM)를 직접 지정 (지정하면 자동 추정/보정을 하지 않음)")
    parser.add_argument("--min-freq", type=float, default=80.0,
                         help="채보에 사용할 최소 주파수(Hz). 사람 목소리 기준 기본값 80Hz")
    parser.add_argument("--max-freq", type=float, default=1200.0,
                         help="채보에 사용할 최대 주파수(Hz). 기본값 1200Hz")
    args = parser.parse_args()

    src = Path(args.input)
    if not src.is_file():
        sys.exit(f"[오류] 파일을 찾을 수 없습니다: {src}")

    try:
        from basic_pitch.inference import predict
    except ImportError as exc:
        sys.exit(
            "[오류] basic-pitch를 import할 수 없습니다.\n"
            f"  실행 중인 파이썬: {sys.executable}\n"
            f"  실제 오류: {exc.__class__.__name__}: {exc}\n"
            "  설치 순서 (둘 다 필요):\n"
            f'    "{sys.executable}" -m pip install -r requirements-transcribe.txt\n'
            f'    "{sys.executable}" -m pip install --no-deps basic-pitch'
        )

    try:
        import onnxruntime  # noqa: F401
    except ImportError as exc:
        sys.exit(
            "[오류] onnxruntime을 import할 수 없습니다 (basic-pitch를 tensorflow 없이 돌리는 데 필요).\n"
            f"  실제 오류: {exc.__class__.__name__}: {exc}\n"
            f'  설치: "{sys.executable}" -m pip install -r requirements-transcribe.txt'
        )

    # basic-pitch에는 TensorFlow SavedModel 외에 ONNX 버전 모델도 이미
    # 들어있다. tensorflow를 설치하지 않았으므로(위 설명 참고) 이 ONNX
    # 파일 경로를 직접 지정해서 ONNX Runtime으로 돌린다.
    import basic_pitch
    onnx_model_path = (
        Path(basic_pitch.__file__).resolve().parent
        / "saved_models" / "icassp_2022" / "nmp.onnx"
    )
    if not onnx_model_path.is_file():
        sys.exit(f"[오류] basic-pitch의 ONNX 모델 파일을 찾을 수 없습니다: {onnx_model_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Basic Pitch로 채보 (음표 목록: 시작/끝 시각(초), 피치, 진폭, 피치벤드)
    print("채보 중... (모델을 처음 쓰면 조금 걸릴 수 있습니다)")
    _, _, note_events = predict(
        str(src),
        str(onnx_model_path),
        minimum_frequency=args.min_freq,
        maximum_frequency=args.max_freq,
        melodia_trick=True,
    )

    if not note_events:
        sys.exit("[오류] 음을 하나도 찾지 못했습니다. 반주가 섞여있지 않은 보컬 파일인지 확인해 주세요.")

    # 2) 쪼개진 음표 합치기
    note_events = merge_close_notes(note_events)

    # 3) 겹치는(다성으로 잘못 잡힌) 음 정리 -> 항상 단선율이 되도록
    note_events = resolve_overlaps(note_events)

    # 4) 박(비트) 그리드 구하기 — 음표들을 "몇 번째 박인지"로 배치하는 기준
    import numpy as np

    if args.tempo is not None:
        tempo_bpm = args.tempo
        tempo_note = "직접 지정 (일정한 템포로 가정)"
        beat_sec = 60.0 / tempo_bpm
        total_end = max(e[1] for e in note_events)
        beat_times = np.arange(0.0, total_end + beat_sec * 2, beat_sec)
    else:
        tempo_source = Path(args.tempo_source) if args.tempo_source else src
        try:
            beat_times, tempo_bpm = get_beat_grid(tempo_source)
        except Exception:  # noqa: BLE001
            beat_times, tempo_bpm = None, None
        if beat_times is None:
            tempo_bpm = 120.0
            tempo_note = "박 추적 실패 -> 기본값 120 BPM 일정한 템포로 가정"
            beat_sec = 60.0 / tempo_bpm
            total_end = max(e[1] for e in note_events)
            beat_times = np.arange(0.0, total_end + beat_sec * 2, beat_sec)
        else:
            tempo_note = f"{tempo_source.name}에서 박 추적"
    print(f"템포: {tempo_bpm:.1f} BPM ({tempo_note})")

    # 5) 초 단위 음표 타이밍을, 실제 박 위치 기준 4분음표 단위로 변환
    quarter_notes = []
    for start, end, pitch, amp, _bend in note_events:
        q_start = time_to_beats(start, beat_times)
        q_end = time_to_beats(end, beat_times)
        if q_end <= q_start:
            continue
        quarter_notes.append((q_start, q_end - q_start, pitch, amp))

    stem = src.stem
    midi_path = outdir / f"{stem}.mid"
    xml_path = outdir / f"{stem}.musicxml"

    # 6) music21 스트림을 (박 기준으로 이미 변환된) 음표로 직접 구성 +
    #    박자 양자화 + 마디 생성 + MIDI/MusicXML 내보내기
    from music21 import stream as m21stream, note as m21note, meter, clef, tempo as m21tempo, metadata

    flat = m21stream.Stream()
    for q_start, q_len, pitch, amp in quarter_notes:
        n = m21note.Note()
        n.pitch.midi = int(round(pitch))
        n.quarterLength = q_len
        n.volume.velocity = max(1, min(127, int(round(amp * 127))))
        flat.insert(q_start, n)

    flat.insert(0, clef.TrebleClef())
    flat.insert(0, meter.TimeSignature("4/4"))
    flat.insert(0, m21tempo.MetronomeMark(number=round(tempo_bpm)))

    quantized = flat.quantize((4, 3), processOffsets=True, processDurations=True, inPlace=False)
    quantized = quantized.makeMeasures(inPlace=False)
    quantized.makeTies(inPlace=True)

    try:
        key_est = quantized.analyze("key")
        from music21 import key as m21key
        quantized.insert(0, m21key.Key(key_est.tonic.name, key_est.mode))
    except Exception:  # noqa: BLE001
        pass  # 조성 추정 실패해도 악보 자체는 문제 없음

    quantized.metadata = metadata.Metadata()
    quantized.metadata.title = stem

    quantized.write("midi", fp=str(midi_path))
    quantized.write("musicxml", fp=str(xml_path))

    print("\n[완료]")
    print(" - MIDI:", midi_path)
    print(" - MusicXML:", xml_path)
    print(f" - 음표 개수: {len(quarter_notes)}")


if __name__ == "__main__":
    main()
