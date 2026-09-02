#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py
음원 분리(Demucs) 결과를 웹 브라우저에서 업로드/진행률 확인/재생/다운로드까지
할 수 있게 해주는 로컬 웹앱.

내 컴퓨터에서만 켜서 쓰는 용도입니다 (인터넷에 공개되는 서버 아님).

사전 준비:
    pip install -r requirements.txt
    (flask, demucs, numpy 가 설치됩니다. ffmpeg도 시스템에 설치되어 있어야 합니다.)

실행:
    python app.py

    실행 후 브라우저에서 http://127.0.0.1:5000 접속

동작 방식:
    - 업로드한 파일(mp3/mp4/wav 등)은 uploads/<작업ID>/ 에 저장됩니다.
    - 분리 결과는 separated/<작업ID>/ 에 저장됩니다.
    - 한글 파일명 때문에 생기는 인코딩 문제를 피하려고, 실제 디스크에는
      작업ID 기반 이름으로 저장하고, 화면에는 원래 파일명을 보여줍니다.
    - 작업은 한 번에 하나씩만 순서대로 처리합니다 (동시에 여러 개 돌리면
      CPU/GPU가 감당하기 힘들어서 큐로 순서를 관리합니다).
    - 작업 목록은 jobs.json 파일에 저장되어, 앱을 껐다 켜도 히스토리가 남습니다.
"""

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory, abort

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
SEPARATED_DIR = BASE_DIR / "separated"
JOBS_FILE = BASE_DIR / "jobs.json"
BACKING_SCRIPT = BASE_DIR / "split_backing_vocal.py"
TRANSCRIBE_SCRIPT = BASE_DIR / "transcribe_vocal.py"

UPLOAD_DIR.mkdir(exist_ok=True)
SEPARATED_DIR.mkdir(exist_ok=True)

ALLOWED_EXTS = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac",
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".wmv",
}

# 화면에 노출할 분리 모드 옵션
# 참고: htdemucs_6s(6-스템)의 피아노/기타 분리는 Demucs 자체의 알려진 한계로
# 학습 데이터가 적어 정확도가 낮은 경우가 많다 (Meta 공식 문서에도 명시됨).
# 완전히 해결할 방법은 없어서, 대신 일반 악기 분리 품질이 더 좋은 htdemucs_ft
# 옵션을 추가로 제공한다.
MODE_OPTIONS = {
    "4stem": {"label": "4-스템 (보컬/드럼/베이스/그 외 악기)", "engine": "demucs", "model": "htdemucs", "two_stems": None},
    "4stem_ft": {"label": "4-스템 고품질 (htdemucs_ft, 약 4배 느림)", "engine": "demucs", "model": "htdemucs_ft", "two_stems": None},
    "6stem": {"label": "6-스템 (+기타/피아노 — 피아노는 부정확할 수 있음)", "engine": "demucs", "model": "htdemucs_6s", "two_stems": None},
    "vocals": {"label": "보컬만 분리 (반주 제거, 가장 빠름)", "engine": "demucs", "model": "htdemucs", "two_stems": "vocals"},
    # 아래 두 개는 일반 분리와 달리, "이미 보컬만 있는 파일"을 넣었을 때
    # 그 안에서 리드 보컬과 백코러스/하모니를 나누는 용도입니다.
    # (UVR 커뮤니티의 "가라오케" 계열 모델 재활용 — split_backing_vocal.py 참고)
    "backing_fast": {
        "label": "백코러스/하모니 분리 - 빠름 (이미 보컬만 있는 파일 전용)",
        "engine": "backing", "model": "UVR_MDXNET_KARA_2.onnx",
    },
    "backing_quality": {
        "label": "백코러스/하모니 분리 - 고품질/느림 (이미 보컬만 있는 파일 전용)",
        "engine": "backing", "model": "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
    },
    # 보컬(단선율) 오디오를 자동 채보해서 MIDI/MusicXML 악보로 만드는 모드.
    # (Basic Pitch + music21 — transcribe_vocal.py 참고. 화음/여러 악기가
    # 섞인 파일은 정확도가 크게 떨어지니 보컬만 있는 파일에 쓰는 걸 권장)
    "transcribe_vocal": {
        "label": "보컬 멜로디 악보로 만들기 (자동 채보, 이미 보컬만 있는 파일 전용)",
        "engine": "transcribe",
    },
}

STEM_LABELS = {
    "vocals": "보컬", "no_vocals": "반주(보컬 제외)",
    "drums": "드럼", "bass": "베이스",
    "guitar": "기타", "piano": "피아노", "other": "그 외 악기",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 최대 2GB 업로드

jobs = {}
jobs_lock = threading.Lock()
job_queue = queue.Queue()


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def save_jobs():
    with jobs_lock:
        data = list(jobs.values())
    with open(JOBS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_jobs():
    if not JOBS_FILE.exists():
        return
    try:
        with open(JOBS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return
    for job in data:
        # 이전 실행 중 서버가 꺼져서 중단된 작업은 에러 처리
        if job.get("status") in ("queued", "processing"):
            job["status"] = "error"
            job["error"] = "서버가 재시작되어 작업이 중단되었습니다. 다시 업로드해 주세요."
        jobs[job["id"]] = job


def update_job(job_id, **fields):
    with jobs_lock:
        jobs[job_id].update(fields)
    save_jobs()


PROGRESS_RE = re.compile(r"(\d{1,3})%\|")


def run_demucs(job_id):
    with jobs_lock:
        job = dict(jobs[job_id])

    update_job(job_id, status="processing", progress=0, started_at=now_iso())

    mode = MODE_OPTIONS[job["mode"]]
    model = mode["model"]
    src_path = UPLOAD_DIR / job_id / job["stored_name"]
    out_root = SEPARATED_DIR / job_id

    cmd = [sys.executable, "-m", "demucs", "-n", model, "-o", str(out_root)]
    if mode["two_stems"]:
        cmd += ["--two-stems", mode["two_stems"]]
    if job.get("output_format") == "mp3":
        cmd += ["--mp3", "--mp3-bitrate", "320"]
    cmd += [str(src_path)]

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    log_lines = []
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            env=env, bufsize=1,
        )
        for line in proc.stdout:
            log_lines.append(line)
            m = None
            for m in PROGRESS_RE.finditer(line):
                pass
            if m:
                pct = min(99, int(m.group(1)))  # 100%는 실제 완료 확인 후에만 표시
                update_job(job_id, progress=pct)
        proc.wait()
    except Exception as exc:  # noqa: BLE001
        update_job(job_id, status="error", error=str(exc), finished_at=now_iso())
        return

    if proc.returncode != 0:
        tail = "".join(log_lines[-40:])
        update_job(job_id, status="error", error=tail, finished_at=now_iso())
        return

    # 결과 파일 찾기: separated/<job_id>/<model>/<원본파일이름_stem>/*.{wav,mp3}
    track_dirname = src_path.stem
    stem_dir = out_root / model / track_dirname
    if not stem_dir.is_dir():
        update_job(job_id, status="error",
                    error=f"결과 폴더를 찾을 수 없습니다: {stem_dir}", finished_at=now_iso())
        return

    stems = []
    for f in sorted(stem_dir.iterdir()):
        if f.suffix.lower() not in (".wav", ".mp3"):
            continue
        key = f.stem
        stems.append({
            "key": key,
            "label": STEM_LABELS.get(key, key),
            "filename": f.name,
        })

    rel_stem_dir = str(stem_dir.relative_to(SEPARATED_DIR))
    update_job(
        job_id,
        status="done",
        progress=100,
        stems=stems,
        stem_dir=rel_stem_dir,
        finished_at=now_iso(),
    )


def run_backing_split(job_id):
    """이미 보컬만 있는 파일에서 리드 보컬 / 백코러스(하모니)를 나눈다.
    (demucs가 아니라 split_backing_vocal.py를 서브프로세스로 돌린다.)"""
    with jobs_lock:
        job = dict(jobs[job_id])

    update_job(job_id, status="processing", progress=0, started_at=now_iso())

    mode = MODE_OPTIONS[job["mode"]]
    model_filename = mode["model"]
    src_path = UPLOAD_DIR / job_id / job["stored_name"]
    out_dir = SEPARATED_DIR / job_id / "backing"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(BACKING_SCRIPT), str(src_path),
        "-o", str(out_dir), "--model", model_filename,
    ]
    if job.get("output_format") == "mp3":
        cmd.append("--mp3")

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    log_lines = []
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            env=env, bufsize=1,
        )
        for line in proc.stdout:
            log_lines.append(line)
            m = None
            for m in PROGRESS_RE.finditer(line):
                pass
            if m:
                pct = min(99, int(m.group(1)))
                update_job(job_id, progress=pct)
        proc.wait()
    except Exception as exc:  # noqa: BLE001
        update_job(job_id, status="error", error=str(exc), finished_at=now_iso())
        return

    if proc.returncode != 0:
        tail = "".join(log_lines[-40:])
        update_job(job_id, status="error", error=tail, finished_at=now_iso())
        return

    stems = []
    for f in sorted(out_dir.iterdir()):
        if f.suffix.lower() not in (".wav", ".mp3"):
            continue
        if "(Vocals)" in f.name:
            key, label = "lead_vocal", "리드 보컬"
        elif "(Instrumental)" in f.name:
            key, label = "backing_vocal", "백코러스 / 하모니"
        else:
            key, label = f.stem, f.stem
        stems.append({"key": key, "label": label, "filename": f.name})

    if not stems:
        update_job(job_id, status="error",
                    error=f"결과 파일을 찾을 수 없습니다: {out_dir}", finished_at=now_iso())
        return

    rel_stem_dir = str(out_dir.relative_to(SEPARATED_DIR))
    update_job(
        job_id,
        status="done",
        progress=100,
        stems=stems,
        stem_dir=rel_stem_dir,
        finished_at=now_iso(),
    )


def run_transcribe(job_id):
    """보컬(단선율) 오디오를 자동 채보해서 MIDI/MusicXML 악보를 만든다.
    (transcribe_vocal.py를 서브프로세스로 돌린다. Basic Pitch + music21 사용)"""
    with jobs_lock:
        job = dict(jobs[job_id])

    update_job(job_id, status="processing", progress=10, started_at=now_iso())

    src_path = UPLOAD_DIR / job_id / job["stored_name"]
    out_dir = SEPARATED_DIR / job_id / "transcribe"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(TRANSCRIBE_SCRIPT), str(src_path), "-o", str(out_dir)]
    tempo_ref_name = job.get("tempo_ref_stored_name")
    if tempo_ref_name:
        tempo_ref_path = UPLOAD_DIR / job_id / tempo_ref_name
        if tempo_ref_path.is_file():
            cmd += ["--tempo-source", str(tempo_ref_path)]

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    log_lines = []
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            env=env, bufsize=1,
        )
        for line in proc.stdout:
            log_lines.append(line)
        proc.wait()
    except Exception as exc:  # noqa: BLE001
        update_job(job_id, status="error", error=str(exc), finished_at=now_iso())
        return

    if proc.returncode != 0:
        tail = "".join(log_lines[-40:])
        update_job(job_id, status="error", error=tail, finished_at=now_iso())
        return

    stem = Path(job["stored_name"]).stem
    midi_file = out_dir / f"{stem}.mid"
    xml_file = out_dir / f"{stem}.musicxml"
    if not midi_file.is_file() or not xml_file.is_file():
        update_job(job_id, status="error",
                    error=f"결과 파일을 찾을 수 없습니다: {out_dir}", finished_at=now_iso())
        return

    rel_stem_dir = str(out_dir.relative_to(SEPARATED_DIR))
    update_job(
        job_id,
        status="done",
        progress=100,
        notation={"midi_filename": midi_file.name, "musicxml_filename": xml_file.name},
        stem_dir=rel_stem_dir,
        finished_at=now_iso(),
    )


def run_job(job_id):
    with jobs_lock:
        job = dict(jobs[job_id])
    engine = MODE_OPTIONS[job["mode"]].get("engine", "demucs")
    if engine == "backing":
        run_backing_split(job_id)
    elif engine == "transcribe":
        run_transcribe(job_id)
    else:
        run_demucs(job_id)


def worker_loop():
    while True:
        job_id = job_queue.get()
        try:
            run_job(job_id)
        except Exception as exc:  # noqa: BLE001
            update_job(job_id, status="error", error=str(exc), finished_at=now_iso())
        finally:
            job_queue.task_done()


@app.route("/")
def index():
    return render_template(
        "index.html",
        mode_options=MODE_OPTIONS,
        mode_options_json=json.dumps(MODE_OPTIONS, ensure_ascii=False),
    )


@app.route("/api/jobs")
def api_jobs():
    with jobs_lock:
        data = sorted(jobs.values(), key=lambda j: j["created_at"], reverse=True)
    return jsonify(data)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    file = request.files.get("file")
    mode = request.form.get("mode", "4stem")
    output_format = request.form.get("output_format", "mp3")

    if file is None or file.filename == "":
        return jsonify({"error": "파일이 선택되지 않았습니다."}), 400
    if mode not in MODE_OPTIONS:
        return jsonify({"error": "알 수 없는 분리 모드입니다."}), 400

    original_name = file.filename
    ext = Path(original_name).suffix.lower()
    if ext not in ALLOWED_EXTS:
        return jsonify({"error": f"지원하지 않는 확장자입니다: {ext}"}), 400

    job_id = uuid.uuid4().hex[:12]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # 한글/특수문자 파일명이 ffmpeg·데뮤스·Windows 경로 처리에서 인코딩 문제를
    # 일으키지 않도록, 디스크에는 job_id 기반의 안전한 이름으로 저장한다.
    stored_name = f"source{ext}"
    file.save(job_dir / stored_name)

    job = {
        "id": job_id,
        "original_name": original_name,
        "stored_name": stored_name,
        "tempo_ref_stored_name": None,
        "mode": mode,
        "mode_label": MODE_OPTIONS[mode]["label"],
        "output_format": output_format,
        "status": "queued",
        "progress": 0,
        "created_at": now_iso(),
        "started_at": None,
        "finished_at": None,
        "stems": [],
        "notation": None,
        "stem_dir": None,
        "error": None,
    }
    with jobs_lock:
        jobs[job_id] = job
    save_jobs()

    job_queue.put(job_id)
    return jsonify(job)


@app.route("/api/split_backing/<job_id>/<path:filename>", methods=["POST"])
def api_split_backing(job_id, filename):
    """완료된 작업의 '보컬' 스템 파일을 다시 업로드할 필요 없이, 그 자리에서
    바로 리드 보컬/백코러스 분리 작업(새 job)을 큐에 넣는다."""
    with jobs_lock:
        src_job = jobs.get(job_id)
    if not src_job or not src_job.get("stem_dir"):
        return jsonify({"error": "원본 작업을 찾을 수 없습니다."}), 404

    src_file = SEPARATED_DIR / src_job["stem_dir"] / filename
    try:
        src_file.resolve().relative_to(SEPARATED_DIR.resolve())
    except ValueError:
        abort(404)
    if not src_file.is_file():
        return jsonify({"error": "원본 파일을 찾을 수 없습니다."}), 404

    preset = request.form.get("preset", "backing_fast")
    if preset not in MODE_OPTIONS or MODE_OPTIONS[preset].get("engine") != "backing":
        return jsonify({"error": "알 수 없는 옵션입니다."}), 400

    new_job_id = uuid.uuid4().hex[:12]
    new_job_dir = UPLOAD_DIR / new_job_id
    new_job_dir.mkdir(parents=True, exist_ok=True)
    ext = src_file.suffix.lower()
    stored_name = f"source{ext}"
    shutil.copy(src_file, new_job_dir / stored_name)

    job = {
        "id": new_job_id,
        "original_name": f"{src_job.get('original_name', '보컬')} → 백코러스 분리",
        "stored_name": stored_name,
        "tempo_ref_stored_name": None,
        "mode": preset,
        "mode_label": MODE_OPTIONS[preset]["label"],
        "output_format": src_job.get("output_format", "mp3"),
        "status": "queued",
        "progress": 0,
        "created_at": now_iso(),
        "started_at": None,
        "finished_at": None,
        "stems": [],
        "notation": None,
        "stem_dir": None,
        "error": None,
    }
    with jobs_lock:
        jobs[new_job_id] = job
    save_jobs()

    job_queue.put(new_job_id)
    return jsonify(job)


@app.route("/api/transcribe/<job_id>/<path:filename>", methods=["POST"])
def api_transcribe(job_id, filename):
    """완료된 작업의 보컬 스템을 다시 업로드할 필요 없이, 그 자리에서 바로
    자동 채보(악보 만들기) 작업(새 job)을 큐에 넣는다. 템포 추정 품질을
    높이려고, 원래 그 job에 업로드됐던 원본 파일도 같이 복사해서 넘긴다."""
    with jobs_lock:
        src_job = jobs.get(job_id)
    if not src_job or not src_job.get("stem_dir"):
        return jsonify({"error": "원본 작업을 찾을 수 없습니다."}), 404

    src_file = SEPARATED_DIR / src_job["stem_dir"] / filename
    try:
        src_file.resolve().relative_to(SEPARATED_DIR.resolve())
    except ValueError:
        abort(404)
    if not src_file.is_file():
        return jsonify({"error": "원본 파일을 찾을 수 없습니다."}), 404

    new_job_id = uuid.uuid4().hex[:12]
    new_job_dir = UPLOAD_DIR / new_job_id
    new_job_dir.mkdir(parents=True, exist_ok=True)
    ext = src_file.suffix.lower()
    stored_name = f"source{ext}"
    shutil.copy(src_file, new_job_dir / stored_name)

    # 템포 추정용 원본(그 스템을 뽑아낸 원래 job에 업로드됐던 파일)도 있으면 복사.
    # (드럼/반주가 있는 원본일수록 템포 추정이 잘 됨. 없거나 이미 보컬만
    # 있던 파일이면 어쩔 수 없이 그것만으로 추정한다.)
    tempo_ref_stored_name = None
    orig_upload = UPLOAD_DIR / job_id / src_job.get("stored_name", "")
    if orig_upload.is_file():
        tempo_ref_stored_name = f"tempo_ref{orig_upload.suffix.lower()}"
        shutil.copy(orig_upload, new_job_dir / tempo_ref_stored_name)

    job = {
        "id": new_job_id,
        "original_name": f"{src_job.get('original_name', '보컬')} → 악보",
        "stored_name": stored_name,
        "tempo_ref_stored_name": tempo_ref_stored_name,
        "mode": "transcribe_vocal",
        "mode_label": MODE_OPTIONS["transcribe_vocal"]["label"],
        "output_format": src_job.get("output_format", "mp3"),
        "status": "queued",
        "progress": 0,
        "created_at": now_iso(),
        "started_at": None,
        "finished_at": None,
        "stems": [],
        "notation": None,
        "stem_dir": None,
        "error": None,
    }
    with jobs_lock:
        jobs[new_job_id] = job
    save_jobs()

    job_queue.put(new_job_id)
    return jsonify(job)


AUDIO_MIMETYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".mid": "audio/midi",
    ".midi": "audio/midi",
    ".musicxml": "application/vnd.recordare.musicxml+xml",
    ".xml": "application/vnd.recordare.musicxml+xml",
}


@app.route("/files/<job_id>/<path:filename>")
def files(job_id, filename):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or not job.get("stem_dir"):
        abort(404)
    directory = SEPARATED_DIR / job["stem_dir"]
    # Windows에서는 시스템 레지스트리 기반으로 확장자별 MIME 타입을 추측하는데,
    # .mp3/.wav 매핑이 없거나 잘못되어 있어 브라우저가 재생을 거부하는 경우가
    # 있다. 그래서 확장자를 보고 직접 지정해준다.
    ext = Path(filename).suffix.lower()
    mimetype = AUDIO_MIMETYPES.get(ext)
    # conditional=True(기본값)라 Range 요청(탐색/구간 재생)도 정상 지원된다.
    return send_from_directory(directory, filename, mimetype=mimetype)


if __name__ == "__main__":
    load_jobs()
    threading.Thread(target=worker_loop, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)