import base64
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import runpod
from PIL import Image

APP_ROOT = Path(__file__).resolve().parent
os.chdir(APP_ROOT)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


def _temp_path(suffix):
    handle, name = tempfile.mkstemp(suffix=suffix)
    os.close(handle)
    return Path(name)


def _decode_image(payload, label):
    if not payload:
        raise ValueError(f"{label} is required.")
    raw = str(payload).split(",", 1)[-1]
    data = base64.b64decode(raw)
    if not data:
        raise ValueError(f"{label} was empty.")
    path = _temp_path(".png")
    path.write_bytes(data)
    with Image.open(path) as image:
        image.convert("RGB").save(path)
    return path


def _decode_video(payload):
    if not payload:
        raise ValueError("A raw video is required.")
    raw = str(payload).split(",", 1)[-1]
    data = base64.b64decode(raw)
    if len(data) < 32:
        raise ValueError("The raw video was empty.")
    path = _temp_path(".mp4")
    path.write_bytes(data)
    return path


def _encode_video(path):
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _clamp_int(value, low, high, fallback):
    try:
        number = int(value)
    except (TypeError, ValueError):
        return fallback
    return min(high, max(low, number))


def _clamp_float(value, low, high, fallback):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return min(high, max(low, number))


def _interpolate(job_input):
    from engine import interpolate

    start = _decode_image(job_input.get("image1_base64"), "Start image")
    end = _decode_image(job_input.get("image2_base64"), "End image")
    prompt = str(job_input.get("prompt") or "").strip()[:300] or "cel-shaded character"
    dest = _temp_path(".mp4")
    try:
        interpolate(
            start,
            end,
            prompt,
            steps=_clamp_int(job_input.get("steps"), 1, 60, 50),
            cfg_scale=_clamp_float(job_input.get("cfg_scale"), 1, 15, 7.5),
            eta=_clamp_float(job_input.get("eta"), 0, 1, 1),
            frame_stride=_clamp_int(job_input.get("frame_stride"), 5, 30, 10),
            seed=_clamp_int(job_input.get("seed"), 0, 2147483647, 123),
            dest=str(dest),
        )
        return {"video_base64": _encode_video(dest)}
    finally:
        start.unlink(missing_ok=True)
        end.unlink(missing_ok=True)
        dest.unlink(missing_ok=True)


def _upscale(job_input):
    from engine import ensure_upscaler, release_model

    try:
        scale = int(job_input.get("scale"))
    except (TypeError, ValueError):
        scale = 0
    if scale not in (2, 4):
        raise ValueError("Upscale scale must be 2 or 4.")
    release_model()
    ensure_upscaler()
    source = _decode_video(job_input.get("video_base64"))
    out_dir = Path(tempfile.mkdtemp(prefix="upscale-"))
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "/opt/Real-ESRGAN/inference_realesrgan_video.py",
                "-i",
                str(source),
                "-o",
                str(out_dir),
                "-n",
                "realesr-animevideov3",
                "-s",
                str(scale),
                "--suffix",
                "upscaled",
            ],
            cwd="/opt/Real-ESRGAN",
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "Upscale failed.").strip()
            raise RuntimeError(detail[-2000:])
        matches = sorted(out_dir.rglob("*.mp4"))
        if not matches:
            raise RuntimeError("Upscale finished without a video.")
        return {"video_base64": _encode_video(matches[0])}
    finally:
        source.unlink(missing_ok=True)
        shutil.rmtree(out_dir, ignore_errors=True)


def handler(job):
    job_input = job.get("input") or {}
    task = job_input.get("task")
    if task == "interpolate":
        return _interpolate(job_input)
    if task == "upscale":
        return _upscale(job_input)
    raise ValueError("Task must be interpolate or upscale.")


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
