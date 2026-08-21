from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image


ProgressCallback = Callable[[float, str], None]
DEFAULT_HAND_PATH = Path(__file__).parent / "assets" / "default_hand.png"


@dataclass(frozen=True)
class RenderConfig:
    sensitivity: int = 12
    minimum_change_area: int = 6
    smoothing: float = 0.42
    hide_after_frames: int = 18
    maximum_jump: int = 220
    hand_width_percent: float = 34.0
    hand_opacity: float = 1.0
    hand_side: str = "Right"
    tip_x_percent: float = 14.8
    tip_y_percent: float = 34.4
    roi_left_percent: float = 0.0
    roi_top_percent: float = 0.0
    roi_right_percent: float = 100.0
    roi_bottom_percent: float = 100.0
    audio_gain: float = 96.0


@dataclass(frozen=True)
class RenderResult:
    frames: int
    tracked_frames: int
    fps: float
    width: int
    height: int
    audio_preserved: bool


def load_default_hand() -> np.ndarray:
    """Load the exact bundled transparent hand illustration as RGBA."""
    with Image.open(DEFAULT_HAND_PATH) as image:
        return np.asarray(image.convert("RGBA"))


def make_default_hand(width: int | None = None) -> np.ndarray:
    """Compatibility helper used by the test suite."""
    hand = load_default_hand()
    if width is None or width <= 0 or hand.shape[1] == width:
        return hand
    target_height = max(1, round(hand.shape[0] * width / hand.shape[1]))
    return _resize_rgba_premultiplied(hand, width, target_height)


def load_hand_image(data: bytes | None) -> np.ndarray:
    if not data:
        return load_default_hand()
    with Image.open(BytesIO(data)) as image:
        return np.asarray(image.convert("RGBA"))


def _resize_rgba_premultiplied(sprite: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize RGBA without creating light/dark halos around transparent edges."""
    rgba = sprite.astype(np.float32)
    alpha = rgba[:, :, 3:4] / 255.0
    premultiplied = rgba[:, :, :3] * alpha
    scale = width / max(1, sprite.shape[1])
    interpolation = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_CUBIC

    resized_alpha = cv2.resize(alpha, (width, height), interpolation=interpolation)
    if resized_alpha.ndim == 2:
        resized_alpha = resized_alpha[:, :, None]
    resized_rgb = cv2.resize(premultiplied, (width, height), interpolation=interpolation)

    unpremultiplied = np.zeros_like(resized_rgb)
    np.divide(
        resized_rgb,
        resized_alpha,
        out=unpremultiplied,
        where=resized_alpha > 1e-6,
    )
    result = np.concatenate(
        [np.clip(unpremultiplied, 0, 255), np.clip(resized_alpha * 255.0, 0, 255)],
        axis=2,
    )
    return result.astype(np.uint8)


def _prepare_sprite(sprite: np.ndarray, frame_width: int, config: RenderConfig) -> tuple[np.ndarray, int, int]:
    target_width = max(60, int(frame_width * config.hand_width_percent / 100.0))
    ratio = target_width / sprite.shape[1]
    target_height = max(1, int(sprite.shape[0] * ratio))
    resized = _resize_rgba_premultiplied(sprite, target_width, target_height)

    tip_x = int(target_width * config.tip_x_percent / 100.0)
    tip_y = int(target_height * config.tip_y_percent / 100.0)
    if config.hand_side.lower() == "left":
        resized = cv2.flip(resized, 1)
        tip_x = target_width - 1 - tip_x
    return resized, tip_x, tip_y


def _overlay_rgba(frame: np.ndarray, sprite: np.ndarray, anchor: tuple[float, float], tip: tuple[int, int], opacity: float) -> None:
    x = int(anchor[0] - tip[0])
    y = int(anchor[1] - tip[1])
    sh, sw = sprite.shape[:2]
    fh, fw = frame.shape[:2]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw, x + sw), min(fh, y + sh)
    if x1 >= x2 or y1 >= y2:
        return

    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
    crop = sprite[sy1:sy2, sx1:sx2]
    alpha = (crop[:, :, 3:4].astype(np.float32) / 255.0) * float(np.clip(opacity, 0.0, 1.0))
    rgb = crop[:, :, :3][:, :, ::-1].astype(np.float32)
    base = frame[y1:y2, x1:x2].astype(np.float32)
    frame[y1:y2, x1:x2] = np.clip(rgb * alpha + base * (1.0 - alpha), 0, 255).astype(np.uint8)


def _candidate_from_change(mask: np.ndarray, previous: tuple[float, float] | None, maximum_jump: float) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    points = np.column_stack((xs, ys)).astype(np.float32)

    if previous is None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return float(np.median(xs)), float(np.median(ys))
        contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(contour)
        if moments["m00"]:
            return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
        x, y, w, h = cv2.boundingRect(contour)
        return x + w / 2.0, y + h / 2.0

    origin = np.asarray(previous, dtype=np.float32)
    distances = np.linalg.norm(points - origin, axis=1)
    nearby = distances <= maximum_jump
    if np.count_nonzero(nearby) >= 3:
        points = points[nearby]
        distances = distances[nearby]

    cutoff = np.percentile(distances, 88)
    leading = points[distances >= cutoff]
    return float(np.median(leading[:, 0])), float(np.median(leading[:, 1]))


def _candidate_from_frames(
    gray: np.ndarray,
    previous_gray: np.ndarray,
    previous_local: tuple[float, float] | None,
    config: RenderConfig,
    tracking_scale: float,
) -> tuple[float, float] | None:
    difference = cv2.absdiff(gray, previous_gray)
    attempts = (
        (config.sensitivity, config.minimum_change_area),
        (max(3, config.sensitivity // 3), max(2, config.minimum_change_area // 3)),
    )

    for threshold, minimum_area in attempts:
        _, mask = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        if int(cv2.countNonZero(mask)) < minimum_area:
            continue
        candidate = _candidate_from_change(mask, previous_local, config.maximum_jump * tracking_scale)
        if candidate is not None:
            return candidate
    return None


def _source_has_audio(ffmpeg: str, original_video: Path) -> bool:
    probe = subprocess.run(
        [ffmpeg, "-i", str(original_video)], capture_output=True, check=False, text=True
    )
    return "Audio:" in probe.stderr


def _mux_original_audio(
    silent_video: Path,
    original_video: Path,
    output_video: Path,
    audio_gain: float,
) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not _source_has_audio(ffmpeg, original_video):
        shutil.move(str(silent_video), str(output_video))
        return False

    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(silent_video),
        "-i",
        str(original_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
    ]
    if abs(audio_gain - 1.0) > 1e-9:
        command += ["-filter:a", f"volume={audio_gain:g}"]
    command += ["-c:a", "aac", "-b:a", "256k", "-shortest", str(output_video)]

    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode == 0 and output_video.exists() and output_video.stat().st_size:
        silent_video.unlink(missing_ok=True)
        return True

    shutil.move(str(silent_video), str(output_video))
    return False


def render_hand_video(
    input_path: str | Path,
    output_path: str | Path,
    hand_rgba: np.ndarray,
    config: RenderConfig,
    progress: ProgressCallback | None = None,
) -> RenderResult:
    input_path = Path(input_path)
    output_path = Path(output_path)
    silent_path = output_path.with_name(output_path.stem + "-silent.mp4")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError("The uploaded video could not be opened.")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width < 2 or height < 2:
        capture.release()
        raise ValueError("The uploaded video has an invalid frame size.")

    writer = cv2.VideoWriter(str(silent_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("The output video encoder could not be started.")

    left = int(width * config.roi_left_percent / 100.0)
    right = int(width * config.roi_right_percent / 100.0)
    top = int(height * config.roi_top_percent / 100.0)
    bottom = int(height * config.roi_bottom_percent / 100.0)
    if right - left < 20 or bottom - top < 20:
        capture.release()
        writer.release()
        raise ValueError("The selected drawing area is too small.")

    roi_width = right - left
    roi_height = bottom - top
    tracking_scale = min(1.0, 720.0 / roi_width)
    track_size = (max(1, int(roi_width * tracking_scale)), max(1, int(roi_height * tracking_scale)))
    sprite, tip_x, tip_y = _prepare_sprite(hand_rgba, width, config)

    previous_gray: np.ndarray | None = None
    tracked: tuple[float, float] | None = None
    idle_frames = config.hide_after_frames + 1
    written = 0
    active_count = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            roi = frame[top:bottom, left:right]
            small = cv2.resize(roi, track_size, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

            candidate = None
            if previous_gray is not None:
                previous_local = None
                if tracked is not None:
                    previous_local = (
                        (tracked[0] - left) * tracking_scale,
                        (tracked[1] - top) * tracking_scale,
                    )
                local = _candidate_from_frames(
                    gray,
                    previous_gray,
                    previous_local,
                    config,
                    tracking_scale,
                )
                if local is not None:
                    candidate = (local[0] / tracking_scale + left, local[1] / tracking_scale + top)

            previous_gray = gray
            if candidate is not None:
                if tracked is None:
                    tracked = candidate
                else:
                    a = float(np.clip(config.smoothing, 0.05, 1.0))
                    tracked = (
                        tracked[0] * (1.0 - a) + candidate[0] * a,
                        tracked[1] * (1.0 - a) + candidate[1] * a,
                    )
                idle_frames = 0
                active_count += 1
            else:
                idle_frames += 1

            if tracked is not None and idle_frames <= config.hide_after_frames:
                fade = 1.0 - idle_frames / max(1, config.hide_after_frames + 1)
                _overlay_rgba(frame, sprite, tracked, (tip_x, tip_y), config.hand_opacity * fade)

            writer.write(frame)
            written += 1
            if progress and (written == 1 or written % max(1, int(fps / 2)) == 0):
                fraction = written / frame_count if frame_count > 0 else 0.0
                progress(min(.97, fraction), f"Processing frame {written:,}")
    finally:
        capture.release()
        writer.release()

    if written == 0:
        silent_path.unlink(missing_ok=True)
        raise ValueError("No readable frames were found in the uploaded video.")

    if progress:
        progress(.98, "Restoring the original audio")
    audio_preserved = _mux_original_audio(silent_path, input_path, output_path, config.audio_gain)
    if progress:
        progress(1.0, "Finished")

    return RenderResult(written, active_count, fps, width, height, audio_preserved)
