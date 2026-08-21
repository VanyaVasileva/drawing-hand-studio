from __future__ import annotations

from collections import deque
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image


ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class RenderConfig:
    sensitivity: int = 24
    minimum_change_area: int = 18
    smoothing: float = 0.42
    hide_after_frames: int = 12
    maximum_jump: int = 220
    hand_width_percent: float = 50.0
    hand_opacity: float = 0.94
    hand_side: str = "Right"
    tip_x_percent: float = 14.8
    tip_y_percent: float = 34.4
    roi_left_percent: float = 0.0
    roi_top_percent: float = 0.0
    roi_right_percent: float = 100.0
    roi_bottom_percent: float = 100.0


@dataclass(frozen=True)
class RenderResult:
    frames: int
    tracked_frames: int
    fps: float
    width: int
    height: int
    audio_preserved: bool


DEFAULT_HAND_PATH = Path(__file__).parent / "assets" / "default_hand.png"


def load_default_hand() -> np.ndarray:
    image = Image.open(DEFAULT_HAND_PATH).convert("RGBA")
    return np.asarray(image)


def make_default_hand(width: int | None = None) -> np.ndarray:
    hand = load_default_hand()
    if width is None or width <= 0 or hand.shape[1] == width:
        return hand
    target_height = max(1, round(hand.shape[0] * width / hand.shape[1]))
    return cv2.resize(hand, (width, target_height), interpolation=cv2.INTER_AREA)


def load_hand_image(data: bytes | None) -> np.ndarray:
    if not data:
        return load_default_hand()
    from io import BytesIO

    image = Image.open(BytesIO(data)).convert("RGBA")
    return np.asarray(image)


def _prepare_sprite(sprite: np.ndarray, frame_width: int, config: RenderConfig) -> tuple[np.ndarray, int, int]:
    target_width = max(60, int(frame_width * config.hand_width_percent / 100.0))
    ratio = target_width / sprite.shape[1]
    target_height = max(1, int(sprite.shape[0] * ratio))
    resized = cv2.resize(sprite, (target_width, target_height), interpolation=cv2.INTER_AREA)
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


def _tracking_frame(small_bgr: np.ndarray) -> np.ndarray:
    """Capture the actual paper texture once, instead of assuming white paper."""
    softened = cv2.GaussianBlur(small_bgr, (3, 3), 0)
    return cv2.cvtColor(softened, cv2.COLOR_BGR2LAB).astype(np.float32)


def _background_delta(current: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Per-pixel perceptual change from the original canvas appearance."""
    difference = np.abs(current - baseline)
    return np.sqrt((difference[:, :, 0] * 0.60) ** 2 + difference[:, :, 1] ** 2 + difference[:, :, 2] ** 2)


def _new_ink_mask(current_delta: np.ndarray, old_delta: np.ndarray, config: RenderConfig) -> np.ndarray:
    """Return only pixels that became more different from the captured canvas recently.

    Because both maps are measured against the same initial canvas, the original
    beige/white color and the paper texture cancel out. Looking several frames
    back also accumulates very faint Procreate brush buildup that consecutive
    frame differencing can miss.
    """
    novelty = np.maximum(current_delta - old_delta, 0.0)
    novelty = cv2.GaussianBlur(novelty, (3, 3), 0)

    threshold = max(1.15, float(config.sensitivity) / 16.0)
    mask = (novelty >= threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask


def _is_large_ui_change(background_delta: np.ndarray) -> bool:
    """Detect large rectangular overlays such as Procreate brush/settings panels."""
    mask = (background_delta >= 3.0).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    frame_area = mask.shape[0] * mask.shape[1]
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        box_area = max(1, int(w) * int(h))
        fill = float(area) / box_area
        if area >= frame_area * 0.20 and fill >= 0.45:
            return True
    return False


def _bbox_distance(x: int, y: int, w: int, h: int, point: tuple[float, float]) -> float:
    px, py = point
    dx = max(float(x) - px, 0.0, px - float(x + w - 1))
    dy = max(float(y) - py, 0.0, py - float(y + h - 1))
    return float(np.hypot(dx, dy))


def _candidate_from_components(
    mask: np.ndarray,
    previous: tuple[float, float] | None,
    maximum_jump: float,
    minimum_area: int,
) -> tuple[float, float] | None:
    """Track the center of coherent new-ink regions to avoid tip jitter."""
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    frame_area = mask.shape[0] * mask.shape[1]
    max_area = max(minimum_area * 4, int(frame_area * 0.045))
    choices: list[tuple[float, int]] = []

    for index in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[index]]
        if area < minimum_area or area > max_area:
            continue
        cx, cy = float(centroids[index][0]), float(centroids[index][1])
        if previous is None:
            score = -float(area)
        else:
            distance = float(np.hypot(cx - previous[0], cy - previous[1]))
            if distance > maximum_jump * 1.20:
                continue
            score = distance - min(area, 1200) * 0.008
        choices.append((score, index))

    if not choices:
        return None
    choices.sort(key=lambda item: item[0])
    best_index = choices[0][1]
    return float(centroids[best_index][0]), float(centroids[best_index][1])


def _open_video_writer(path: Path, fps: float, width: int, height: int):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        command = [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fps:.6f}",
            "-i", "-", "-an",
            "-c:v", "libx264", "-preset", "medium", "-crf", "13",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(path),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        return ("ffmpeg", process)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("The output video encoder could not be started.")
    return ("opencv", writer)


def _write_video_frame(writer_state, frame: np.ndarray) -> None:
    kind, writer = writer_state
    if kind == "ffmpeg":
        if writer.stdin is None:
            raise RuntimeError("The H.264 encoder input closed unexpectedly.")
        writer.stdin.write(frame.tobytes())
    else:
        writer.write(frame)


def _close_video_writer(writer_state) -> None:
    kind, writer = writer_state
    if kind == "ffmpeg":
        if writer.stdin is not None:
            writer.stdin.close()
        stderr = writer.stderr.read().decode("utf-8", errors="replace") if writer.stderr else ""
        code = writer.wait()
        if code != 0:
            raise RuntimeError(f"The H.264 encoder failed: {stderr[-500:]}")
    else:
        writer.release()


def _mux_original_audio(silent_video: Path, original_video: Path, output_video: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        shutil.move(str(silent_video), str(output_video))
        return False
    command = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(silent_video), "-i", str(original_video),
        "-map", "0:v:0", "-map", "1:a?",
        "-c:v", "copy", "-c:a", "aac", "-shortest", str(output_video),
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode == 0 and output_video.exists() and output_video.stat().st_size:
        silent_video.unlink(missing_ok=True)
        probe = subprocess.run([ffmpeg, "-i", str(original_video)], capture_output=True, check=False, text=True)
        return "Audio:" in probe.stderr
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

    writer_state = _open_video_writer(silent_path, fps, width, height)

    left = int(width * config.roi_left_percent / 100.0)
    right = int(width * config.roi_right_percent / 100.0)
    top = int(height * config.roi_top_percent / 100.0)
    bottom = int(height * config.roi_bottom_percent / 100.0)
    if right - left < 20 or bottom - top < 20:
        capture.release()
        _close_video_writer(writer_state)
        raise ValueError("The selected drawing area is too small.")

    roi_width = right - left
    roi_height = bottom - top
    tracking_scale = min(1.0, 720.0 / roi_width)
    track_size = (max(1, int(roi_width * tracking_scale)), max(1, int(roi_height * tracking_scale)))
    sprite, tip_x, tip_y = _prepare_sprite(hand_rgba, width, config)

    baseline: np.ndarray | None = None
    delta_history: deque[np.ndarray] = deque(maxlen=6)
    tracked: tuple[float, float] | None = None
    candidate_history: deque[tuple[float, float]] = deque(maxlen=5)
    idle_frames = config.hide_after_frames + 1
    hold_frames = max(config.hide_after_frames, int(round(fps * 0.35)))
    minimum_component_area = max(10, int(round(config.minimum_change_area * max(0.6, tracking_scale))))
    written = 0
    active_count = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            roi = frame[top:bottom, left:right]
            small = cv2.resize(roi, track_size, interpolation=cv2.INTER_AREA)
            tracking = _tracking_frame(small)
            if baseline is None:
                baseline = tracking.copy()

            background_delta = _background_delta(tracking, baseline)
            candidate = None
            if len(delta_history) >= 4 and not _is_large_ui_change(background_delta):
                old_delta = delta_history[-4]
                mask = _new_ink_mask(background_delta, old_delta, config)
                previous_local = None
                if tracked is not None:
                    previous_local = (
                        (tracked[0] - left) * tracking_scale,
                        (tracked[1] - top) * tracking_scale,
                    )
                local = _candidate_from_components(
                    mask,
                    previous_local,
                    config.maximum_jump * tracking_scale,
                    minimum_component_area,
                )
                if local is not None:
                    candidate = (local[0] / tracking_scale + left, local[1] / tracking_scale + top)

            delta_history.append(background_delta)
            if candidate is not None:
                candidate_history.append(candidate)
                stable_candidate = (
                    float(np.median([point[0] for point in candidate_history])),
                    float(np.median([point[1] for point in candidate_history])),
                )
                if tracked is None:
                    tracked = stable_candidate
                else:
                    dx = stable_candidate[0] - tracked[0]
                    dy = stable_candidate[1] - tracked[1]
                    distance = float(np.hypot(dx, dy))
                    dead_zone = max(3.0, width * 0.008)
                    if distance > dead_zone:
                        max_step = max(14.0, width * 0.045)
                        scale = min(1.0, max_step / max(distance, 1e-6))
                        target = (tracked[0] + dx * scale, tracked[1] + dy * scale)
                        alpha = 0.24
                        tracked = (
                            tracked[0] * (1.0 - alpha) + target[0] * alpha,
                            tracked[1] * (1.0 - alpha) + target[1] * alpha,
                        )
                idle_frames = 0
                active_count += 1
            else:
                idle_frames += 1

            if tracked is not None and idle_frames <= hold_frames:
                _overlay_rgba(frame, sprite, tracked, (tip_x, tip_y), config.hand_opacity)

            _write_video_frame(writer_state, frame)
            written += 1
            if progress and (written == 1 or written % max(1, int(fps / 2)) == 0):
                fraction = written / frame_count if frame_count > 0 else 0.0
                progress(min(.97, fraction), f"Processing frame {written:,}")
    finally:
        capture.release()
        _close_video_writer(writer_state)

    if written == 0:
        silent_path.unlink(missing_ok=True)
        raise ValueError("No readable frames were found in the uploaded video.")

    if progress:
        progress(.98, "Restoring the original audio")
    audio_preserved = _mux_original_audio(silent_path, input_path, output_path)
    if progress:
        progress(1.0, "Finished")

    return RenderResult(written, active_count, fps, width, height, audio_preserved)
