from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw


ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class RenderConfig:
    sensitivity: int = 24
    minimum_change_area: int = 18
    smoothing: float = 0.42
    hide_after_frames: int = 5
    maximum_jump: int = 220
    hand_width_percent: float = 34.0
    hand_opacity: float = 0.96
    hand_side: str = "Right"
    tip_x_percent: float = 8.0
    tip_y_percent: float = 13.0
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


def make_default_hand(width: int = 900) -> np.ndarray:
    """Create a clean placeholder hand-and-stylus sprite as transparent RGBA."""
    scale = 3
    w = width * scale
    h = int(width * 0.86) * scale
    image = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image, "RGBA")

    def pts(values: list[tuple[float, float]]) -> list[tuple[int, int]]:
        return [(int(x * w), int(y * h)) for x, y in values]

    # Soft shadow below the hand.
    draw.ellipse((int(.33*w), int(.57*h), int(1.02*w), int(1.03*h)), fill=(50, 43, 36, 26))

    # White stylus, with its tip intentionally close to (8%, 13%).
    stylus = pts([(.075, .13), (.105, .095), (.765, .67), (.72, .72)])
    draw.line(stylus, fill=(62, 59, 55, 255), width=int(.043*w), joint="curve")
    draw.line(stylus, fill=(244, 243, 239, 255), width=int(.029*w), joint="curve")
    draw.ellipse((int(.055*w), int(.105*h), int(.095*w), int(.155*h)), fill=(45, 43, 40, 255))

    skin = (224, 184, 151, 255)
    skin_light = (241, 207, 178, 255)
    outline = (116, 82, 65, 155)

    # Palm and wrist entering from the lower-right edge.
    palm = pts([(.47,.61),(.60,.47),(.77,.48),(.91,.61),(1.02,.75),(1.03,1.02),(.48,1.02),(.38,.83)])
    draw.polygon(palm, fill=skin)
    draw.line(palm + [palm[0]], fill=outline, width=int(.008*w), joint="curve")

    # Fingers wrapped around the stylus.
    draw.rounded_rectangle((int(.39*w),int(.47*h),int(.73*w),int(.64*h)), radius=int(.07*w), fill=skin_light, outline=outline, width=int(.008*w))
    draw.rounded_rectangle((int(.49*w),int(.55*h),int(.83*w),int(.72*h)), radius=int(.07*w), fill=skin, outline=outline, width=int(.008*w))
    draw.rounded_rectangle((int(.54*w),int(.64*h),int(.88*w),int(.80*h)), radius=int(.065*w), fill=skin, outline=outline, width=int(.008*w))
    draw.ellipse((int(.37*w),int(.39*h),int(.62*w),int(.67*h)), fill=skin_light, outline=outline, width=int(.008*w))

    image = image.resize((width, int(width * .86)), Image.Resampling.LANCZOS)
    return np.asarray(image)


def load_hand_image(data: bytes | None) -> np.ndarray:
    if not data:
        return make_default_hand()
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

    # The newest brush tip is usually at the leading edge of the changed pixels.
    cutoff = np.percentile(distances, 88)
    leading = points[distances >= cutoff]
    return float(np.median(leading[:, 0])), float(np.median(leading[:, 1]))


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
        probe = subprocess.run(
            [ffmpeg, "-i", str(original_video)], capture_output=True, check=False, text=True
        )
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
            changed_area = 0
            if previous_gray is not None:
                difference = cv2.absdiff(gray, previous_gray)
                _, mask = cv2.threshold(difference, config.sensitivity, 255, cv2.THRESH_BINARY)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
                mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
                changed_area = int(cv2.countNonZero(mask))
                previous_local = None
                if tracked is not None:
                    previous_local = ((tracked[0] - left) * tracking_scale, (tracked[1] - top) * tracking_scale)
                if changed_area >= config.minimum_change_area:
                    local = _candidate_from_change(mask, previous_local, config.maximum_jump * tracking_scale)
                    if local is not None:
                        candidate = (local[0] / tracking_scale + left, local[1] / tracking_scale + top)

            previous_gray = gray
            if candidate is not None:
                if tracked is None:
                    tracked = candidate
                else:
                    a = float(np.clip(config.smoothing, 0.05, 1.0))
                    tracked = (tracked[0] * (1.0 - a) + candidate[0] * a, tracked[1] * (1.0 - a) + candidate[1] * a)
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
    audio_preserved = _mux_original_audio(silent_path, input_path, output_path)
    if progress:
        progress(1.0, "Finished")

    return RenderResult(written, active_count, fps, width, height, audio_preserved)
