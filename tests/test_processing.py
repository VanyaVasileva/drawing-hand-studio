from pathlib import Path

import cv2
import numpy as np

from processing import RenderConfig, make_default_hand, render_hand_video


def _make_test_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (320, 240))
    assert writer.isOpened()
    for index in range(40):
        frame = np.full((240, 320, 3), 255, dtype=np.uint8)
        if index > 3:
            cv2.line(frame, (40, 120), (40 + min(index - 3, 30) * 6, 120), (30, 30, 30), 5)
        writer.write(frame)
    writer.release()


def test_renders_hand_over_synthetic_stroke(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "result.mp4"
    _make_test_video(source)

    result = render_hand_video(
        source,
        output,
        make_default_hand(360),
        RenderConfig(sensitivity=12, minimum_change_area=4, hide_after_frames=3),
    )

    assert output.exists()
    assert output.stat().st_size > 1_000
    assert result.frames == 40
    assert result.tracked_frames > 10
