from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import streamlit as st

from processing import RenderConfig, load_hand_image, render_hand_video


st.set_page_config(page_title="Drawing Hand Studio", page_icon="✍️", layout="wide")

st.title("Drawing Hand Studio")
st.caption("Add a natural hand-and-stylus overlay to any drawing screen recording.")

with st.expander("How this first prototype works", expanded=False):
    st.write(
        "The tracker follows newly added drawing changes inside the selected canvas area. "
        "It uses the original canvas as a background reference so white, beige, and textured paper "
        "can be tracked without changing the hand artwork."
    )

video = st.file_uploader("Upload a drawing screen recording", type=["mp4", "mov", "m4v"])

if video is not None:
    suffix = Path(video.name).suffix.lower() or ".mp4"
    input_bytes = video.getvalue()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as source:
        source.write(input_bytes)
        source_path = Path(source.name)

    capture = cv2.VideoCapture(str(source_path))
    ok, first_frame = capture.read()
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    source_path.unlink(missing_ok=True)

    if not ok:
        st.error("This video could not be previewed. Please try an MP4 or MOV screen recording.")
        st.stop()

    duration = frames / fps if fps else 0
    preview_col, info_col = st.columns([1.4, 1])
    with preview_col:
        st.image(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB), caption="First video frame")
    with info_col:
        st.metric("Video size", f"{width} × {height}")
        st.metric("Duration", f"{duration:.1f} seconds")
        if duration > 90:
            st.warning("For the first hosted version, test with a clip under 90 seconds.")

    st.subheader("1. Drawing area")
    st.caption("Exclude Procreate or app menus when possible. Start with the full frame for a simple recording.")
    area_a, area_b = st.columns(2)
    with area_a:
        left = st.slider("Left edge (%)", 0, 80, 0)
        right = st.slider("Right edge (%)", 20, 100, 100)
    with area_b:
        top = st.slider("Top edge (%)", 0, 80, 0)
        bottom = st.slider("Bottom edge (%)", 20, 100, 100)

    st.subheader("2. Hand")
    custom_hand = st.file_uploader(
        "Optional transparent hand PNG",
        type=["png"],
        help="Leave empty to use the built-in illustrated hand, or upload your own transparent PNG to override it.",
    )
    hand_a, hand_b, hand_c = st.columns(3)
    with hand_a:
        hand_side = st.selectbox("Hand enters from", ["Right", "Left"])
        hand_size = st.slider("Hand size (% of video width)", 12, 80, 50)
    with hand_b:
        tip_x = st.slider("Pencil tip X inside hand (%)", 0, 100, 15)
        tip_y = st.slider("Pencil tip Y inside hand (%)", 0, 100, 34)
    with hand_c:
        opacity = st.slider("Hand opacity", 0.20, 1.00, 0.94, 0.02)
        smoothing = st.slider("Movement smoothing", 0.05, 1.00, 0.42, 0.01)

    st.subheader("3. Tracking")
    track_a, track_b, track_c = st.columns(3)
    with track_a:
        sensitivity = st.slider("Pixel-change sensitivity", 5, 80, 24)
    with track_b:
        minimum_area = st.slider("Minimum changed area", 4, 150, 18)
    with track_c:
        hide_after = st.slider("Lift hand after frames", 1, 30, 12)

    if left >= right or top >= bottom:
        st.error("The drawing area edges overlap. Please widen the selected area.")
    elif st.button("Create hand-following video", type="primary", use_container_width=True):
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as source:
            source.write(input_bytes)
            source_path = Path(source.name)
        config = RenderConfig(
            sensitivity=sensitivity,
            minimum_change_area=minimum_area,
            smoothing=smoothing,
            hide_after_frames=hide_after,
            hand_width_percent=hand_size,
            hand_opacity=opacity,
            hand_side=hand_side,
            tip_x_percent=tip_x,
            tip_y_percent=tip_y,
            roi_left_percent=left,
            roi_top_percent=top,
            roi_right_percent=right,
            roi_bottom_percent=bottom,
        )
        hand = load_hand_image(custom_hand.getvalue() if custom_hand else None)
        progress_bar = st.progress(0.0, text="Starting")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as result_file:
            result_path = Path(result_file.name)

        try:
            result = render_hand_video(
                source_path,
                result_path,
                hand,
                config,
                progress=lambda value, label: progress_bar.progress(value, text=label),
            )
            result_bytes = result_path.read_bytes()
            st.session_state["rendered_video"] = result_bytes
            st.session_state["rendered_name"] = f"hand-{Path(video.name).stem}.mp4"
            st.session_state["render_stats"] = result
        except Exception as exc:
            st.error(f"The video could not be processed: {exc}")
        finally:
            result_path.unlink(missing_ok=True)
            source_path.unlink(missing_ok=True)

    if "rendered_video" in st.session_state:
        stats = st.session_state.get("render_stats")
        st.subheader("Result")
        st.video(st.session_state["rendered_video"])
        if stats:
            st.caption(
                f"Tracked movement in {stats.tracked_frames:,} of {stats.frames:,} frames. "
                + ("Original audio was preserved." if stats.audio_preserved else "No original audio track was found or FFmpeg was unavailable.")
            )
        st.download_button(
            "Download MP4",
            data=st.session_state["rendered_video"],
            file_name=st.session_state["rendered_name"],
            mime="video/mp4",
            use_container_width=True,
        )
else:
    st.info("Upload a short screen recording to begin. A 10–30 second test clip is ideal for the first validation.")

st.divider()
st.caption("Automatic drawing-hand tracking with canvas-color and texture compensation.")
