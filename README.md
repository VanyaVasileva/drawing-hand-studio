# Drawing Hand Studio

Drawing Hand Studio is an experimental Streamlit tool that adds an animated hand-and-stylus overlay to drawing screen recordings. It is designed to work with recordings from Procreate and other drawing apps without recreating their brushes.

## Current MVP

- Upload MP4, MOV, or M4V screen recordings.
- Restrict tracking to the drawing area.
- Detect changing pixels and estimate the active brush tip.
- Smooth the tracked motion and hide the hand during pauses.
- Use the built-in placeholder hand or upload a transparent PNG.
- Choose left/right entry, hand scale, opacity, and tip alignment.
- Export an MP4 and preserve the original audio when FFmpeg is available.

## Important limitation

The first tracker is deliberately labelled experimental. It works best when the canvas is stationary. Large interface changes, zooming, rotating, undoing, or moving the canvas may be mistaken for drawing. It must be tested with a real drawing screen recording before its accuracy can be considered verified.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

FFmpeg is recommended for copying the source video's audio into the rendered result.

## Streamlit Community Cloud

1. Connect this repository to Streamlit Community Cloud.
2. Set `app.py` as the entry point.
3. The included `packages.txt` installs FFmpeg on the hosted app.

For the first hosted tests, use short clips (about 10–30 seconds) to keep processing time and memory predictable.
