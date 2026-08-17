# NovaRoute.AI ML Module

This module implements Koushik's track from the concept note:

**YOLO → ByteTrack → active vehicle measurement → congestion factor → transparent risk score**.

The design intentionally avoids an unnecessary custom ML model. YOLO/ByteTrack provides the live vehicle observations; congestion and risk are deterministic and explainable.

## Important measurement rule

Do **not** sum detections across video frames to estimate the number of vehicles. The same car appears in many frames. The pipeline therefore uses the **rolling average number of active tracked vehicles per frame** as an occupancy-style measurement.

`traffic_density` in the integration contract means:

`average active vehicles / calibrated concurrent vehicle capacity`

This is a V/C ratio, not a physical vehicles-per-kilometre density.

## Setup

From the repository root:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ultralytics will download the configured YOLO checkpoint the first time it is needed if the checkpoint is not already available.

## Run detection

```bash
python ml/scripts/run_detection.py --source ml/data/test_videos/traffic_video.mp4 --no-display
```

## Run tracking

```bash
python ml/scripts/run_tracking.py --source ml/data/test_videos/traffic_video.mp4 --no-display
```

## Run the complete pipeline

```bash
python ml/scripts/run_pipeline.py \
  --source ml/data/test_videos/traffic_video.mp4 \
  --location-id junction_1 \
  --road-capacity 50 \
  --accident-count 2 \
  --pedestrian-count 8 \
  --hour 18 \
  --no-display
```

The pipeline writes:

- annotated demo video under `ml/outputs/pipeline/`
- rolling JSONL observations under `ml/outputs/metrics/`
- latest JSON snapshot under `ml/outputs/metrics/`

The JSON output follows `integration/schemas/traffic_data.json`.

## Risk model

Risk is a weighted, normalized sum of:

- accident history
- traffic V/C ratio
- pedestrian conflict input
- time of day

The current time profile and normalization caps are prototype parameters and should be calibrated with Nagpur data before operational use.

## Testing

```bash
pytest -q
```

The unit tests do not require a YOLO checkpoint or camera/video file. Full inference additionally requires the dependencies in `requirements.txt` and a working OpenCV/Ultralytics runtime.
