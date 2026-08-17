"""Run the complete NovaRoute.AI ML pipeline.

Pipeline:
    YOLO + ByteTrack -> active vehicle counts -> congestion -> risk

The pipeline emits one integration record per observation interval and writes
an annotated demo video. Congestion is based on the rolling average number of
concurrently active tracked vehicles, never on a sum of detections across
frames.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[2]

# The source modules use "src.*" imports.
if str(ROOT / "ml") not in sys.path:
    sys.path.insert(0, str(ROOT / "ml"))

from src.detection.vehicle_detector import VehicleDetector  # noqa: E402
from src.risk.risk_scorer import RiskScorer, build_risk_config  # noqa: E402
from src.tracking.vehicle_tracker import VehicleTracker  # noqa: E402
from src.traffic.congestion import (  # noqa: E402
    CongestionEstimator,
    build_congestion_config,
)
from src.traffic.vehicle_counter import VehicleCounter  # noqa: E402
from src.utils.visualization import draw_tracks, put_summary_text  # noqa: E402


SCHEMA_VERSION = "1.1.0"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    """Load and validate the central YAML configuration."""

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    required_sections = (
        "detection",
        "tracking",
        "congestion",
        "risk",
        "pipeline",
        "io",
    )

    for key in required_sections:
        if key not in cfg:
            raise ValueError(
                f"Missing required config section: {key}"
            )

    return cfg


# ---------------------------------------------------------------------------
# Component builders
# ---------------------------------------------------------------------------

def build_detector(cfg: dict) -> VehicleDetector:
    """Build the YOLO vehicle detector."""

    det = cfg["detection"]

    configured_model = str(
        det["model_weights"]
    )

    model_path = Path(
        configured_model
    )

    # Preserve bare model names such as yolov8n.pt so
    # Ultralytics can download the official checkpoint.
    if model_path.is_absolute() or model_path.parent != Path("."):
        if not model_path.is_absolute():
            model_path = ROOT / model_path

        configured_model = str(
            model_path
        )

    return VehicleDetector(
        model_weights=configured_model,
        target_classes=det["target_classes"],
        confidence_threshold=float(
            det["confidence_threshold"]
        ),
        iou_threshold=float(
            det["iou_threshold"]
        ),
        device=str(
            det["device"]
        ),
    )


def build_tracker(
    cfg: dict,
    detector: VehicleDetector,
) -> VehicleTracker:
    """Build the ByteTrack vehicle tracker."""

    trk = cfg.get(
        "tracking",
        {}
    ) or {}

    return VehicleTracker(
        detector=detector,
        tracker_config=str(
            trk.get(
                "tracker_config",
                "bytetrack.yaml",
            )
        ),
        confidence_threshold=trk.get(
            "confidence_threshold"
        ),
        iou_threshold=trk.get(
            "iou_threshold"
        ),
        device=trk.get(
            "device"
        ),
    )


def build_counter(
    cfg: dict,
    fps: float,
) -> VehicleCounter:
    """Build a rolling vehicle counter using a time-based window."""

    pipeline_cfg = cfg["pipeline"]

    observation_window_seconds = float(
        pipeline_cfg.get(
            "observation_window_seconds",
            5.0,
        )
    )

    if observation_window_seconds <= 0:
        raise ValueError(
            "pipeline.observation_window_seconds must be > 0"
        )

    if fps <= 0:
        raise ValueError(
            "FPS must be > 0"
        )

    window_frames = max(
        1,
        int(
            round(
                observation_window_seconds * fps
            )
        ),
    )

    return VehicleCounter(
        window_frames
    )


def build_congestion_estimator(
    cfg: dict,
) -> CongestionEstimator:
    """Build the BPR congestion estimator."""

    return CongestionEstimator(
        build_congestion_config(cfg)
    )


def build_risk_scorer(
    cfg: dict,
) -> RiskScorer:
    """Build the transparent risk scorer."""

    return RiskScorer(
        build_risk_config(cfg)
    )


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def _fps(
    cap: cv2.VideoCapture,
    fallback_fps: float = 25.0,
) -> float:
    """Return video FPS or a configured fallback."""

    value = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    if value > 0:
        return value

    if fallback_fps <= 0:
        return 25.0

    return fallback_fps


def _resolve_hour(
    hour: Optional[int],
) -> int:
    """Resolve simulation hour for recorded videos."""

    if hour is None:
        return datetime.now().hour

    if not 0 <= hour <= 23:
        raise ValueError(
            "--hour must be between 0 and 23"
        )

    return hour


def open_writer(
    out_path: Path,
    cap: cv2.VideoCapture,
    fps: float,
) -> cv2.VideoWriter:
    """Create and validate the output video writer."""

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    if width <= 0 or height <= 0:
        raise RuntimeError(
            "Could not determine video dimensions"
        )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open output video for writing: "
            f"{out_path}"
        )

    return writer


# ---------------------------------------------------------------------------
# Observation / JSON
# ---------------------------------------------------------------------------

def build_observation(
    location_id: str,
    window_count,
    congestion_result,
    risk_result,
    observation_window_seconds: float,
) -> Dict[str, Any]:
    """Build the JSON integration record."""

    counts = window_count.to_dict()

    return {
        "schema_version": SCHEMA_VERSION,

        "road_segment_id": location_id,

        "osm_edge": None,

        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "observation_window_seconds": (
            observation_window_seconds
        ),

        "vehicle_counts": {
            "car": counts["car"],
            "motorcycle": counts["motorcycle"],
            "bus": counts["bus"],
            "truck": counts["truck"],
            "total": counts["total"],
        },

        "total_vehicles": counts["total"],

        "peak_vehicles": counts["peak_total"],

        "road_capacity": (
            congestion_result.road_capacity
        ),

        # Kept as "traffic_density" for compatibility
        # with the existing JSON schema/backend contract.
        #
        # Semantically this value is the V/C ratio:
        # concurrent active vehicles / road capacity.
        "traffic_density": (
            congestion_result.traffic_density
        ),

        "congestion_level": (
            congestion_result.congestion_level.value
        ),

        "congestion_factor": (
            congestion_result.congestion_factor
        ),

        "aggregation_method": (
            counts["aggregation_method"]
        ),

        "risk_score": round(
            risk_result.overall_risk_score,
            4,
        ),

        "risk_level": (
            risk_result.risk_level.value
        ),

        "risk_factor_scores": (
            risk_result.factor_scores.to_dict()
        ),

        "risk_contributions": (
            risk_result.contributions.to_dict()
        ),
    }


def write_jsonl(
    observation: Dict[str, Any],
    metrics_dir: Path,
    location_id: str,
) -> Path:
    """Append an observation to the JSONL history file."""

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        metrics_dir
        / f"{location_id}_metrics.jsonl"
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(observation)
            + "\n"
        )

    return path


def write_latest(
    observation: Dict[str, Any],
    metrics_dir: Path,
    location_id: str,
) -> Path:
    """Write the latest observation JSON."""

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        metrics_dir
        / f"{location_id}_latest.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            observation,
            f,
            indent=2,
        )

    return path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    cfg: dict,
    source,
    location_id: str,
    metrics_dir: Path,
    output_video: Path,
    road_capacity: Optional[float],
    accident_count: float,
    pedestrian_count: float,
    hour: Optional[int],
    show: bool,
) -> None:
    """Run YOLO -> ByteTrack -> counting -> congestion -> risk."""

    # -----------------------------------------------------------------------
    # Validate inputs
    # -----------------------------------------------------------------------

    if road_capacity is not None:
        if road_capacity <= 0:
            raise ValueError(
                "--road-capacity must be > 0"
            )

    if accident_count < 0:
        raise ValueError(
            "--accident-count cannot be negative"
        )

    if pedestrian_count < 0:
        raise ValueError(
            "--pedestrian-count cannot be negative"
        )

    # Validate hour before starting expensive inference.
    resolved_hour = _resolve_hour(hour)

    # -----------------------------------------------------------------------
    # Build ML components
    # -----------------------------------------------------------------------

    detector = build_detector(
        cfg
    )

    tracker = build_tracker(
        cfg,
        detector,
    )

    tracker.reset()

    congestion_estimator = (
        build_congestion_estimator(cfg)
    )

    risk_scorer = build_risk_scorer(
        cfg
    )

    # -----------------------------------------------------------------------
    # IMPORTANT:
    # Open the source BEFORE using cap or fps.
    #
    # This fixes the previous:
    # NameError: name 'cap' is not defined
    # -----------------------------------------------------------------------

    cap = cv2.VideoCapture(
        source
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video source: {source}"
        )

    # -----------------------------------------------------------------------
    # FPS
    # -----------------------------------------------------------------------

    fallback_fps = float(
        cfg["pipeline"].get(
            "fallback_fps",
            25.0,
        )
    )

    fps = _fps(
        cap,
        fallback_fps,
    )

    # -----------------------------------------------------------------------
    # Time-based pipeline configuration
    # -----------------------------------------------------------------------

    pipeline_cfg = cfg["pipeline"]

    observation_window_seconds = float(
        pipeline_cfg.get(
            "observation_window_seconds",
            5.0,
        )
    )

    emit_every_seconds = float(
        pipeline_cfg.get(
            "emit_every_seconds",
            1.0,
        )
    )

    if observation_window_seconds <= 0:
        cap.release()

        raise ValueError(
            "pipeline.observation_window_seconds "
            "must be > 0"
        )

    if emit_every_seconds <= 0:
        cap.release()

        raise ValueError(
            "pipeline.emit_every_seconds "
            "must be > 0"
        )

    # -----------------------------------------------------------------------
    # Rolling vehicle counter
    # -----------------------------------------------------------------------

    counter = build_counter(
        cfg,
        fps,
    )

    # Convert emission interval from seconds to frames.
    emit_every = max(
        1,
        int(
            round(
                emit_every_seconds * fps
            )
        ),
    )

    # -----------------------------------------------------------------------
    # Output writer
    # -----------------------------------------------------------------------

    writer = open_writer(
        output_video,
        cap,
        fps,
    )

    # -----------------------------------------------------------------------
    # Runtime state
    # -----------------------------------------------------------------------

    frame_idx = 0

    last_observation: Optional[
        Dict[str, Any]
    ] = None

    print(
        f"=== NovaRoute.AI pipeline: "
        f"{location_id} ==="
    )

    print(
        f"FPS={fps:.2f}, "
        f"window="
        f"{counter.window_size / fps:.2f}s "
        f"({counter.window_size} frames), "
        f"emit_every="
        f"{emit_every / fps:.2f}s "
        f"({emit_every} frames)"
    )

    print(
        f"Road capacity="
        f"{road_capacity if road_capacity is not None else 'config default'}"
    )

    print(
        f"Simulation hour="
        f"{resolved_hour:02d}:00"
    )

    print()

    # -----------------------------------------------------------------------
    # Process video
    # -----------------------------------------------------------------------

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            frame_idx += 1

            # ---------------------------------------------------------------
            # YOLO + ByteTrack
            # ---------------------------------------------------------------

            tracked = tracker.track(
                frame,
                frame_idx,
            )

            # ---------------------------------------------------------------
            # Current-frame active vehicle counts
            # ---------------------------------------------------------------

            frame_count = (
                counter.update_tracked(
                    tracked
                )
            )

            # ---------------------------------------------------------------
            # Rolling-window aggregate
            # ---------------------------------------------------------------

            window_count = (
                counter.get_window_aggregate()
            )

            # ---------------------------------------------------------------
            # Draw vehicle tracks
            # ---------------------------------------------------------------

            annotated = draw_tracks(
                frame,
                tracked,
            )

            active_track_ids = {
                vehicle.track_id
                for vehicle in tracked
                if vehicle.track_id is not None
            }

            # ---------------------------------------------------------------
            # Basic dashboard
            # ---------------------------------------------------------------

            lines = [
                "NovaRoute.AI",
                f"Frame: {frame_idx}",
                f"Cars: {frame_count.car}",
                (
                    f"Motorcycles: "
                    f"{frame_count.motorcycle}"
                ),
                f"Buses: {frame_count.bus}",
                f"Trucks: {frame_count.truck}",
                (
                    f"Active vehicles: "
                    f"{frame_count.total}"
                ),
                (
                    f"Tracked IDs: "
                    f"{len(active_track_ids)}"
                ),
            ]

            # ---------------------------------------------------------------
            # Emit observation after rolling window is ready
            # ---------------------------------------------------------------

            if (
                frame_idx >= counter.window_size
                and frame_idx % emit_every == 0
            ):

                window_seconds = (
                    window_count.frame_count
                    / fps
                )

                congestion = (
                    congestion_estimator.estimate(
                        road_id=location_id,
                        vehicle_counts=window_count,
                        window_start=(
                            window_count.start_frame
                            / fps
                        ),
                        window_end=(
                            window_count.end_frame
                            / fps
                        ),
                        road_capacity=road_capacity,
                        aggregation_method=(
                            window_count
                            .aggregation_method
                        ),
                    )
                )

                risk = risk_scorer.score(
                    junction_id=location_id,
                    accident_count=(
                        accident_count
                    ),
                    traffic_density=(
                        congestion
                    ),
                    pedestrian_count=(
                        pedestrian_count
                    ),
                    hour=resolved_hour,
                )

                last_observation = (
                    build_observation(
                        location_id=location_id,
                        window_count=window_count,
                        congestion_result=congestion,
                        risk_result=risk,
                        observation_window_seconds=(
                            window_seconds
                        ),
                    )
                )

                write_jsonl(
                    last_observation,
                    metrics_dir,
                    location_id,
                )

                write_latest(
                    last_observation,
                    metrics_dir,
                    location_id,
                )

            # ---------------------------------------------------------------
            # Add latest ML results to dashboard
            # ---------------------------------------------------------------

            if last_observation:

                traffic_load_percent = (
                    last_observation[
                        "traffic_density"
                    ]
                    * 100
                )

                lines.extend(
                    [
                        (
                            f"Traffic Load: "
                            f"{traffic_load_percent:.1f}%"
                        ),
                        (
                            f"Congestion: "
                            f"{last_observation['congestion_level']}"
                        ),
                        (
                            f"Congestion Factor: "
                            f"{last_observation['congestion_factor']:.3f}"
                        ),
                        (
                            f"Risk Score: "
                            f"{last_observation['risk_score']:.1f}"
                        ),
                        (
                            f"Risk Level: "
                            f"{last_observation['risk_level'].upper()}"
                        ),
                    ]
                )

            else:

                lines.append(
                    "Warming up observation window..."
                )

            # ---------------------------------------------------------------
            # Draw dashboard
            # ---------------------------------------------------------------

            put_summary_text(
                annotated,
                lines,
            )

            # ---------------------------------------------------------------
            # Write annotated frame
            # ---------------------------------------------------------------

            writer.write(
                annotated
            )

            # ---------------------------------------------------------------
            # Optional display
            # ---------------------------------------------------------------

            if show:

                cv2.imshow(
                    "NovaRoute.AI - ML Pipeline",
                    annotated,
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key == ord("q") or key == 27:
                    print(
                        "Pipeline stopped by user."
                    )
                    break

    finally:

        cap.release()

        writer.release()

        if show:
            cv2.destroyAllWindows()

    # -----------------------------------------------------------------------
    # Final result
    # -----------------------------------------------------------------------

    print()
    print(
        "=== Pipeline finished ==="
    )

    print(
        f"Annotated output: "
        f"{output_video.resolve()}"
    )

    if last_observation:

        latest_json = (
            metrics_dir
            / f"{location_id}_latest.json"
        )

        print(
            f"Latest metrics: "
            f"{latest_json.resolve()}"
        )

        print()
        print(
            "=== Latest ML observation ==="
        )

        print(
            json.dumps(
                last_observation,
                indent=2,
            )
        )

    else:

        print(
            "No observation was emitted. "
            "The video did not reach the configured "
            "observation window."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete NovaRoute.AI "
            "YOLO + ByteTrack + congestion + "
            "risk ML pipeline."
        )
    )

    parser.add_argument(
        "--source",
        required=True,
        help=(
            "Video path or webcam index. "
            "Example: traffic_video.mp4 or 0"
        ),
    )

    parser.add_argument(
        "--location-id",
        default="junction_1",
        help=(
            "Road segment / junction identifier."
        ),
    )

    parser.add_argument(
        "--config",
        default=str(
            ROOT / "ml/config/config.yaml"
        ),
        help=(
            "Path to config.yaml."
        ),
    )

    parser.add_argument(
        "--metrics-dir",
        default=str(
            ROOT / "ml/outputs/metrics"
        ),
        help=(
            "Directory for JSON/JSONL metrics."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional annotated pipeline video path."
        ),
    )

    parser.add_argument(
        "--road-capacity",
        type=float,
        default=None,
        help=(
            "Calibrated concurrent vehicle capacity. "
            "If omitted, the configured road/default "
            "capacity is used."
        ),
    )

    parser.add_argument(
        "--accident-count",
        type=float,
        default=0.0,
        help=(
            "Historical accident count used by "
            "the transparent risk scorer."
        ),
    )

    parser.add_argument(
        "--pedestrian-count",
        type=float,
        default=0.0,
        help=(
            "Pedestrian conflict count used by "
            "the transparent risk scorer."
        ),
    )

    parser.add_argument(
        "--hour",
        type=int,
        default=None,
        help=(
            "Hour 0-23 represented by a recorded video. "
            "If omitted, the current local hour is used."
        ),
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help=(
            "Disable the OpenCV preview window."
        ),
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load config
    # -----------------------------------------------------------------------

    config_path = Path(
        args.config
    )

    cfg = load_config(
        config_path
    )

    # -----------------------------------------------------------------------
    # Resolve source
    # -----------------------------------------------------------------------

    is_webcam = (
        args.source.isdigit()
    )

    source = (
        int(args.source)
        if is_webcam
        else args.source
    )

    # -----------------------------------------------------------------------
    # Resolve output path
    # -----------------------------------------------------------------------

    if args.output:

        output_video = Path(
            args.output
        )

    else:

        pipeline_output_dir = (
            ROOT
            / cfg["io"]["pipeline_out"]
        )

        if is_webcam:

            output_video = (
                pipeline_output_dir
                / "webcam_pipeline.mp4"
            )

        else:

            output_video = (
                pipeline_output_dir
                / (
                    f"{Path(args.source).stem}"
                    "_pipeline.mp4"
                )
            )

    # -----------------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------------

    run_pipeline(
        cfg=cfg,
        source=source,
        location_id=args.location_id,
        metrics_dir=Path(
            args.metrics_dir
        ),
        output_video=output_video,
        road_capacity=args.road_capacity,
        accident_count=args.accident_count,
        pedestrian_count=args.pedestrian_count,
        hour=args.hour,
        show=not args.no_display,
    )


if __name__ == "__main__":
    main()