from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import yaml


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from ml.src.detection.vehicle_detector import VehicleDetector
from ml.src.tracking.vehicle_tracker import VehicleTracker
from ml.src.traffic.vehicle_counter import VehicleCounter
from ml.src.traffic.congestion import CongestionEstimator
from ml.src.risk.risk_scorer import RiskScorer


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "ml" / "config" / "config.yaml"

DEFAULT_METRICS_DIR = PROJECT_ROOT / "ml" / "outputs" / "metrics"

DEFAULT_PIPELINE_OUTPUT_DIR = (
    PROJECT_ROOT / "ml" / "outputs" / "pipeline"
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Invalid configuration file: {path}"
        )

    return config


# ---------------------------------------------------------------------------
# Component builders
# ---------------------------------------------------------------------------

def build_detector(cfg: dict) -> VehicleDetector:
    detection_cfg = cfg.get("detection", {})

    model_weights = detection_cfg.get(
        "model_weights",
        "yolov8n.pt",
    )

    confidence_threshold = float(
        detection_cfg.get(
            "confidence_threshold",
            0.25,
        )
    )

    iou_threshold = float(
        detection_cfg.get(
            "iou_threshold",
            0.45,
        )
    )

    device = detection_cfg.get(
        "device",
        "cpu",
    )

    return VehicleDetector(
        model_weights=model_weights,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        device=device,
    )


def build_tracker(
    cfg: dict,
    detector: VehicleDetector,
) -> VehicleTracker:
    tracking_cfg = cfg.get("tracking", {})

    tracker_config = tracking_cfg.get(
        "tracker_config",
        "bytetrack.yaml",
    )

    return VehicleTracker(
        detector=detector,
        tracker_config=tracker_config,
    )


def build_counter(
    cfg: dict,
    fps: float,
) -> VehicleCounter:
    """
    Build the rolling vehicle counter.

    The configuration is time-based rather than frame-based.

    Example:
        observation_window_seconds = 5
        fps = 25

        window_frames = 5 * 25 = 125
    """

    pipeline_cfg = cfg.get("pipeline", {})

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
        raise ValueError("FPS must be > 0")

    window_frames = max(
        1,
        int(
            round(
                observation_window_seconds * fps
            )
        ),
    )

    return VehicleCounter(window_frames)


def build_congestion_estimator(
    cfg: dict,
) -> CongestionEstimator:
    congestion_cfg = cfg.get(
        "congestion",
        {},
    )

    return CongestionEstimator(
        alpha=float(
            congestion_cfg.get(
                "alpha",
                0.15,
            )
        ),
        beta=float(
            congestion_cfg.get(
                "beta",
                4.0,
            )
        ),
        thresholds=congestion_cfg.get(
            "thresholds",
            None,
        ),
    )


def build_risk_scorer(
    cfg: dict,
) -> RiskScorer:
    risk_cfg = cfg.get(
        "risk",
        {},
    )

    return RiskScorer(
        weights=risk_cfg.get(
            "weights",
            None,
        ),
        normalization=risk_cfg.get(
            "normalization",
            None,
        ),
        thresholds=risk_cfg.get(
            "thresholds",
            None,
        ),
    )


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def get_fps(
    cap: cv2.VideoCapture,
    cfg: dict,
) -> float:
    """
    Return FPS reported by the video.

    If OpenCV cannot determine the FPS, use the configured fallback.
    """

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    if fps <= 0:
        fps = float(
            cfg.get(
                "pipeline",
                {},
            ).get(
                "fallback_fps",
                25.0,
            )
        )

    if fps <= 0:
        fps = 25.0

    return fps


def open_writer(
    output_path: Path,
    cap: cv2.VideoCapture,
    fps: float,
) -> cv2.VideoWriter:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
            "Could not determine video dimensions."
        )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create output video: "
            f"{output_path}"
        )

    return writer


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def draw_tracked_vehicles(
    frame,
    tracked_vehicles,
):
    """
    Draw ByteTrack bounding boxes and IDs.
    """

    for vehicle in tracked_vehicles:
        x1, y1, x2, y2 = map(
            int,
            vehicle.bounding_box,
        )

        track_id = (
            vehicle.track_id
            if vehicle.track_id is not None
            else "?"
        )

        label = (
            f"{vehicle.class_name} "
            f"ID:{track_id} "
            f"{vehicle.confidence:.2f}"
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return frame


def draw_dashboard(
    frame,
    lines: list[str],
):
    """
    Draw traffic/risk information
    on top of the output frame.
    """

    x = 15
    y = 30

    line_height = 25

    overlay = frame.copy()

    panel_width = 430

    panel_height = (
        len(lines) * line_height + 25
    )

    cv2.rectangle(
        overlay,
        (5, 5),
        (
            panel_width,
            panel_height,
        ),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0,
        frame,
    )

    for line in lines:
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        y += line_height

    return frame


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def make_json_safe(value: Any) -> Any:
    """
    Convert objects such as dataclasses
    into JSON-safe Python structures.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(val)
            for key, val in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return [
            make_json_safe(item)
            for item in value
        ]

    if hasattr(value, "__dict__"):
        return make_json_safe(
            vars(value)
        )

    return str(value)


def save_metrics(
    metrics_dir: Path,
    location_id: str,
    payload: dict,
) -> None:
    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    latest_file = (
        metrics_dir
        / f"{location_id}_latest.json"
    )

    history_file = (
        metrics_dir
        / f"{location_id}_metrics.jsonl"
    )

    with latest_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )

    with history_file.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(payload)
            + "\n"
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    source: str,
    location_id: str,
    road_capacity: float,
    accident_count: float,
    pedestrian_count: float,
    hour: int | None,
    config_path: str | Path,
    output_video: str | Path | None = None,
    metrics_dir: str | Path | None = None,
    show: bool = True,
) -> dict | None:

    # -------------------------------------------------------
    # Validate user inputs
    # -------------------------------------------------------

    if road_capacity <= 0:
        raise ValueError(
            "road_capacity must be > 0"
        )

    if accident_count < 0:
        raise ValueError(
            "accident_count cannot be negative"
        )

    if pedestrian_count < 0:
        raise ValueError(
            "pedestrian_count cannot be negative"
        )

    if hour is not None and not (
        0 <= hour <= 23
    ):
        raise ValueError(
            "hour must be between 0 and 23"
        )

    # -------------------------------------------------------
    # Load configuration
    # -------------------------------------------------------

    cfg = load_config(config_path)

    # -------------------------------------------------------
    # Create ML components
    # -------------------------------------------------------

    detector = build_detector(cfg)

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

    # -------------------------------------------------------
    # Open video
    #
    # IMPORTANT:
    # cap must be created BEFORE FPS,
    # VehicleCounter or VideoWriter.
    # -------------------------------------------------------

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video source: "
            f"{source}"
        )

    # -------------------------------------------------------
    # Determine FPS
    # -------------------------------------------------------

    fps = get_fps(
        cap,
        cfg,
    )

    # -------------------------------------------------------
    # Pipeline timing configuration
    # -------------------------------------------------------

    pipeline_cfg = cfg.get(
        "pipeline",
        {},
    )

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

    # -------------------------------------------------------
    # Vehicle counter
    # -------------------------------------------------------

    counter = build_counter(
        cfg,
        fps,
    )

    emit_every_frames = max(
        1,
        int(
            round(
                emit_every_seconds * fps
            )
        ),
    )

    # -------------------------------------------------------
    # Output paths
    # -------------------------------------------------------

    source_path = Path(source)

    if output_video is None:
        output_video_path = (
            DEFAULT_PIPELINE_OUTPUT_DIR
            / (
                source_path.stem
                + "_pipeline.mp4"
            )
        )
    else:
        output_video_path = Path(
            output_video
        )

    if metrics_dir is None:
        metrics_dir_path = (
            DEFAULT_METRICS_DIR
        )
    else:
        metrics_dir_path = Path(
            metrics_dir
        )

    # -------------------------------------------------------
    # Open output video writer
    # -------------------------------------------------------

    writer = open_writer(
        output_video_path,
        cap,
        fps,
    )

    print()
    print(
        "=== NovaRoute.AI ML Pipeline ==="
    )

    print(
        f"Source: {source}"
    )

    print(
        f"FPS: {fps:.2f}"
    )

    print(
        f"Observation window: "
        f"{counter.window_size / fps:.2f}s "
        f"({counter.window_size} frames)"
    )

    print(
        f"Emit interval: "
        f"{emit_every_frames / fps:.2f}s "
        f"({emit_every_frames} frames)"
    )

    print(
        f"Road capacity: "
        f"{road_capacity:.2f}"
    )

    print()

    # -------------------------------------------------------
    # Runtime state
    # -------------------------------------------------------

    frame_index = 0

    last_payload: dict | None = None

    latest_congestion = None
    latest_risk = None

    # -------------------------------------------------------
    # Process video
    # -------------------------------------------------------

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            frame_index += 1

            # -----------------------------------------------
            # ByteTrack
            #
            # VehicleTracker internally uses YOLO.
            # -----------------------------------------------

            tracked_vehicles = (
                tracker.track_frame(
                    frame,
                    frame_number=frame_index,
                )
            )

            # -----------------------------------------------
            # Vehicle count for current frame
            # -----------------------------------------------

            frame_count = (
                counter.count_frame(
                    tracked_vehicles
                )
            )

            # -----------------------------------------------
            # Add frame to rolling observation window
            # -----------------------------------------------

            counter.add_frame(
                frame_count
            )

            # -----------------------------------------------
            # Emit metrics every configured interval
            # -----------------------------------------------

            should_emit = (
                frame_index
                % emit_every_frames
                == 0
            )

            # Wait until the rolling window
            # has accumulated useful observations.
            window_ready = (
                len(counter.history)
                >= min(
                    counter.window_size,
                    emit_every_frames,
                )
            )

            if (
                should_emit
                and window_ready
            ):

                aggregate = (
                    counter.aggregate()
                )

                # -------------------------------------------
                # Congestion
                # -------------------------------------------

                latest_congestion = (
                    congestion_estimator.estimate(
                        total_vehicle_count=(
                            aggregate.total
                        ),
                        road_capacity=(
                            road_capacity
                        ),
                    )
                )

                # -------------------------------------------
                # Time of day
                # -------------------------------------------

                effective_hour = (
                    hour
                    if hour is not None
                    else datetime.now().hour
                )

                # -------------------------------------------
                # Risk
                # -------------------------------------------

                latest_risk = (
                    risk_scorer.score(
                        accident_count=(
                            accident_count
                        ),
                        traffic_density=(
                            latest_congestion
                            .traffic_density
                        ),
                        pedestrian_count=(
                            pedestrian_count
                        ),
                        hour=(
                            effective_hour
                        ),
                    )
                )

                # -------------------------------------------
                # JSON payload
                # -------------------------------------------

                observation_seconds = (
                    len(counter.history)
                    / fps
                )

                payload = {
                    "schema_version": "1.1.0",

                    "road_segment_id": (
                        location_id
                    ),

                    "osm_edge": None,

                    "timestamp": (
                        datetime.now(
                            timezone.utc
                        ).isoformat()
                    ),

                    "observation_window_seconds": (
                        round(
                            observation_seconds,
                            3,
                        )
                    ),

                    "vehicle_counts": {
                        "car": int(
                            round(
                                aggregate.car
                            )
                        ),
                        "motorcycle": int(
                            round(
                                aggregate.motorcycle
                            )
                        ),
                        "bus": int(
                            round(
                                aggregate.bus
                            )
                        ),
                        "truck": int(
                            round(
                                aggregate.truck
                            )
                        ),
                        "total": int(
                            round(
                                aggregate.total
                            )
                        ),
                    },

                    "total_vehicles": int(
                        round(
                            aggregate.total
                        )
                    ),

                    "peak_vehicles": int(
                        counter.peak_total()
                    ),

                    "road_capacity": float(
                        road_capacity
                    ),

                    # Kept for compatibility with the
                    # existing backend/schema.
                    #
                    # Semantically this is the concurrent
                    # vehicle/capacity ratio.
                    "traffic_density": float(
                        latest_congestion
                        .traffic_density
                    ),

                    "congestion_level": (
                        latest_congestion
                        .congestion_level
                    ),

                    "congestion_factor": float(
                        latest_congestion
                        .congestion_factor
                    ),

                    "aggregation_method": (
                        "average_active_vehicles"
                    ),

                    "risk_score": float(
                        latest_risk.risk_score
                    ),

                    "risk_level": (
                        latest_risk.risk_level
                    ),

                    "risk_factor_scores": (
                        make_json_safe(
                            latest_risk
                            .factor_scores
                        )
                    ),

                    "risk_contributions": (
                        make_json_safe(
                            latest_risk
                            .contributions
                        )
                    ),
                }

                save_metrics(
                    metrics_dir_path,
                    location_id,
                    payload,
                )

                last_payload = payload

            # -----------------------------------------------
            # Visualization
            # -----------------------------------------------

            annotated = (
                draw_tracked_vehicles(
                    frame.copy(),
                    tracked_vehicles,
                )
            )

            active_track_ids = {
                vehicle.track_id
                for vehicle
                in tracked_vehicles
                if vehicle.track_id
                is not None
            }

            lines = [
                "NovaRoute.AI",
                f"Frame: {frame_index}",
                (
                    f"Cars: "
                    f"{frame_count.car}"
                ),
                (
                    f"Motorcycles: "
                    f"{frame_count.motorcycle}"
                ),
                (
                    f"Buses: "
                    f"{frame_count.bus}"
                ),
                (
                    f"Trucks: "
                    f"{frame_count.truck}"
                ),
                (
                    f"Active vehicles: "
                    f"{frame_count.total}"
                ),
                (
                    f"Tracked IDs: "
                    f"{len(active_track_ids)}"
                ),
            ]

            if latest_congestion is not None:

                traffic_percent = (
                    latest_congestion
                    .traffic_density
                    * 100
                )

                lines.extend(
                    [
                        (
                            "Traffic Load: "
                            f"{traffic_percent:.1f}%"
                        ),
                        (
                            "Congestion: "
                            f"{latest_congestion.congestion_level}"
                        ),
                        (
                            "Congestion Factor: "
                            f"{latest_congestion.congestion_factor:.3f}"
                        ),
                    ]
                )

            if latest_risk is not None:

                lines.extend(
                    [
                        (
                            "Risk Score: "
                            f"{latest_risk.risk_score:.1f}"
                        ),
                        (
                            "Risk Level: "
                            f"{latest_risk.risk_level.upper()}"
                        ),
                    ]
                )

            annotated = draw_dashboard(
                annotated,
                lines,
            )

            # -----------------------------------------------
            # Save output video
            # -----------------------------------------------

            writer.write(
                annotated
            )

            # -----------------------------------------------
            # Optional live display
            # -----------------------------------------------

            if show:

                cv2.imshow(
                    "NovaRoute.AI",
                    annotated,
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key in (
                    ord("q"),
                    27,
                ):
                    print(
                        "Stopped by user."
                    )

                    break

    finally:

        cap.release()

        writer.release()

        if show:
            cv2.destroyAllWindows()

    # -------------------------------------------------------
    # Final output
    # -------------------------------------------------------

    print()

    print(
        "Pipeline processing completed."
    )

    print(
        f"Annotated video written to: "
        f"{output_video_path.resolve()}"
    )

    if last_payload is not None:

        print(
            f"Latest metrics written to: "
            f"{(
                metrics_dir_path
                / f'{location_id}_latest.json'
            ).resolve()}"
        )

        print()

        print(
            "=== Latest ML observation ==="
        )

        print(
            json.dumps(
                last_payload,
                indent=2,
            )
        )

    else:

        print(
            "Warning: no metrics were emitted. "
            "The video may be shorter than the "
            "configured emission interval."
        )

    return last_payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete NovaRoute.AI "
            "traffic ML pipeline."
        )
    )

    parser.add_argument(
        "--source",
        required=True,
        help=(
            "Path to input traffic video."
        ),
    )

    parser.add_argument(
        "--location-id",
        required=True,
        help=(
            "Road segment or junction ID."
        ),
    )

    parser.add_argument(
        "--road-capacity",
        required=True,
        type=float,
        help=(
            "Calibrated concurrent vehicle "
            "capacity for this road segment."
        ),
    )

    parser.add_argument(
        "--accident-count",
        type=float,
        default=0.0,
        help=(
            "Historical accident input used "
            "by the transparent risk scorer."
        ),
    )

    parser.add_argument(
        "--pedestrian-count",
        type=float,
        default=0.0,
        help=(
            "Pedestrian conflict input used "
            "by the transparent risk scorer."
        ),
    )

    parser.add_argument(
        "--hour",
        type=int,
        default=None,
        help=(
            "Hour 0-23 represented by the "
            "recorded video. If omitted, "
            "the current local hour is used."
        ),
    )

    parser.add_argument(
        "--config",
        default=str(
            DEFAULT_CONFIG_PATH
        ),
        help=(
            "Path to config.yaml."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output annotated "
            "video path."
        ),
    )

    parser.add_argument(
        "--metrics-dir",
        default=None,
        help=(
            "Optional metrics output "
            "directory."
        ),
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help=(
            "Disable the OpenCV preview "
            "window."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    run_pipeline(
        source=args.source,
        location_id=args.location_id,
        road_capacity=args.road_capacity,
        accident_count=args.accident_count,
        pedestrian_count=args.pedestrian_count,
        hour=args.hour,
        config_path=args.config,
        output_video=args.output,
        metrics_dir=args.metrics_dir,
        show=not args.no_display,
    )


if __name__ == "__main__":
    main()