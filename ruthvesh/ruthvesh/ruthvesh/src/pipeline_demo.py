"""
pipeline_demo.py

End-to-end demonstration of the NovaRoute.AI Ruthvesh routing pipeline,
using the real, already-built Nagpur road network.

Pipeline demonstrated (matches the concept note's steps 1, 2, and 6):
    1. Load the processed Nagpur road graph (graph_builder.py's output).
    2. Use the travel_time edge weights already attached to that graph
       (travel_time.py's output) — the base, free-flow response-time
       metric.
    3. Define sample officer locations (hardcoded below).
    4. Define sample high-risk junction locations (hardcoded below).
    5. Snap each officer/junction to its nearest graph node
       (graph_utils.find_nearest_node).
    6. Calculate officer -> junction Dijkstra response times
       (routing.shortest_path, via cost_matrix.py).
    7. Assemble the full officer x junction response-time cost matrix
       (cost_matrix.build_cost_matrix).
    8. Identify under-covered ("unmanned") high-risk junctions
       (coverage.analyze_coverage).
    9. Print all of the above clearly, including an explicit contrast
       between straight-line ("as the crow flies") distance and the
       actual road-network Dijkstra route — to make it obvious this
       system is not just using straight-line distance.

This is a DEMONSTRATION script only. It does NOT implement the officer
allocation optimizer, React, FastAPI, or YOLO — those are separate
modules / separate team members' work.

Run with:
    python src/pipeline_demo.py
(from the ruthvesh/ directory), or:
    python -m src.pipeline_demo
"""

import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    from src.config import PROCESSED_GRAPH_PATH, WEIGHT_ATTRIBUTE
    from src.graph_utils import load_graph, find_nearest_node
    from src.routing import shortest_path
    from src.cost_matrix import build_cost_matrix, to_dataframe
    from src.coverage import analyze_coverage, DEFAULT_THRESHOLD_MINUTES
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.config import PROCESSED_GRAPH_PATH, WEIGHT_ATTRIBUTE
    from src.graph_utils import load_graph, find_nearest_node
    from src.routing import shortest_path
    from src.cost_matrix import build_cost_matrix, to_dataframe
    from src.coverage import analyze_coverage, DEFAULT_THRESHOLD_MINUTES

logging.basicConfig(
    level=logging.WARNING,  # keep the demo's own print()s front and center
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 3. Sample officer locations (hardcoded, real coordinates within Nagpur)
# ---------------------------------------------------------------------------

SAMPLE_OFFICERS: List[Dict[str, Any]] = [
    {"id": "officer_1", "latitude": 21.1458, "longitude": 79.0882},  # near Sitabuldi
    {"id": "officer_2", "latitude": 21.1300, "longitude": 79.0700},  # near Wardha Rd side
    {"id": "officer_3", "latitude": 21.1700, "longitude": 79.1100},  # near Kamptee Rd side
]

# ---------------------------------------------------------------------------
# 4. Sample high-risk junction locations (hardcoded, real coordinates)
# ---------------------------------------------------------------------------

SAMPLE_HIGH_RISK_JUNCTIONS: List[Dict[str, Any]] = [
    {"id": "junction_1", "latitude": 21.1600, "longitude": 79.1000, "risk_score": 0.90},
    {"id": "junction_2", "latitude": 21.1200, "longitude": 79.0600, "risk_score": 0.75},
    {"id": "junction_3", "latitude": 21.2200, "longitude": 79.2200, "risk_score": 0.60},  # far -> demo an uncovered junction
]

# Response-time threshold for the coverage demo. Matches the concept
# note's initial demonstration value (coverage.DEFAULT_THRESHOLD_MINUTES).
DEMO_THRESHOLD_MINUTES = DEFAULT_THRESHOLD_MINUTES


# ---------------------------------------------------------------------------
# Straight-line distance, for CONTRAST ONLY.
#
# This is never used for routing, response-time calculation, or
# coverage decisions anywhere in NovaRoute.AI (see routing.py,
# cost_matrix.py, coverage.py — all of them route exclusively via
# Dijkstra over real road-network travel time). It exists only here, in
# the demo, to make the difference between "as the crow flies" and
# "actual road-network response time" visible and concrete.
# ---------------------------------------------------------------------------

def _straight_line_distance_meters(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """
    Great-circle (haversine) distance between two lat/lon points, in
    meters. FOR DEMONSTRATION/CONTRAST ONLY — see module note above.
    """
    earth_radius_m = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return earth_radius_m * c


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

def _print_section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _print_subsection(title: str) -> None:
    print("\n--- " + title + " ---")


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_1_2_load_graph_with_travel_times():
    """
    Steps 1 & 2: Load the processed Nagpur road graph, which already has
    base travel_time attached to every edge (travel_time.py's output,
    via graph_builder.py).
    """
    _print_section("STEP 1-2: Load processed Nagpur road graph + travel_time weights")

    try:
        graph = load_graph(PROCESSED_GRAPH_PATH)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        print("Run graph_builder.py first to build the processed graph.")
        sys.exit(1)

    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    edges_with_travel_time = sum(
        1 for _, _, data in graph.edges(data=True) if WEIGHT_ATTRIBUTE in data
    )

    print(f"Graph loaded from: {PROCESSED_GRAPH_PATH}")
    print(f"Nodes (intersections): {num_nodes}")
    print(f"Edges (road segments):  {num_edges}")
    print(
        f"Edges with '{WEIGHT_ATTRIBUTE}' (base response-time weight): "
        f"{edges_with_travel_time} / {num_edges}"
    )
    print(
        "This confirms every routing decision below will be made over "
        "real road segments with real travel-time weights, not a "
        "simplified/abstract graph."
    )

    return graph


def step_3_4_print_sample_locations():
    """
    Steps 3 & 4: Define (print) the hardcoded sample officer and
    high-risk junction locations used throughout this demo.
    """
    _print_section("STEP 3-4: Sample officer & high-risk junction locations")

    _print_subsection("Officers")
    for officer in SAMPLE_OFFICERS:
        print(
            f"  {officer['id']:<12} lat={officer['latitude']:.4f}, "
            f"lon={officer['longitude']:.4f}"
        )

    _print_subsection("High-risk junctions")
    for junction in SAMPLE_HIGH_RISK_JUNCTIONS:
        print(
            f"  {junction['id']:<12} lat={junction['latitude']:.4f}, "
            f"lon={junction['longitude']:.4f}, risk_score={junction['risk_score']}"
        )


def step_5_snap_to_nearest_nodes(graph):
    """
    Step 5: Snap every officer and junction coordinate to its nearest
    graph node, printing the result.
    """
    _print_section("STEP 5: Snap officers & junctions to nearest graph nodes")

    _print_subsection("Officer -> nearest graph node")
    for officer in SAMPLE_OFFICERS:
        node = find_nearest_node(graph, officer["latitude"], officer["longitude"])
        print(f"  {officer['id']:<12} -> node {node}")

    _print_subsection("Junction -> nearest graph node")
    for junction in SAMPLE_HIGH_RISK_JUNCTIONS:
        node = find_nearest_node(graph, junction["latitude"], junction["longitude"])
        print(f"  {junction['id']:<12} -> node {node}")


def step_6_dijkstra_vs_straight_line(graph):
    """
    Step 6 (+ demonstration goal): Run one explicit Dijkstra
    officer -> junction query, and print it side-by-side with the
    straight-line distance between the same two points — to make the
    real-road-network-vs-straight-line contrast concrete and visible.
    """
    _print_section(
        "STEP 6: Dijkstra response time vs. straight-line distance (contrast)"
    )

    officer = SAMPLE_OFFICERS[0]
    junction = SAMPLE_HIGH_RISK_JUNCTIONS[0]

    straight_line_m = _straight_line_distance_meters(
        officer["latitude"], officer["longitude"],
        junction["latitude"], junction["longitude"],
    )

    result = shortest_path(
        graph,
        find_nearest_node(graph, officer["latitude"], officer["longitude"]),
        find_nearest_node(graph, junction["latitude"], junction["longitude"]),
    )

    print(f"Officer:  {officer['id']} ({officer['latitude']:.4f}, {officer['longitude']:.4f})")
    print(f"Junction: {junction['id']} ({junction['latitude']:.4f}, {junction['longitude']:.4f})")
    print()
    print(
        f"  Straight-line ('as the crow flies') distance: "
        f"{straight_line_m:,.0f} m  (NOT used anywhere in NovaRoute.AI's "
        f"routing/coverage logic — shown for contrast only)"
    )

    if result["is_reachable"]:
        route_length = len(result["route_nodes"])
        detour_ratio = (
            result["total_distance_meters"] / straight_line_m
            if straight_line_m > 0 else float("inf")
        )
        print(
            f"  Actual Dijkstra road-network distance:         "
            f"{result['total_distance_meters']:,.0f} m "
            f"({route_length} nodes along the route)"
        )
        print(
            f"  Actual Dijkstra road-network response time:    "
            f"{result['total_time_seconds'] / 60.0:.2f} minutes"
        )
        print(
            f"  Detour ratio (road distance / straight-line):  "
            f"{detour_ratio:.2f}x"
        )
        print(
            "\n  This gap between straight-line distance and the actual "
            "routed distance/time is exactly why NovaRoute.AI computes "
            "response time via Dijkstra over real road geometry — a "
            "radius or straight-line estimate would misjudge which "
            "officer is actually fastest whenever one-way streets, the "
            "Nag River, or missing direct roads are in the way."
        )
    else:
        print(f"  No road-network route found: {result['error']}")


def step_7_build_cost_matrix(graph):
    """
    Step 7: Build the full officer x junction response-time cost
    matrix.
    """
    _print_section("STEP 7: Officer x junction response-time cost matrix")

    cost_matrix_result = build_cost_matrix(
        graph, SAMPLE_OFFICERS, SAMPLE_HIGH_RISK_JUNCTIONS, time_unit="minutes",
    )

    print(
        f"weight_mode='{cost_matrix_result['weight_mode']}' "
        f"(edge attribute: '{cost_matrix_result['weight_attribute']}'), "
        f"time_unit='{cost_matrix_result['time_unit']}'"
    )
    print()
    print(to_dataframe(cost_matrix_result).round(2).to_string())

    if cost_matrix_result["unreachable_pairs"]:
        print(
            f"\nUnreachable pairs: {cost_matrix_result['unreachable_pairs']}"
        )

    return cost_matrix_result


def step_8_identify_undercovered_junctions(graph):
    """
    Step 8: Run coverage analysis to identify under-covered ("unmanned")
    high-risk junctions given the sample officers and the demo
    threshold.
    """
    _print_section(
        f"STEP 8: Coverage analysis (threshold = {DEMO_THRESHOLD_MINUTES:.1f} min)"
    )

    coverage_result = analyze_coverage(
        graph, SAMPLE_OFFICERS, SAMPLE_HIGH_RISK_JUNCTIONS,
        threshold_minutes=DEMO_THRESHOLD_MINUTES,
    )

    for report in coverage_result["junctions"]:
        status = "COVERED" if report["is_covered"] else "UNCOVERED"
        min_time = report["min_response_time_minutes"]
        min_time_str = f"{min_time:.2f} min" if min_time is not None else "unreachable"
        officer = report["nearest_officer_id"] or "none"

        print(
            f"  [{status:<9}] {report['junction_id']:<12} "
            f"risk_score={report['risk_score']:<5} "
            f"nearest_officer={officer:<12} "
            f"min_response_time={min_time_str}"
        )

    print()
    print(
        f"Coverage summary: {coverage_result['num_covered']} covered, "
        f"{coverage_result['num_uncovered']} uncovered "
        f"({coverage_result['coverage_rate'] * 100:.1f}% coverage)."
    )

    if coverage_result["uncovered_junction_ids"]:
        print(
            "\nUnder-covered high-risk junctions (highest risk first):"
        )
        for junction_id in coverage_result["uncovered_junction_ids"]:
            report = next(
                r for r in coverage_result["junctions"] if r["junction_id"] == junction_id
            )
            print(
                f"  - {junction_id} (risk_score={report['risk_score']}): "
                f"no officer can respond within "
                f"{DEMO_THRESHOLD_MINUTES:.1f} min "
                f"(fastest is {report['min_response_time_minutes']:.2f} min "
                f"via {report['nearest_officer_id']})"
                if report["min_response_time_minutes"] is not None
                else f"  - {junction_id} (risk_score={report['risk_score']}): "
                     f"NOT REACHABLE by any officer at all."
            )
    else:
        print("\nAll high-risk junctions are covered within the threshold.")

    return coverage_result


# ---------------------------------------------------------------------------
# Step 9: main() ties every step together and prints a final summary
# ---------------------------------------------------------------------------

def main():
    print("#" * 78)
    print("# NovaRoute.AI - Ruthvesh module: full pipeline demo")
    print("# Graph -> travel_time -> Dijkstra -> cost matrix -> coverage")
    print("#" * 78)

    graph = step_1_2_load_graph_with_travel_times()
    step_3_4_print_sample_locations()
    step_5_snap_to_nearest_nodes(graph)
    step_6_dijkstra_vs_straight_line(graph)
    cost_matrix_result = step_7_build_cost_matrix(graph)
    coverage_result = step_8_identify_undercovered_junctions(graph)

    # Step 9: final summary
    _print_section("STEP 9: Summary")
    print(
        f"Routed {len(SAMPLE_OFFICERS)} officer(s) x "
        f"{len(SAMPLE_HIGH_RISK_JUNCTIONS)} high-risk junction(s) using "
        f"Dijkstra over {graph.number_of_edges()} real road segments."
    )
    print(
        f"Every response time above came from actual road-network "
        f"travel time (weight_mode='{cost_matrix_result['weight_mode']}'), "
        f"never straight-line distance."
    )
    print(
        f"Result: {coverage_result['num_covered']} / "
        f"{len(SAMPLE_HIGH_RISK_JUNCTIONS)} high-risk junctions are "
        f"covered within {DEMO_THRESHOLD_MINUTES:.1f} minutes; "
        f"{coverage_result['num_uncovered']} need attention."
    )
    print(
        "\n(Officer allocation optimization, the live dashboard, and "
        "vehicle-detection-based congestion are handled by other "
        "modules — this demo covers the Dijkstra/response-time "
        "pipeline only.)"
    )


if __name__ == "__main__":
    main()