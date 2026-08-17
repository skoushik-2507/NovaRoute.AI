"""
demo_routing.py

Simple command-line demonstration of the NovaRoute.AI routing system.

This script:
1. Loads the processed Nagpur road graph.
2. Uses two example lat/lon locations (defined below).
3. Finds the nearest graph node for each location.
4. Runs Dijkstra-based shortest travel-time routing between them.
5. Prints a summary of the result.
6. Plots the route over the Nagpur road network, if a route was found.

This is a DEMONSTRATION script only — not part of the reusable module
API. No FastAPI, React, YOLO, or database code is included here.
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import osmnx as ox

try:
    from src.config import PROCESSED_GRAPH_PATH, WEIGHT_ATTRIBUTE
    from src.graph_utils import load_graph
    from src.routing import route_between_coordinates
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.config import PROCESSED_GRAPH_PATH, WEIGHT_ATTRIBUTE
    from src.graph_utils import load_graph
    from src.routing import route_between_coordinates

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Demo locations — edit these to try different origin/destination pairs.
# Coordinates are (latitude, longitude), both within Nagpur.
# ---------------------------------------------------------------------------

ORIGIN_LAT = 21.1458
ORIGIN_LON = 79.0882

DESTINATION_LAT = 21.1600
DESTINATION_LON = 79.1000


def print_route_summary(result: dict) -> None:
    """
    Print a human-readable summary of a routing result.

    Parameters
    ----------
    result : dict
        The dict returned by route_between_coordinates() /
        shortest_path(), containing route/timing/distance info or an
        error.
    """
    print("\n" + "=" * 50)
    print("NovaRoute.AI - Routing Demo Result")
    print("=" * 50)

    print(f"Origin node:      {result.get('origin_node')}")
    print(f"Destination node: {result.get('destination_node')}")

    if not result["is_reachable"]:
        print(f"Route status:     NOT REACHABLE")
        print(f"Reason:           {result.get('error')}")
        print("=" * 50 + "\n")
        return

    total_distance_m = result["total_distance_meters"]
    total_time_s = result["total_time_seconds"]
    num_route_nodes = len(result["route_nodes"])

    print(f"Route status:     FOUND")
    print(f"Total distance:   {total_distance_m:.1f} m ({total_distance_m / 1000:.2f} km)")
    print(f"Estimated time:   {total_time_s:.1f} s ({total_time_s / 60:.2f} min)")
    print(f"Nodes in route:   {num_route_nodes}")
    print("=" * 50 + "\n")


def plot_route(graph, route_nodes) -> None:
    """
    Plot the selected route over the full Nagpur road network.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The full road network graph the route was computed on.
    route_nodes : list
        Ordered list of node IDs representing the route to highlight.
    """
    try:
        fig, ax = ox.plot_graph_route(
            graph,
            route_nodes,
            route_color="red",
            route_linewidth=3,
            node_size=0,
            edge_linewidth=0.5,
            bgcolor="white",
            edge_color="gray",
            show=False,
            close=False,
        )
        ax.set_title("NovaRoute.AI - Nagpur Route Demo")
        plt.show()
    except Exception as exc:
        logger.warning("Could not plot route: %s", exc)


def main():
    """
    Entry point: load the graph, run a demo route between the fixed
    origin/destination coordinates defined at the top of this file, and
    print + plot the result.
    """
    logger.info("=== NovaRoute.AI - Routing Demo ===")

    try:
        graph = load_graph(PROCESSED_GRAPH_PATH)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    logger.info(
        "Finding route from (%.4f, %.4f) to (%.4f, %.4f)...",
        ORIGIN_LAT, ORIGIN_LON, DESTINATION_LAT, DESTINATION_LON,
    )

    result = route_between_coordinates(
        graph,
        origin_lat=ORIGIN_LAT,
        origin_lon=ORIGIN_LON,
        destination_lat=DESTINATION_LAT,
        destination_lon=DESTINATION_LON,
        weight_attribute=WEIGHT_ATTRIBUTE,
    )

    print_route_summary(result)

    if result["is_reachable"]:
        logger.info("Plotting route...")
        plot_route(graph, result["route_nodes"])
    else:
        logger.info("Skipping plot since no route was found.")


if __name__ == "__main__":
    main()