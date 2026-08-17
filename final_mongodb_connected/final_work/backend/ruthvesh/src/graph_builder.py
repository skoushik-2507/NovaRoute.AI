"""
graph_builder.py

Downloads Nagpur's drivable road network using OSMnx, saves the raw graph,
then processes it (calculates and attaches base travel_time per edge) and
saves the processed graph separately for later routing.

Pipeline:
    1. Download raw graph from OSMnx        -> data/raw/nagpur_drive.graphml
    2. Calculate travel_time for every edge
    3. Validate travel_time coverage
    4. Save processed graph                  -> data/processed/nagpur_routing.graphml

This script does NOT implement routing (Dijkstra) or congestion logic.
"""

import logging
import sys
from pathlib import Path

import osmnx as ox

# Allow running this file both as part of the package (src.graph_builder)
# and directly as a script (python src/graph_builder.py).
try:
    from src.config import (
        PLACE_NAME,
        NETWORK_TYPE,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        RAW_GRAPH_PATH,
        PROCESSED_GRAPH_PATH,
        WEIGHT_ATTRIBUTE,
    )
    from src.travel_time import add_travel_times_to_graph
    from src.graph_utils import validate_graph_attributes
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.config import (
        PLACE_NAME,
        NETWORK_TYPE,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        RAW_GRAPH_PATH,
        PROCESSED_GRAPH_PATH,
        WEIGHT_ATTRIBUTE,
    )
    from src.travel_time import add_travel_times_to_graph
    from src.graph_utils import validate_graph_attributes

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def download_nagpur_graph(place_name: str = PLACE_NAME, network_type: str = NETWORK_TYPE):
    """
    Download the drivable road network for Nagpur using OSMnx.

    Parameters
    ----------
    place_name : str
        The place query passed to OSMnx (e.g. "Nagpur, Maharashtra, India").
    network_type : str
        The OSMnx network type to fetch (e.g. "drive").

    Returns
    -------
    networkx.MultiDiGraph
        The downloaded road network graph, with OSM attributes preserved
        on both nodes (e.g. x, y, osmid) and edges (e.g. highway, length,
        maxspeed, oneway).
    """
    logger.info("Download started for place: '%s' (network_type='%s')", place_name, network_type)

    try:
        graph = ox.graph_from_place(place_name, network_type=network_type)
    except Exception as exc:
        logger.error("Failed to download graph for '%s': %s", place_name, exc)
        raise

    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    logger.info("Download complete.")
    logger.info("Number of nodes: %d", num_nodes)
    logger.info("Number of edges: %d", num_edges)

    return graph


def save_graph(graph, save_path: Path):
    """
    Save a NetworkX/OSMnx graph to disk as a GraphML file.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The graph to save.
    save_path : Path
        Destination file path.
    """
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ox.save_graphml(graph, filepath=save_path)
    except Exception as exc:
        logger.error("Failed to save graph to '%s': %s", save_path, exc)
        raise

    logger.info("Graph saved to: %s", save_path)


def process_graph(graph):
    """
    Process the raw graph into a routing-ready graph.

    Steps:
    1. Calculate base travel_time for every edge (using travel_time.py).
    2. Validate that travel_time exists on the required edges.

    The input graph is not modified; a new processed graph is returned.
    Note: some edges may legitimately be missing travel_time if they lack
    valid length data (see travel_time.py's validation rules). This is
    reported, not silently ignored.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The raw graph downloaded from OSMnx.

    Returns
    -------
    networkx.MultiDiGraph
        The processed graph with a 'travel_time' attribute on every edge
        that could be validly calculated.
    """
    logger.info("Processing graph: calculating travel_time for all edges...")
    processed_graph = add_travel_times_to_graph(graph)

    total_edges = processed_graph.number_of_edges()
    edges_with_travel_time = sum(
        1 for _, _, data in processed_graph.edges(data=True)
        if WEIGHT_ATTRIBUTE in data
    )

    logger.info(
        "travel_time calculated for %d / %d edges.",
        edges_with_travel_time, total_edges,
    )

    logger.info("Validating processed graph attributes...")
    validation_report = validate_graph_attributes(
        processed_graph,
        required_node_attrs=("x", "y"),
        required_edge_attrs=(WEIGHT_ATTRIBUTE,),
    )

    if validation_report["is_valid"]:
        logger.info("Validation passed: all edges have '%s'.", WEIGHT_ATTRIBUTE)
    else:
        missing_count = validation_report["missing_edge_attrs"].get(WEIGHT_ATTRIBUTE, 0)
        logger.warning(
            "Validation found %d edge(s) missing '%s'. "
            "These edges should be reviewed before routing is implemented.",
            missing_count, WEIGHT_ATTRIBUTE,
        )

    return processed_graph


def main():
    """
    Entry point: download Nagpur's drivable road network, save the raw
    graph, process it (calculate + validate travel_time), and save the
    processed graph separately.

    Output structure:
        data/raw/nagpur_drive.graphml        -> unmodified raw graph
        data/processed/nagpur_routing.graphml -> graph with travel_time
    """
    logger.info("=== NovaRoute.AI - Ruthvesh module: graph_builder ===")

    try:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Download and save raw graph (kept unchanged thereafter)
        raw_graph = download_nagpur_graph()
        save_graph(raw_graph, RAW_GRAPH_PATH)

        # 2-4. Process (travel_time + validation) and save separately
        processed_graph = process_graph(raw_graph)
        save_graph(processed_graph, PROCESSED_GRAPH_PATH)

        logger.info("Graph build step finished successfully.")
        logger.info("Raw graph:       %s", RAW_GRAPH_PATH)
        logger.info("Processed graph: %s", PROCESSED_GRAPH_PATH)
    except Exception:
        logger.exception("Graph build step failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()