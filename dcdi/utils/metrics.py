import networkx as nx
from cdt.metrics import retrieve_adjacency_matrix



# def edge_errors(pred, target):
#     """
#     Counts all types of edge errors (false negatives, false positives, reversed edges)
#
#     Parameters:
#     -----------
#     pred: ndarray
#         The predicted adjacency matrix
#     target: ndarray
#         The true adjacency matrix
#
#     Returns:
#     --------
#     fn, fp, rev
#     """
#     true_labels = target
#     predictions = pred
#
#     diff = true_labels - predictions
#
#     # Reversed edges: An edge exists in both directions in pred but only one in target
#     reversed_edges = (((diff + diff.T) == 0) & (diff != 0)).sum() / 2
#
#     # False negatives: Missing edges in pred
#     fn = (diff == 1).sum() - reversed_edges
#
#     # False positives: Extra edges in pred
#     fp = (diff == -1).sum() - reversed_edges
#
#     return int(fn), int(fp), int(reversed_edges)

def edge_errors(pred, target):
    """
    Counts all types of edge errors (false negatives, false positives, reversed edges)

    Parameters:
    -----------
    pred: nx.DiGraph or ndarray
        The predicted adjacency matrix
    target: nx.DiGraph or ndarray
        The true adjacency matrix

    Returns:
    --------
    fn, fp, rev

    """
    true_labels = retrieve_adjacency_matrix(target)
    predictions = retrieve_adjacency_matrix(pred, target.nodes() if isinstance(target, nx.DiGraph) else None)

    diff = true_labels - predictions

    rev = (((diff + diff.transpose()) == 0) & (diff != 0)).sum() / 2
    # Each reversed edge necessarily leads to one fp and one fn so we need to subtract those
    fn = (diff == 1).sum() - rev
    fp = (diff == -1).sum() - rev

    return fn, fp, rev


# def edge_accurate(pred, target):
#     """
#     Counts the number of edge in ground truth DAG, true positives and the true
#     negatives
#
#     Parameters:
#     -----------
#     pred: nx.DiGraph or ndarray
#         The predicted adjacency matrix
#     target: nx.DiGraph or ndarray
#         The true adjacency matrix
#
#     Returns:
#     --------
#     total_edges, tp, tn
#
#     """
#     true_labels = retrieve_adjacency_matrix(target)
#     predictions = retrieve_adjacency_matrix(pred, target.nodes() if isinstance(target, nx.DiGraph) else None)
#
#     total_edges = (true_labels).sum()
#
#     tp = ((predictions == 1) & (predictions == true_labels)).sum()
#     tn = ((predictions == 0) & (predictions == true_labels)).sum()
#
#     return total_edges, tp, tn


def shd(pred, target):
    """
    Calculates the structural hamming distance

    Parameters:
    -----------
    pred: nx.DiGraph or ndarray
        The predicted adjacency matrix
    target: nx.DiGraph or ndarray
        The true adjacency matrix

    Returns:
    --------
    shd

    """
    return sum(edge_errors(pred, target))


def compute_shd(pred, target):
    """
    Compute the Structural Hamming Distance (SHD) between two DAGs.

    SHD counts the number of edge additions, deletions, or flips required to convert
    the predicted graph into the target graph.
    """

    return shd(pred, target)


def to_cpdag(graph):
    """
    Convert a DAG to its CPDAG representation.

    Note: Implementing a full CPDAG conversion is non-trivial.
    This function uses NetworkX's built-in functionality where possible.
    """
    # Check if the graph is a DAG
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Graph must be a DAG to convert to CPDAG.")

    # Skeleton of the graph
    skeleton = graph.to_undirected()

    # Find the v-structures
    v_structures = set()
    for triplet in nx.triangles(skeleton):
        # Implement v-structure detection if needed
        pass  # Placeholder for v-structure logic

    # Note: NetworkX does not have a built-in CPDAG converter.
    # You may need to use other libraries or implement the conversion.
    # Here, we'll return the skeleton as a placeholder.
    return skeleton


def compute_shd_cpdag(pred, target):
    """
    Compute the SHD for CPDAGs (Completed Partially Directed Acyclic Graphs).

    This requires converting both graphs to their CPDAG representations.
    """
    pred_cpdag = to_cpdag(pred)
    target_cpdag = to_cpdag(target)
    return compute_shd(pred_cpdag, target_cpdag)


def compute_sid(pred, target):
    """
    Compute the Structural Intervention Distance (SID) between two DAGs.

    SID measures the number of intervention distributions that differ between
    the predicted and target causal graphs.

    Note: A full implementation of SID is complex and typically requires
    enumeration of all possible interventions. Below is a simplified version.
    """
    # Placeholder implementation
    # For a complete SID computation, consider using specialized libraries or algorithms
    # Here, we'll approximate SID using SHD as a proxy
    return compute_shd(pred, target)


