import os
import pickle

# Load gt-adjacency.pkl (Ground Truth Adjacency)
with open("exp/gt-adjacency.pkl", "rb") as f:
    gt_adjacency = pickle.load(f)

# Load opt.pkl (Optimization Result)
with open("exp/opt.pkl", "rb") as f:
    opt = pickle.load(f)

print("Ground Truth Adjacency Matrix:")
print(gt_adjacency)

print("\nOptimization Result:")
print(opt)

import matplotlib
matplotlib.use('TkAgg')  # Use 'TkAgg' for interactive plotting, or 'Agg' for non-interactive plotting
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_adjacency_matrix(adjacency_matrix, title="Graph", filename=None):
    """Plot a directed graph from an adjacency matrix."""
    G = nx.DiGraph()
    num_nodes = adjacency_matrix.shape[0]

    # Add nodes and edges based on adjacency matrix
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adjacency_matrix[i][j] != 0:
                G.add_edge(i, j)

    # Draw the graph
    plt.figure(figsize=(8, 6))
    pos = nx.circular_layout(G)  # Layout for better visualization
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=700, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20)
    plt.title(title)

    # Save the plot to a file if a filename is provided
    if filename:
        plt.savefig(filename)
    plt.show()

# Paths to initial graph and intervention files
initial_graph_path = "./data/perfect/data_p10_e10_n10000_linear_struct/DAG1.npy"  # Update this path as needed
intervention_graph_path = "./data/perfect/data_p10_e10_n10000_linear_struct/data_interv1.npy"  # If interventions are saved separately

# Load the initial graph (adjacency matrix)
if os.path.exists(initial_graph_path):
    initial_adjacency = np.load(initial_graph_path)
    plot_adjacency_matrix(initial_adjacency, title="Initial Graph", filename="initial_graph.png")
else:
    print(f"File not found: {initial_graph_path}")

# Load and visualize intervention graph if available
if os.path.exists(intervention_graph_path):
    intervention_adjacency = np.load(intervention_graph_path)
    plot_adjacency_matrix(intervention_adjacency, title="Intervention Graph", filename="intervention_graph.png")
else:
    print(f"File not found: {intervention_graph_path}")

# Plot Ground Truth Graph
plot_adjacency_matrix(np.array(gt_adjacency), title="Ground Truth Graph")

# Plot Optimization Result Graph
if 'adjacency_matrix' in opt:
    plot_adjacency_matrix(np.array(opt['adjacency_matrix']), title="Learned Graph (Optimization Result)")
else:
    print("Optimization result does not contain an adjacency matrix.")

