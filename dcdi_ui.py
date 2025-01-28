import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import os
import subprocess
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
from sklearn.feature_selection import mutual_info_regression
# Get the path of the current Python interpreter
python_executable = sys.executable
matplotlib.use("TkAgg")  # Set the backend for Tkinter compatibility
import matplotlib.pyplot as plt
import torch
import threading
import pickle
import seaborn as sns
import json

# --------------------------------------
# Default parameters
# --------------------------------------
DEFAULT_PARAMS = {
    # Entry-based parameters (key must match label exactly if used in entry_widgets)
    "Data Path:": "data_p10_e10_n10000_linear_struct",
    "Number of Variables:": "10",
    "Model (DCDI-G or DCDI-DSF):": "DCDI-DSF",
    "Experiment Path:": "exp",
    "Dataset Index:": "1",
    "Number of Training Iterations:": "1000",
    "Training Batch Size:": "64",
    "DAG for Retraining (path):": "",
    "Random Seed:": "42",
    "Train Samples (e.g., 0.8 for 80%)": "0.8",
    "Test Samples (leave blank for default):": "",
    "Number of Folds:": "5",
    "Fold for Testing:": "0",
    "Number of Layers:": "2",
    "Hidden Dimension:": "16",
    "Flow Number of Layers:": "2",
    "Flow Hidden Dimension:": "16",
    "Coefficient Intervention Sparsity:": "1e-8",
    "Learning Rate:": "1e-3",
    "Learning Rate Reinit (optional):": "",
    "Learning Rate Schedule (optional):": "",
    "Stop Criterion Window:": "100",
    "Regularization Coefficient:": "0.1",
    "Omega Gamma:": "1e-4",
    "Omega Mu:": "0.9",
    "Mu Init:": "1e-8",
    "Mu Mult Factor:": "2",
    "Gamma Init:": "1e-8",
    "H Threshold:": "1e-8",
    "Patience:": "10",
    "Train Patience:": "5",
    "Train Patience Post:": "5",
    "Plot Frequency:": "10000",

    # Non-entry parameters (checkboxes, radio buttons, etc.)
    "intervention_type_var": "perfect",    # or "imperfect"
    "intervention_knowledge_var": "known", # or "unknown"
    "intervention_var": False,             # Use Intervention
    "dcd_var": False,                      # Use DCD
    "normalize_data_var": False,           # Normalize Data
    "retrain_var": False,                  # Retrain
    "no_w_adjs_log_var": False,            # No W Adjs Log
    "plot_density_var": False,             # Plot Density
    "float_var": False,                    # Use Float Precision
    "test_on_new_regimes_var": False,      # Test on New Regimes
    "nonlin_var": "leaky-relu",            # "leaky-relu" or "sigmoid"
    "optimizer_var": "rmsprop",            # "sgd" or "rmsprop"
    "regimes_to_ignore": ""                # space-separated ints
}

# Parameter explanations dictionary
PARAMETER_EXPLANATIONS = {
    "Data Path:": "Path to the dataset directory relative to the data folder. Example: 'data_p10_e10_n10000_linear_struct'.",
    "Number of Variables:": "The number of variables/features in your dataset.",
    "Model (DCDI-G or DCDI-DSF):": "Choose between 'DCDI-G' or 'DCDI-DSF' models for causal discovery.",
    "Experiment Path:": "Directory where experiment results and logs will be saved.",
    "Dataset Index:": "Index of the dataset to use, especially useful when multiple datasets are available.",
    "Number of Training Iterations:": "Total number of training iterations for the model.",
    "Training Batch Size:": "Number of samples per batch during training.",
    "DAG for Retraining (path):": "Path to a Directed Acyclic Graph (DAG) in .npy format for retraining purposes.",
    "Random Seed:": "Seed value to ensure reproducibility of results.",
    "Train Samples (e.g., 0.8 for 80%)": "Proportion of the dataset to be used for training (between 0 and 1).",
    "Test Samples (leave blank for default):": "Number of samples to be used for testing. If left blank, it defaults to the remaining samples not used for training.",
    "Number of Folds:": "Number of folds for cross-validation.",
    "Fold for Testing:": "Specific fold index to be used for testing in cross-validation.",
    "Number of Layers:": "Number of hidden layers in the neural network model.",
    "Hidden Dimension:": "Number of neurons in each hidden layer.",
    "Flow Number of Layers:": "Number of layers in the normalizing flow component of the model.",
    "Flow Hidden Dimension:": "Number of neurons in each layer of the normalizing flow.",
    "Coefficient Intervention Sparsity:": "Regularization coefficient to enforce sparsity in intervention coefficients.",
    "Learning Rate:": "Step size for the optimizer during training.",
    "Learning Rate Reinit (optional):": "Learning rate for the optimizer after the first subproblem. Defaults to the initial learning rate if left blank.",
    "Learning Rate Schedule (optional):": "Strategy for adjusting the learning rate during training. Options might include 'None', 'sqrt-mu', or 'log-mu'.",
    "Stop Criterion Window:": "Window size to compute the stopping criterion for early termination of training.",
    "Regularization Coefficient:": "Regularization parameter (lambda) to prevent overfitting.",
    "Omega Gamma:": "Precision parameter to declare convergence of subproblems in the augmented Lagrangian method.",
    "Omega Mu:": "Factor by which the constraint should reduce after a subproblem is solved.",
    "Mu Init:": "Initial value of the Lagrangian multiplier mu.",
    "Mu Mult Factor:": "Factor to multiply mu by when the constraint is not sufficiently decreasing.",
    "Gamma Init:": "Initial value of the Lagrangian multiplier gamma.",
    "H Threshold:": "Threshold for the constraint violation. Training stops when |h| < threshold. A value of 0 means training stops only when h == 0.",
    "Patience:": "Number of iterations to wait for improvement before early stopping during retraining.",
    "Train Patience:": "Early stopping patience during training after a constraint is applied.",
    "Train Patience Post:": "Early stopping patience during training after reaching a certain threshold.",
    "Plot Frequency:": "Frequency (in iterations) at which plots are generated during training.",
}

# --------------------------------------
# Utility / Helper Functions
# --------------------------------------
def show_info(parameter_label):
    """Display parameter explanation in a popup."""
    explanation = PARAMETER_EXPLANATIONS.get(parameter_label, "No explanation available for this parameter.")
    messagebox.showinfo(title=f"Info: {parameter_label}", message=explanation)

def is_gpu_available():
    """Check if GPU is available via torch."""
    return torch.cuda.is_available()

def create_labeled_entry(parent, label_text, default_value):
    """
    Create a frame containing:
    - Info button (ℹ️)
    - Label
    - Entry with default value
    """
    frame = tk.Frame(parent)

    # Info button
    info_button = tk.Button(frame, text="ℹ️", command=lambda: show_info(label_text), width=2)
    info_button.pack(side=tk.LEFT, padx=(0, 5))

    # Label
    label = tk.Label(frame, text=label_text)
    label.pack(side=tk.LEFT)

    # Entry
    entry = tk.Entry(frame, width=30)
    entry.insert(0, default_value)
    entry.pack(side=tk.LEFT, padx=(10, 0))

    return frame, entry

# --------------------------------------
# Functions to set and load parameters
# --------------------------------------
def set_params(params_dict):
    """
    Set UI fields according to the values in params_dict.
    If a key doesn't exist, it retains its previous (default) value.
    """
    # For labeled entry widgets
    for label, widget in entry_widgets.items():
        if label in params_dict:
            widget.delete(0, tk.END)
            widget.insert(0, str(params_dict[label]))

    # For radiobuttons
    if "intervention_type_var" in params_dict:
        intervention_type_var.set(params_dict["intervention_type_var"])
    if "intervention_knowledge_var" in params_dict:
        intervention_knowledge_var.set(params_dict["intervention_knowledge_var"])

    # For checkboxes
    if "intervention_var" in params_dict:
        intervention_var.set(params_dict["intervention_var"])
    if "dcd_var" in params_dict:
        dcd_var.set(params_dict["dcd_var"])
    if "normalize_data_var" in params_dict:
        normalize_data_var.set(params_dict["normalize_data_var"])
    if "retrain_var" in params_dict:
        retrain_var.set(params_dict["retrain_var"])
    if "no_w_adjs_log_var" in params_dict:
        no_w_adjs_log_var.set(params_dict["no_w_adjs_log_var"])
    if "plot_density_var" in params_dict:
        plot_density_var.set(params_dict["plot_density_var"])
    if "float_var" in params_dict:
        float_var.set(params_dict["float_var"])
    if "test_on_new_regimes_var" in params_dict:
        test_on_new_regimes_var.set(params_dict["test_on_new_regimes_var"])

    # For other dropdowns/option menus
    if "nonlin_var" in params_dict:
        nonlin_var.set(params_dict["nonlin_var"])
    if "optimizer_var" in params_dict:
        optimizer_var.set(params_dict["optimizer_var"])

    # For regimes to ignore
    if "regimes_to_ignore" in params_dict:
        regimes_to_ignore_entry.delete(0, tk.END)
        regimes_to_ignore_entry.insert(0, params_dict["regimes_to_ignore"])

def load_param_file(*args):
    """
    Callback for the param_file_var dropdown. Loads parameters from the selected file
    or sets them to default if "default" is selected.
    If file-based parameters are missing any keys, default values are used for those.
    """
    selected_file = param_file_var.get()
    if selected_file == "default":
        # Restore default params
        set_params(DEFAULT_PARAMS)
    else:
        # Attempt to load from the JSON file in the "params" folder
        filepath = os.path.join("params", selected_file)
        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    file_params = json.load(f)
                # Merge file_params onto DEFAULT_PARAMS
                merged_params = {**DEFAULT_PARAMS, **file_params}
                set_params(merged_params)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load parameters from {selected_file}:\n{e}")
                # If error occurs, revert to default
                set_params(DEFAULT_PARAMS)
        else:
            messagebox.showerror("Error", f"File not found: {filepath}")
            set_params(DEFAULT_PARAMS)

def save_params():
    """
    Save the current parameters to a JSON file in the 'params' folder.
    Prompts the user to enter a filename.
    """
    # Gather current parameters
    current_params = {}

    # Collect entry-based parameters
    for label, widget in entry_widgets.items():
        current_params[label] = widget.get()

    # Collect radiobutton selections
    current_params["intervention_type_var"] = intervention_type_var.get()
    current_params["intervention_knowledge_var"] = intervention_knowledge_var.get()

    # Collect checkbox states
    current_params["intervention_var"] = intervention_var.get()
    current_params["dcd_var"] = dcd_var.get()
    current_params["normalize_data_var"] = normalize_data_var.get()
    current_params["retrain_var"] = retrain_var.get()
    current_params["no_w_adjs_log_var"] = no_w_adjs_log_var.get()
    current_params["plot_density_var"] = plot_density_var.get()
    current_params["float_var"] = float_var.get()
    current_params["test_on_new_regimes_var"] = test_on_new_regimes_var.get()

    # Collect dropdown selections
    current_params["nonlin_var"] = nonlin_var.get()
    current_params["optimizer_var"] = optimizer_var.get()

    # Collect regimes to ignore
    current_params["regimes_to_ignore"] = regimes_to_ignore_entry.get()

    # Prompt user for filename
    save_filename = filedialog.asksaveasfilename(
        defaultextension=".json",
        filetypes=[("JSON files", "*.json")],
        initialdir="params",
        title="Save Parameters As"
    )

    if not save_filename:
        # User cancelled the save dialog
        return

    # Ensure the file is saved inside the 'params' folder
    params_folder = "params"
    if not os.path.exists(params_folder):
        os.makedirs(params_folder)

    # If user didn't specify the 'params' directory, prepend it
    if not save_filename.startswith(os.path.abspath(params_folder)):
        save_filename = os.path.join(params_folder, os.path.basename(save_filename))

    try:
        with open(save_filename, "w") as f:
            json.dump(current_params, f, indent=4)
        messagebox.showinfo("Success", f"Parameters saved successfully to {save_filename}!")

        # Refresh the dropdown menu
        refresh_param_dropdown()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save parameters:\n{e}")

def refresh_param_dropdown():
    """
    Refresh the param_file_dropdown to include any new JSON files in the 'params' folder.
    """
    param_file_dropdown['menu'].delete(0, 'end')

    # Collect .json files in params folder
    available_param_files = ["default"]
    params_folder = "params"
    if os.path.exists(params_folder):
        for f in os.listdir(params_folder):
            if f.endswith(".json"):
                available_param_files.append(f)

    # Add options to the dropdown menu
    for file in available_param_files:
        param_file_dropdown['menu'].add_command(label=file, command=tk._setit(param_file_var, file))

    # If the current selection is not in the new list, reset to default
    if param_file_var.get() not in available_param_files:
        param_file_var.set("default")

# --------------------------------------
# Main training function
# --------------------------------------
def run_training():
    # Validate train_samples
    train_samples_input = train_samples_entry.get()
    try:
        train_samples_value = float(train_samples_input)
        if not (0.0 < train_samples_value < 1.0):
            raise ValueError
    except ValueError:
        messagebox.showerror("Input Error", "Train Samples must be a float between 0 and 1 (e.g., 0.8 for 80%).")
        return

    # Collect all input values
    data_path = os.path.normpath(f"data/{intervention_type_var.get()}/" + data_path_entry.get())
    num_vars = num_vars_entry.get()
    model = model_entry.get()
    exp_path = os.path.normpath("experiments/" + exp_path_entry.get())
    i_dataset = i_dataset_entry.get()
    num_train_iter = num_train_iter_entry.get()
    train_batch_size = train_batch_size_entry.get()

    # Additional parameters
    retrain = retrain_var.get()
    dag_for_retrain = dag_for_retrain_entry.get()
    random_seed = random_seed_entry.get()

    train_samples = train_samples_entry.get()
    test_samples = test_samples_entry.get()
    num_folds = num_folds_entry.get()
    fold = fold_entry.get()
    normalize_data = normalize_data_var.get()
    regimes_to_ignore = regimes_to_ignore_entry.get()
    test_on_new_regimes = test_on_new_regimes_var.get()

    num_layers = num_layers_entry.get()
    hid_dim = hid_dim_entry.get()
    nonlin = nonlin_var.get()
    flow_num_layers = flow_num_layers_entry.get()
    flow_hid_dim = flow_hid_dim_entry.get()

    dcd = dcd_var.get()
    coeff_interv_sparsity = coeff_interv_sparsity_entry.get()

    optimizer = optimizer_var.get()
    lr = lr_entry.get()
    lr_reinit = lr_reinit_entry.get()
    lr_schedule = lr_schedule_entry.get()
    stop_crit_win = stop_crit_win_entry.get()
    reg_coeff = reg_coeff_entry.get()

    omega_gamma = omega_gamma_entry.get()
    omega_mu = omega_mu_entry.get()
    mu_init = mu_init_entry.get()
    mu_mult_factor = mu_mult_factor_entry.get()
    gamma_init = gamma_init_entry.get()
    h_threshold = h_threshold_entry.get()

    patience = patience_entry.get()
    train_patience = train_patience_entry.get()
    train_patience_post = train_patience_post_entry.get()

    plot_freq = plot_freq_entry.get()
    no_w_adjs_log = no_w_adjs_log_var.get()
    plot_density = plot_density_var.get()

    use_float = float_var.get()
    intervention = intervention_var.get()  # Whether to use intervention or not

    # Construct the command
    command = [
        python_executable, "main.py",
        "--train",
        "--data-path", data_path,
        "--num-vars", num_vars,
        "--exp-path", exp_path,
        "--model", model,
        "--i-dataset", i_dataset,
        "--num-train-iter", num_train_iter,
        "--train-batch-size", train_batch_size,
        "--random-seed", random_seed,
        "--train-samples", train_samples,
        "--num-folds", num_folds,
        "--fold", fold,
        "--num-layers", num_layers,
        "--hid-dim", hid_dim,
        "--flow-num-layers", flow_num_layers,
        "--flow-hid-dim", flow_hid_dim,
        "--optimizer", optimizer,
        "--lr", lr,
        "--stop-crit-win", stop_crit_win,
        "--reg-coeff", reg_coeff,
        "--omega-gamma", omega_gamma,
        "--omega-mu", omega_mu,
        "--mu-init", mu_init,
        "--mu-mult-factor", mu_mult_factor,
        "--gamma-init", gamma_init,
        "--h-threshold", h_threshold,
        "--patience", patience,
        "--train-patience", train_patience,
        "--train-patience-post", train_patience_post,
        "--plot-freq", plot_freq,
    ]

    # Handle boolean flags
    if retrain:
        command.append("--retrain")
    if dag_for_retrain:
        command.extend(["--dag-for-retrain", dag_for_retrain])
    if normalize_data:
        command.append("--normalize-data")
    if regimes_to_ignore:
        regimes = regimes_to_ignore.split()
        command.extend(["--regimes-to-ignore"] + regimes)
    if test_on_new_regimes:
        command.append("--test-on-new-regimes")
    if dcd:
        command.append("--dcd")
    if no_w_adjs_log:
        command.append("--no-w-adjs-log")
    if plot_density:
        command.append("--plot-density")
    if use_float:
        command.append("--float")
    if intervention:
        command.append("--intervention")
        command.extend(["--intervention-type", intervention_type_var.get()])
        command.extend(["--intervention-knowledge", intervention_knowledge_var.get()])

    # Handle optional values
    if lr_reinit:
        command.extend(["--lr-reinit", lr_reinit])
    if lr_schedule:
        command.extend(["--lr-schedule", lr_schedule])

    # GPU check
    if is_gpu_available():
        command.append("--gpu")
        print("GPU is available. Using GPU for training.")
    else:
        print("GPU is not available. Training will use CPU.")

    # Function to run the training process in a separate thread
    def run_training_process():
        try:
            print(f"Running command: {' '.join(command)}")
            subprocess.run(command, check=True)
            messagebox.showinfo("Success", "Training completed successfully!")
            enable_predicted_button()
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Training failed: {e}")

    # Start training in a separate thread
    threading.Thread(target=run_training_process).start()

def enable_predicted_button():
    show_predicted_button.config(state=tk.NORMAL)
    shd_button.config(state=tk.NORMAL)

def show_interventions():
    initial_file_path = os.path.normpath(
        os.path.join(f"data/{intervention_type_var.get()}/" + data_path_entry.get(),
                     f"data{i_dataset_entry.get()}.npy")
    )
    intervention_file_path = os.path.normpath(
        os.path.join(f"data/{intervention_type_var.get()}/" + data_path_entry.get(),
                     f"data_interv{i_dataset_entry.get()}.npy")
    )

    if os.path.exists(initial_file_path) and os.path.exists(intervention_file_path):
        # Load data
        initial_data = np.load(initial_file_path)
        intervention_data = np.load(intervention_file_path)

        # Correlation
        initial_corr = np.corrcoef(initial_data, rowvar=False)
        intervention_corr = np.corrcoef(intervention_data, rowvar=False)

        # MI
        def calculate_mi(data):
            num_vars = data.shape[1]
            mi_matrix = np.zeros((num_vars, num_vars))
            for i in range(num_vars):
                for j in range(num_vars):
                    if i != j:
                        mi_matrix[i, j] = mutual_info_regression(data[:, i].reshape(-1, 1), data[:, j])[0]
            return mi_matrix

        initial_mi = calculate_mi(initial_data)
        intervention_mi = calculate_mi(intervention_data)

        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        fig.suptitle("Intervention Analysis", fontsize=20)

        # Plot correlation matrix
        sns.heatmap(np.round(initial_corr, 2), annot=True, fmt=".2f", cmap='coolwarm', ax=axes[0, 0])
        axes[0, 0].set_title("Initial Correlation Matrix")

        sns.heatmap(np.round(intervention_corr, 2), annot=True, fmt=".2f", cmap='coolwarm', ax=axes[0, 1])
        axes[0, 1].set_title("Intervention Correlation Matrix")

        # MI heatmap
        sns.heatmap(np.round(initial_mi, 2), annot=True, fmt=".2f", cmap='viridis', ax=axes[1, 0])
        axes[1, 0].set_title("Initial Mutual Information")

        sns.heatmap(np.round(intervention_mi, 2), annot=True, fmt=".2f", cmap='viridis', ax=axes[1, 1])
        axes[1, 1].set_title("Intervention Mutual Information")

        # Graphs
        def plot_graph(matrix, title, ax):
            G = nx.Graph()
            num_vars = matrix.shape[0]
            G.add_nodes_from(range(num_vars))

            threshold = 0.1
            for i in range(num_vars):
                for j in range(i + 1, num_vars):
                    weight = abs(matrix[i, j])
                    if weight > threshold:
                        G.add_edge(i, j, weight=weight)

            pos = nx.circular_layout(G)
            edges = G.edges(data=True)
            weights = [d['weight'] * 5 for _, _, d in edges]

            nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=700,
                    font_size=12, font_weight='bold', ax=ax)
            nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, ax=ax)

            edge_labels = {(i, j): f"{d['weight']:.2f}" for i, j, d in edges}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, ax=ax)

            ax.set_title(title, fontsize=14)

        plot_graph(initial_corr, "Initial Correlation Graph", axes[0, 2])
        plot_graph(intervention_corr, "Intervention Correlation Graph", axes[0, 3])
        plot_graph(initial_mi, "Initial Mutual Info Graph", axes[1, 2])
        plot_graph(intervention_mi, "Intervention Mutual Info Graph", axes[1, 3])

        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    else:
        messagebox.showerror("Error", f"File not found: {initial_file_path} or {intervention_file_path}")

def show_initial_graph():
    file_path = os.path.normpath(
        os.path.join(f"data/{intervention_type_var.get()}/" + data_path_entry.get(),
                     f"DAG{i_dataset_entry.get()}.npy")
    )
    if os.path.exists(file_path):
        adjacency_matrix = np.load(file_path)
        plot_adjacency_matrix(adjacency_matrix, title="Initial Graph")
    else:
        messagebox.showerror("Error", f"File not found: {file_path}")

def show_predicted_graph():
    file_path = os.path.normpath(os.path.join("experiments/" + exp_path_entry.get() + "/train/DAG.npy"))
    if os.path.exists(file_path):
        adjacency_matrix = np.load(file_path)
        plot_adjacency_matrix(adjacency_matrix, title="Predicted Graph")
    else:
        messagebox.showerror("Error", f"File not found: {file_path}")

def plot_adjacency_matrix(adjacency_matrix, title="Graph"):
    G = nx.DiGraph()
    num_nodes = len(adjacency_matrix)
    G.add_nodes_from(range(num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if adjacency_matrix[i][j] != 0:
                G.add_edge(i, j)

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.circular_layout(G)

    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=700,
            font_size=12, font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20, ax=ax)

    ax.set_title(title, fontsize=16, pad=30)
    plt.subplots_adjust(top=0.9)
    plt.show()

def show_intervention_graphs():
    initial_dag_path = os.path.normpath(
        os.path.join(f"data/{intervention_type_var.get()}/" + data_path_entry.get(),
                     f"DAG{i_dataset_entry.get()}.npy")
    )
    intervention_data_path = os.path.normpath(
        os.path.join(f"data/{intervention_type_var.get()}/" + data_path_entry.get(),
                     f"data_interv{i_dataset_entry.get()}.npy")
    )
    intervention_info_path = os.path.normpath(
        os.path.join(f"data/{intervention_type_var.get()}/" + data_path_entry.get(),
                     f"intervention{i_dataset_entry.get()}.csv")
    )
    regime_info_path = os.path.normpath(
        os.path.join(f"data/{intervention_type_var.get()}/" + data_path_entry.get(),
                     f"regime{i_dataset_entry.get()}.csv")
    )

    try:
        adjacency_matrix = np.load(initial_dag_path)
        intervention_data = np.load(intervention_data_path)  # Not used directly in the plot, but loaded for reference
        intervention_info = pd.read_csv(intervention_info_path, names=["Node", "Value"])
        regime_info = pd.read_csv(regime_info_path, names=["Regime"])

        G = nx.DiGraph()
        num_nodes = adjacency_matrix.shape[0]
        G.add_nodes_from(range(num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adjacency_matrix[i, j] != 0:
                    G.add_edge(i, j)

        regime_info["Node"] = intervention_info["Node"]
        regimes = regime_info["Regime"].unique()

        for regime in regimes:
            intervened_nodes = regime_info[regime_info["Regime"] == regime]["Node"].dropna().astype(int).tolist()

            fig, ax = plt.subplots(figsize=(8, 6))
            pos = nx.circular_layout(G)

            nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=700,
                    font_size=12, font_weight='bold', ax=ax)
            nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20, ax=ax)

            if intervened_nodes:
                nx.draw_networkx_nodes(G, pos, nodelist=intervened_nodes, node_color='red', node_size=700, ax=ax)

            ax.set_title(f"Intervention Graph for Regime {regime}", fontsize=16, pad=20)
            plt.show()

    except Exception as e:
        messagebox.showerror("Error", str(e))

def compute_shd(G_true, G_pred):
    """
    Compute the Structural Hamming Distance between two DAGs.
    """
    true_edges = set(G_true.edges())
    pred_edges = set(G_pred.edges())

    missing_edges = true_edges - pred_edges
    extra_edges = pred_edges - true_edges

    reversed_edges = set()
    for (u, v) in missing_edges.copy():
        if (v, u) in extra_edges:
            reversed_edges.add((v, u))
            missing_edges.remove((u, v))
            extra_edges.remove((v, u))

    shd = len(missing_edges) + len(extra_edges) + len(reversed_edges)
    return shd

def calculate_shd():
    initial_path = os.path.normpath(
        os.path.join(f"data/{intervention_type_var.get()}/" + data_path_entry.get(),
                     f"DAG{i_dataset_entry.get()}.npy")
    )
    predicted_path = os.path.normpath(
        os.path.join("experiments/" + exp_path_entry.get() + "/train/DAG.npy")
    )

    if not os.path.exists(initial_path):
        messagebox.showerror("Error", f"Initial graph file not found:\n{initial_path}")
        return
    if not os.path.exists(predicted_path):
        messagebox.showerror("Error", f"Predicted graph file not found:\n{predicted_path}")
        return

    try:
        initial_adj = np.load(initial_path)
        predicted_adj = np.load(predicted_path)

        G_initial = nx.DiGraph(initial_adj)
        G_predicted = nx.DiGraph(predicted_adj)

        if G_initial.number_of_nodes() != G_predicted.number_of_nodes():
            messagebox.showerror("Error", "Graphs have different numbers of nodes.")
            return

        shd_value = compute_shd(G_initial, G_predicted)
        messagebox.showinfo("Structural Hamming Distance",
                            f"The SHD between the initial and predicted graphs is: {shd_value}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to calculate SHD:\n{e}")

# --------------------------------------
# Tkinter UI Setup
# --------------------------------------
root = tk.Tk()
root.title("Causal Discovery Training and Visualization")
root.geometry("1500x800")

main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=1)

canvas = tk.Canvas(main_frame)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

input_frame = tk.Frame(canvas)
canvas.create_window((0, 0), window=input_frame, anchor="nw")

columns = 3

# --------------------------------------
# Parameter Layout Setup
# --------------------------------------
parameters = [
    # Column 1
    [
        ("Data Path:", DEFAULT_PARAMS["Data Path:"]),
        ("Number of Variables:", DEFAULT_PARAMS["Number of Variables:"]),
        ("Model (DCDI-G or DCDI-DSF):", DEFAULT_PARAMS["Model (DCDI-G or DCDI-DSF):"]),
        ("Experiment Path:", DEFAULT_PARAMS["Experiment Path:"]),
        ("Dataset Index:", DEFAULT_PARAMS["Dataset Index:"]),
        ("Number of Training Iterations:", DEFAULT_PARAMS["Number of Training Iterations:"]),
        ("Training Batch Size:", DEFAULT_PARAMS["Training Batch Size:"]),
        ("DAG for Retraining (path):", DEFAULT_PARAMS["DAG for Retraining (path):"]),
        ("Random Seed:", DEFAULT_PARAMS["Random Seed:"]),
        ("Train Samples (e.g., 0.8 for 80%)", DEFAULT_PARAMS["Train Samples (e.g., 0.8 for 80%)"]),
        ("Test Samples (leave blank for default):", DEFAULT_PARAMS["Test Samples (leave blank for default):"]),
        ("Number of Folds:", DEFAULT_PARAMS["Number of Folds:"]),
        ("Fold for Testing:", DEFAULT_PARAMS["Fold for Testing:"]),
        ("Number of Layers:", DEFAULT_PARAMS["Number of Layers:"]),
        ("Hidden Dimension:", DEFAULT_PARAMS["Hidden Dimension:"]),
        ("Flow Number of Layers:", DEFAULT_PARAMS["Flow Number of Layers:"]),
        ("Flow Hidden Dimension:", DEFAULT_PARAMS["Flow Hidden Dimension:"]),
    ],
    # Column 2
    [
        ("Coefficient Intervention Sparsity:", DEFAULT_PARAMS["Coefficient Intervention Sparsity:"]),
        ("Learning Rate:", DEFAULT_PARAMS["Learning Rate:"]),
        ("Learning Rate Reinit (optional):", DEFAULT_PARAMS["Learning Rate Reinit (optional):"]),
        ("Learning Rate Schedule (optional):", DEFAULT_PARAMS["Learning Rate Schedule (optional):"]),
        ("Stop Criterion Window:", DEFAULT_PARAMS["Stop Criterion Window:"]),
        ("Regularization Coefficient:", DEFAULT_PARAMS["Regularization Coefficient:"]),
        ("Omega Gamma:", DEFAULT_PARAMS["Omega Gamma:"]),
        ("Omega Mu:", DEFAULT_PARAMS["Omega Mu:"]),
        ("Mu Init:", DEFAULT_PARAMS["Mu Init:"]),
        ("Mu Mult Factor:", DEFAULT_PARAMS["Mu Mult Factor:"]),
        ("Gamma Init:", DEFAULT_PARAMS["Gamma Init:"]),
        ("H Threshold:", DEFAULT_PARAMS["H Threshold:"]),
        ("Patience:", DEFAULT_PARAMS["Patience:"]),
        ("Train Patience:", DEFAULT_PARAMS["Train Patience:"]),
        ("Train Patience Post:", DEFAULT_PARAMS["Train Patience Post:"]),
        ("Plot Frequency:", DEFAULT_PARAMS["Plot Frequency:"]),
    ],
    # Column 3
    [
        # Currently no additional fields here, can be used for extension
    ]
]

entry_widgets = {}
column_frames = []

for col in range(columns):
    frame = tk.Frame(input_frame, padx=10, pady=10)
    frame.grid(row=0, column=col, sticky="nw")
    column_frames.append(frame)

for col in range(columns):
    params = parameters[col]
    for row, (label_text, default_val) in enumerate(params):
        labeled_frame, ent = create_labeled_entry(column_frames[col], label_text, default_val)
        labeled_frame.grid(row=row, column=0, sticky="w", pady=2)
        entry_widgets[label_text] = ent

# Unpack entry fields from the entry_widgets dictionary
data_path_entry = entry_widgets["Data Path:"]
num_vars_entry = entry_widgets["Number of Variables:"]
model_entry = entry_widgets["Model (DCDI-G or DCDI-DSF):"]
exp_path_entry = entry_widgets["Experiment Path:"]
i_dataset_entry = entry_widgets["Dataset Index:"]
num_train_iter_entry = entry_widgets["Number of Training Iterations:"]
train_batch_size_entry = entry_widgets["Training Batch Size:"]
dag_for_retrain_entry = entry_widgets["DAG for Retraining (path):"]
random_seed_entry = entry_widgets["Random Seed:"]
train_samples_entry = entry_widgets["Train Samples (e.g., 0.8 for 80%)"]
test_samples_entry = entry_widgets["Test Samples (leave blank for default):"]
num_folds_entry = entry_widgets["Number of Folds:"]
fold_entry = entry_widgets["Fold for Testing:"]
num_layers_entry = entry_widgets["Number of Layers:"]
hid_dim_entry = entry_widgets["Hidden Dimension:"]
flow_num_layers_entry = entry_widgets["Flow Number of Layers:"]
flow_hid_dim_entry = entry_widgets["Flow Hidden Dimension:"]
coeff_interv_sparsity_entry = entry_widgets["Coefficient Intervention Sparsity:"]
lr_entry = entry_widgets["Learning Rate:"]
lr_reinit_entry = entry_widgets["Learning Rate Reinit (optional):"]
lr_schedule_entry = entry_widgets["Learning Rate Schedule (optional):"]
stop_crit_win_entry = entry_widgets["Stop Criterion Window:"]
reg_coeff_entry = entry_widgets["Regularization Coefficient:"]
omega_gamma_entry = entry_widgets["Omega Gamma:"]
omega_mu_entry = entry_widgets["Omega Mu:"]
mu_init_entry = entry_widgets["Mu Init:"]
mu_mult_factor_entry = entry_widgets["Mu Mult Factor:"]
gamma_init_entry = entry_widgets["Gamma Init:"]
h_threshold_entry = entry_widgets["H Threshold:"]
patience_entry = entry_widgets["Patience:"]
train_patience_entry = entry_widgets["Train Patience:"]
train_patience_post_entry = entry_widgets["Train Patience Post:"]
plot_freq_entry = entry_widgets["Plot Frequency:"]

# --------------------------------------
# Intervention Type Radiobuttons
# --------------------------------------
intervention_type_var = tk.StringVar(value=DEFAULT_PARAMS["intervention_type_var"])
intervention_type_frame = tk.LabelFrame(input_frame, text="Intervention Type", padx=10, pady=10)
intervention_type_frame.grid(row=1, column=0, columnspan=columns, sticky="w", padx=10, pady=5)

tk.Radiobutton(intervention_type_frame, text="Perfect", variable=intervention_type_var, value="perfect").pack(side=tk.LEFT, padx=5)
tk.Radiobutton(intervention_type_frame, text="Imperfect", variable=intervention_type_var, value="imperfect").pack(side=tk.LEFT, padx=5)

# --------------------------------------
# Intervention Knowledge Radiobuttons
# --------------------------------------
intervention_knowledge_var = tk.StringVar(value=DEFAULT_PARAMS["intervention_knowledge_var"])
intervention_knowledge_frame = tk.LabelFrame(input_frame, text="Intervention Knowledge", padx=10, pady=10)
intervention_knowledge_frame.grid(row=2, column=0, columnspan=columns, sticky="w", padx=10, pady=5)

tk.Radiobutton(intervention_knowledge_frame, text="Known", variable=intervention_knowledge_var, value="known").pack(side=tk.LEFT, padx=5)
tk.Radiobutton(intervention_knowledge_frame, text="Unknown", variable=intervention_knowledge_var, value="unknown").pack(side=tk.LEFT, padx=5)

# --------------------------------------
# Checkboxes
# --------------------------------------
checkbox_frame = tk.Frame(input_frame, padx=10, pady=10)
checkbox_frame.grid(row=3, column=0, columnspan=columns, sticky="w", padx=10, pady=5)

intervention_var = tk.BooleanVar(value=DEFAULT_PARAMS["intervention_var"])
tk.Checkbutton(checkbox_frame, text="Use Intervention", variable=intervention_var).pack(side=tk.LEFT, padx=5)

dcd_var = tk.BooleanVar(value=DEFAULT_PARAMS["dcd_var"])
tk.Checkbutton(checkbox_frame, text="Use DCD", variable=dcd_var).pack(side=tk.LEFT, padx=5)

normalize_data_var = tk.BooleanVar(value=DEFAULT_PARAMS["normalize_data_var"])
tk.Checkbutton(checkbox_frame, text="Normalize Data", variable=normalize_data_var).pack(side=tk.LEFT, padx=5)

retrain_var = tk.BooleanVar(value=DEFAULT_PARAMS["retrain_var"])
tk.Checkbutton(checkbox_frame, text="Use Retrain", variable=retrain_var).pack(side=tk.LEFT, padx=5)

no_w_adjs_log_var = tk.BooleanVar(value=DEFAULT_PARAMS["no_w_adjs_log_var"])
tk.Checkbutton(checkbox_frame, text="No W Adjs Log", variable=no_w_adjs_log_var).pack(side=tk.LEFT, padx=5)

plot_density_var = tk.BooleanVar(value=DEFAULT_PARAMS["plot_density_var"])
tk.Checkbutton(checkbox_frame, text="Plot Density", variable=plot_density_var).pack(side=tk.LEFT, padx=5)

float_var = tk.BooleanVar(value=DEFAULT_PARAMS["float_var"])
tk.Checkbutton(checkbox_frame, text="Use Float Precision", variable=float_var).pack(side=tk.LEFT, padx=5)

# --------------------------------------
# Regimes to Ignore
# --------------------------------------
regimes_frame = tk.Frame(input_frame, padx=10, pady=10)
regimes_frame.grid(row=4, column=0, columnspan=columns, sticky="w", padx=10, pady=5)

tk.Label(regimes_frame, text="Regimes to Ignore (space-separated ints):").pack(side=tk.LEFT)
regimes_to_ignore_entry = tk.Entry(regimes_frame, width=30)
regimes_to_ignore_entry.pack(side=tk.LEFT, padx=5)
regimes_to_ignore_entry.insert(0, DEFAULT_PARAMS["regimes_to_ignore"])

# --------------------------------------
# Test on New Regimes
# --------------------------------------
test_on_new_regimes_var = tk.BooleanVar(value=DEFAULT_PARAMS["test_on_new_regimes_var"])
tk.Checkbutton(input_frame, text="Test on New Regimes", variable=test_on_new_regimes_var).grid(
    row=5, column=0, columnspan=columns, sticky="w", padx=10, pady=5
)

# --------------------------------------
# Nonlinearity OptionMenu
# --------------------------------------
nonlin_var = tk.StringVar(value=DEFAULT_PARAMS["nonlin_var"])
nonlin_frame = tk.Frame(input_frame, padx=10, pady=10)
nonlin_frame.grid(row=6, column=0, columnspan=columns, sticky="w", padx=10, pady=5)

tk.Label(nonlin_frame, text="Nonlinearity:").pack(side=tk.LEFT)
nonlin_menu = tk.OptionMenu(nonlin_frame, nonlin_var, "leaky-relu", "sigmoid")
nonlin_menu.config(width=20)
nonlin_menu.pack(side=tk.LEFT, padx=5)

# --------------------------------------
# Optimizer OptionMenu
# --------------------------------------
optimizer_var = tk.StringVar(value=DEFAULT_PARAMS["optimizer_var"])
optimizer_frame = tk.Frame(input_frame, padx=10, pady=10)
optimizer_frame.grid(row=7, column=0, columnspan=columns, sticky="w", padx=10, pady=5)

tk.Label(optimizer_frame, text="Optimizer:").pack(side=tk.LEFT)
optimizer_menu = tk.OptionMenu(optimizer_frame, optimizer_var, "sgd", "rmsprop")
optimizer_menu.config(width=20)
optimizer_menu.pack(side=tk.LEFT, padx=5)

# --------------------------------------
# Dropdown to select param file from "params" folder
# --------------------------------------
param_files_frame = tk.Frame(input_frame, padx=10, pady=10)
param_files_frame.grid(row=8, column=0, columnspan=columns, sticky="w", padx=10, pady=5)

tk.Label(param_files_frame, text="Load Params from File:").pack(side=tk.LEFT)

# Collect .json files in params folder
params_folder = "params"
if not os.path.exists(params_folder):
    os.makedirs(params_folder)  # Create if doesn't exist

available_param_files = ["default"]
for f in os.listdir(params_folder):
    if f.endswith(".json"):
        available_param_files.append(f)

param_file_var = tk.StringVar(value="default")
param_file_dropdown = tk.OptionMenu(param_files_frame, param_file_var, *available_param_files)
param_file_dropdown.config(width=30)
param_file_dropdown.pack(side=tk.LEFT, padx=5)

# When dropdown changes, load the selected params
param_file_var.trace("w", load_param_file)

# --------------------------------------
# Buttons to run training and visualize
# --------------------------------------
button_frame = tk.Frame(input_frame, padx=10, pady=10)
button_frame.grid(row=9, column=0, columnspan=columns, pady=20)

tk.Button(button_frame, text="Run Training", command=run_training, width=20, bg="green", fg="white").pack(side=tk.LEFT, padx=10)

show_initial_button = tk.Button(button_frame, text="Show Initial Graph", command=show_initial_graph,
                                state=tk.NORMAL, width=20)
show_initial_button.pack(side=tk.LEFT, padx=10)

show_intervention_button = tk.Button(button_frame, text="Intervention Analysis", command=show_interventions,
                                     state=tk.NORMAL, width=20)
show_intervention_button.pack(side=tk.LEFT, padx=10)

show_intervention_graph_button = tk.Button(button_frame, text="Show Intervention Graphs",
                                           command=show_intervention_graphs, state=tk.NORMAL, width=25)
show_intervention_graph_button.pack(side=tk.LEFT, padx=10)

show_predicted_button = tk.Button(button_frame, text="Show Predicted Graph", command=show_predicted_graph,
                                  state=tk.DISABLED, width=20)
show_predicted_button.pack(side=tk.LEFT, padx=10)

shd_button = tk.Button(button_frame, text="SHD", command=calculate_shd,
                       state=tk.DISABLED, width=10, bg="blue", fg="white")
shd_button.pack(side=tk.LEFT, padx=10)

# --------------------------------------
# Save Parameters Button
# --------------------------------------
save_button = tk.Button(button_frame, text="Save Parameters", command=save_params, width=20, bg="orange", fg="white")
save_button.pack(side=tk.LEFT, padx=10)

# --------------------------------------
# Initially set to default parameters
# --------------------------------------
set_params(DEFAULT_PARAMS)

# --------------------------------------
# Start the Tkinter main loop
# --------------------------------------
root.mainloop()
