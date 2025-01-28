import numpy as np
import pandas as pd
import networkx as nx
from cdt.metrics import get_CPDAG

from data.generation.dag_generator import DagGenerator
from data.generation.generate_data import dataset_generator

##########################################
# The code you provided (classes/functions) should be placed here.
# For clarity, we assume you've placed all the code you provided in the same file.
# ------------------ BEGIN OF PROVIDED CODE ------------------

# Please copy the entire code snippet you provided (with classes and functions:
# LinearMechanism, SigmoidMix_Mechanism, etc., DagGenerator, dataset_generator, ...)
# here. For brevity, it's not repeated in this answer, but you must include it fully
# in your final script before running.
#
# Make sure to include:
# - All imports done in your code snippet
# - All classes: LinearMechanism, SigmoidMix_Mechanism, ... , DagGenerator, dataset_generator, etc.
# - The causal_mechanisms.py content you provided
#
# ------------------- END OF PROVIDED CODE -------------------

##########################################

# Based on macroeconomic reasoning, we define a DAG among the 10 columns:
# Columns:
# 0: 'Country Code'
# 1: 'Year'
# 2: 'Adjusted savings: education expenditure (% of GNI)'
# 3: 'Adjusted savings: education expenditure (current US$)'
# 4: 'Foreign direct investment, net inflows (% of GDP)'
# 5: 'GDP growth (annual %)'
# 6: 'Life expectancy at birth, total (years)'
# 7: 'Population, total'
# 8: 'Rural population (% of total population)'
# 9: 'Urban population (% of total population)'

# Proposed causal structure (1=Country Code, 2=Year in indexing start from 1 for clarity):
# Country Code(0) -> Edu%GNI(2), FDI(4), Population(7)
# Year(1) -> Edu%GNI(2), FDI(4), GDP growth(5), Life Exp(6), Population(7), Rural%(8), Urban%(9)
# Edu%GNI(2) -> Edu$US(3), GDP growth(5), Life Exp(6)
# FDI(4) -> GDP growth(5)
# GDP growth(5) -> Life Exp(6)
# Population(7) -> Rural%(8), Urban%(9)

# Adjacency matrix (10x10) according to the above structure:
adj_matrix = np.array([
    [0,0,1,0,1,0,0,1,0,0],  # Country Code -> Edu%GNI(2), FDI(4), Pop(7)
    [0,0,1,0,1,1,1,1,1,1],  # Year -> Edu%GNI(2), FDI(4), GDP(5), Life(6), Pop(7), Rural(8), Urban(9)
    [0,0,0,1,0,1,1,0,0,0],  # Edu%GNI(2) -> Edu$US(3), GDP(5), Life(6)
    [0,0,0,0,0,0,0,0,0,0],  # Edu$US(3) no children
    [0,0,0,0,0,1,0,0,0,0],  # FDI(4) -> GDP(5)
    [0,0,0,0,0,0,1,0,0,0],  # GDP(5) -> Life(6)
    [0,0,0,0,0,0,0,0,0,0],  # Life(6) no children
    [0,0,0,0,0,0,0,0,1,1],  # Pop(7) -> Rural(8), Urban(9)
    [0,0,0,0,0,0,0,0,0,0],  # Rural(8) no children
    [0,0,0,0,0,0,0,0,0,0]   # Urban(9) no children
])

# Save DAG
np.save("DAG1.npy", adj_matrix)

# Compute CPDAG
cpdag = get_CPDAG(adj_matrix)
np.save("CPDAG1.npy", cpdag)

# Load main_data.csv and save as data1.npy
data = pd.read_csv("main_data.csv")
np.save("data1.npy", data.values)

#############################################################
# Generate interventional data using the provided code.
# We will create a single structural intervention setting on one node.
# Let's intervene on the node corresponding to 'Adjusted savings: education expenditure (current US$)' which is index 3.

# Import classes now that we have them (assuming they are in the same file or scope)
# If they are defined above, this step is not needed again.
# from dataset_generator import dataset_generator
# from causal_mechanisms import UniformCause

mechanism = "linear"          # arbitrary choice, we just need a mechanism for the generator
initial_cause = "gaussian"    # arbitrary
noise = "gaussian"            # arbitrary
noise_coeff = 0.4
nb_nodes = 10
expected_degree = 1
nb_points = data.shape[0]     # 9600 to match our real data size
suffix = "example"
rescale = False
obs_data = True
nb_interventions = 1
min_nb_target = 1
max_nb_target = 1
conservative = False
cover = False

# Create the dataset_generator object
generator = dataset_generator(mechanism, initial_cause, "structural", "uniform", noise, noise_coeff,
                              nb_nodes, expected_degree, nb_points, suffix, rescale,
                              obs_data, nb_interventions, min_nb_target, max_nb_target,
                              conservative, cover, verbose=False)

# Initialize the internal DagGenerator
generator.generator = DagGenerator(mechanism, noise=noise, noise_coeff=noise_coeff, cause=initial_cause,
                                       npoints=nb_points, nodes=nb_nodes, expected_density=expected_degree,
                                       rescale=rescale)

# Override with our chosen DAG and data
generator.generator.original_adjacency_matrix = adj_matrix
generator.generator.adjacency_matrix = adj_matrix.copy()
generator.generator.g = nx.DiGraph(adj_matrix)
generator.generator.cfunctions = None
generator.generator.data = pd.DataFrame(data.values, columns=[
    "Country Code", "Year",
    "Adjusted savings: education expenditure (% of GNI)",
    "Adjusted savings: education expenditure (current US$)",
    "Foreign direct investment, net inflows (% of GDP)",
    "GDP growth (annual %)", "Life expectancy at birth, total (years)",
    "Population, total", "Rural population (% of total population)",
    "Urban population (% of total population)"
])

# Perform a structural intervention on node index 3 (Edu$US)
from causal_mechanisms import UniformCause
intervention_nodes = [3]
interv_data, g = generator.generator.intervene("structural", intervention_nodes, nb_points, UniformCause())

# Save data_interv1.npy
np.save("data_interv1.npy", interv_data)

# Save intervention1.csv: the same intervention (node 3) applied to all rows
with open("intervention1.csv", "w") as f:
    for _ in range(nb_points):
        f.write("3\n")

# Save regime1.csv: we have a single interventional regime labeled as '1'
with open("regime1.csv", "w") as f:
    for _ in range(nb_points):
        f.write("1\n")

print("CPDAG1.npy, DAG1.npy, data_interv1.npy, data1.npy, intervention1.csv, and regime1.csv have been generated.")
