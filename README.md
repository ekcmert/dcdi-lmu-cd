# Differentiable Causal Discovery from Interventional Data

This project builds upon the original implementation of "Differentiable Causal Discovery from Interventional Data" (DCDI) to explore its theoretical and practical capabilities in learning causal structures from interventional data. Leveraging the original codebase, we aimed to assess and enhance the framework's performance under diverse experimental conditions. Our primary focus was on improving the usability and flexibility of the codebase to facilitate more comprehensive experiments and analyses.

To achieve this, we made several significant modifications and improvements:  
1. **Graphical User Interface (GUI) Development**:  
   A GUI was designed to streamline the experimentation process. This interface allows users to intuitively adjust hyperparameters, execute experiments, and visualize results. Key features include customizable fields for learning rates, regularization coefficients, training iterations, batch sizes, and neural network architecture, along with tooltip-based explanations for parameter functionality.

2. **Synthetic Data Generation Enhancements**:  
   The data generation framework was extended to allow for explicit custom DAG initialization by accepting user-defined adjacency matrices. This enables experiments on specific causal structures rather than relying on randomly generated graphs. Additionally, intervention settings were made more configurable, supporting various intervention types, noise distributions, and functional mechanisms (e.g., linear, polynomial, or neural networks).

3. **Metric Calculation Updates**:  
   The calculation of evaluation metrics, such as structural Hamming distance (SHD) and edge errors, was updated to use adjacency matrix and graph-based libraries (e.g., `networkx`) directly, bypassing the previous reliance on the `cdt` package. This improved efficiency and accuracy in the evaluation pipeline.

4. **Training Routine Modifications**:  
   Computational limitations were addressed by introducing early stopping with a reduced number of iterations for both pre- and post-thresholding steps. Additionally, the checkpoint frequency for updating the best score was reduced from every 1000 iterations to 500 iterations, enabling faster training convergence while maintaining performance.

5. **Result Organization and Visualization Pipeline**:  
   An additional Python notebook was developed to organize and visualize the results obtained from the training pipeline. This notebook enables systematic analysis of experiment setups and outcomes, including visualizing predicted graphs, performance metrics, and intervention results. It simplifies the process of extracting insights from the data, ensuring that results are presented in a clear and interpretable format.

These enhancements enabled us to conduct systematic experiments to evaluate the DCDI framework's robustness and adaptability. Specifically, we analyzed the impact of the number and type of interventions, explored hyperparameter sensitivity, and tested the framework's generalizability to diverse datasets. Our findings demonstrate that increasing the number of interventions significantly improves DAG identifiability while DCDI remains robust across various intervention types. 

By bridging theoretical research with practical implementations, our extended version of DCDI provides valuable insights into causal discovery and its applications in real-world domains such as macroeconomic growth analysis.

---

# Differentiable Causal Discovery from Interventional Data

![DCDI paper graphics](https://github.com/slachapelle/dcdi/raw/master/paper_graphic.png)

Repository for "Differentiable Causal Discovery from Interventional Data".

* The code for DCDI is available [here](./dcdi).

* The datasets used in the paper are available [here](./data)

The code for the following baselines is also provided:
* [CAM](./cam)
* [GIES](./gies)
* [IGSP and UT-IGSP](./igsp)

To run DCDI, you can use a command similar to this one:  
`python ./dcdi/main.py --train --data-path ./data/perfect/data_p10_e10_n10000_linear_struct --num-vars 10 --i-dataset 1 --exp-path exp --model DCDI-DSF --intervention --intervention-type perfect --intervention-knowledge known --reg-coeff 0.5`

Here, we assume the zip files in the directory `data` have been unzipped and that the results will be saved in the directory `exp`. With this command, you will train the model DCDI-DSF in the perfect known setting (with reasonable default hyperparameters) using the first instance (`i_dataset = 1`) of the 10-node graph dataset with linear mechanisms and perfect intervention. For further details on other hyperparameters (e.g., the architecture of the networks), see the `main.py` file where hyperparameters have a description. 