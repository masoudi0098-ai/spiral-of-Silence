# spiral-of-Silence
# Network Modeling and Opinion Dynamics: Implementations and Reproducibility

This repository contains the source code, implementation frameworks, and dataset preprocessing pipelines for the experiments presented in our research paper. The codebase provides massively parallel, GPU-accelerated Monte Carlo simulations to ensure absolute computational transparency and methodological reproducibility.

## 1. Dataset & Empirical Preprocessing

The empirical analysis is conducted on a real-world political interaction network.

*   **Dataset Name**: `soc-political-retweet`
*   **Source**: [Network Repository](https://networkrepository.com/soc-political-retweet.php)
*   **Raw Network Specifications**: The initial raw dataset comprises 18,500 nodes and 61,157 edges.

### Preprocessing Pipeline
To ensure structural coherence and computational tractability while preserving underlying topological heterogeneity, the data undergoes a rigorous preprocessing pipeline:
1.  **Instantiation**: The raw edge list is loaded as an unweighted, undirected graph.
2.  **LCC Extraction**: The Largest Connected Component (LCC) is extracted from the global structure.
3.  **Snowball Sampling**: A representative subgraph is extracted via a Breadth-First Search (BFS)-based snowball sampling technique. This extraction is initialized with a fixed master seed (`42`) to guarantee deterministic replicability.

*   **Final Network Specifications**: The resulting induced subgraph contains exactly $N = 6000$ nodes and $E = 26,639$ edges. The network exhibits a mean degree of $\langle k \rangle \approx 8.88$, a maximum degree of $k_{max} = 366$, and a mean target hub degree of $k_{hub} \approx 99.59$.

## 2. Simulation Setup & Model Parameters

All simulation configurations are executed asynchronously and evaluated across independent Monte Carlo realizations. The dynamical evolution reaches a macroscopic steady state when the absolute change in the expressive fraction drops below $10^{-5}$ over 500 consecutive time steps.

*   **Master Seed**: `42` (ensures exact reproducibility across all scripts)
*   **Monte Carlo Runs (Ensemble Size)**: `1000` independent realizations per configuration
*   **Committed-Agent Fraction ($z$)**: Swept over $[0.0, 0.25]$
*   **Visibility Coefficient ($\alpha$)**: Swept over $[0.1, 1.0]$
*   **Base Cost ($c_0$)**: `1.0`
*   **Cost Exponent / Sensitivity ($\beta$)**: `3.0`
*   **Theoretical Threshold ($\theta$)**: `0.3`
*   **Threshold Steepness ($\sigma$)**: `15.0` (provides a numerically smooth Heaviside approximation)
*   **Payoff Coefficient ($b$)**: `0.3`

## 3. System Requirements

The simulation engine is highly optimized, utilizing PyTorch JIT compilation and sparse tensor operations on GPUs. 

*   **Python Version**: 3.8 or higher
*   **Core Libraries**:
    *   `torch` (CUDA-enabled is highly recommended for parallel execution)
    *   `networkx` (v3.0+)
    *   `numpy`
    *   `pandas`
    *   `matplotlib`
    *   `scipy`


## 4. Execution Guide & Figure Generation

Ensure the dataset file `soc-political-retweet.edges` is located in the root directory before running the scripts.

### Reproducing Figure 3 (Empirical Phase Diagram)
EmpiricalTwitterNetwork.py: This script utilizes an ultra-fast JIT-compiled physics kernel to compute the macroscopic steady-state expressive fraction θ across a grid of z and α.
Output Files Generated:Fig-003.png / Fig-003.pdf: High-resolution phase diagram heatmaps featuring 0.2, 0.5, and 0.8 contour markers.
### Reproducing Figure 4 (Placement Strategy Comparison)
This script evaluates the impact of distinct targeting strategies (Degree, Betweenness, PageRank, Random) on achieving network-wide consensus.

Output Files Generated:
raw_results.csv: A comprehensive dataset containing mean expressive fractions, standard deviations, and 95% Confidence Intervals (CI) for all z values and strategies.  
Fig_Placement_with_CI.png / Fig_Placement_with_CI.pdf: A high-quality line plot detailing the behavioral thresholds and theoretical Zc predictions with shaded error margins.   
Console Output: A specialized reviewer analysis report computing the theoretical Zc (Eq. 13) and empirical relative errors directly in the terminal. 
