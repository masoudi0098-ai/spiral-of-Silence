# spiral-of-Silence
# Network Modeling and Opinion Dynamics: Implementations and Reproducibility

This repository contains the source code and implementation frameworks for the experiments presented in our research paper[cite: 9]. These implementations are provided to ensure the transparency and reproducibility of our results[cite: 9].

## Project Structure

The repository is organized into distinct modules, each corresponding to different stages of our network analysis and opinion dynamics modeling[cite: 9]:

*   `Empirical Twitter Network.py`: Implementation for processing and analyzing the empirical Twitter dataset[cite: 9].
*   `Simulation_Models/`: Contains scripts for the agent-based Monte Carlo simulations[cite: 9].
*   `Requirements.txt`: List of necessary Python libraries (e.g., PyTorch, NetworkX, Matplotlib)[cite: 9].

## Dataset Information

For the empirical analysis presented in our manuscript, we utilize the real-world political retweet network dataset[cite: 9]. 

*   **Dataset Source**: [soc-political-retweet](https://networkrepository.com/soc-political-retweet.php)[cite: 9]
*   **Implementation File**: `Empirical Twitter Network.py`[cite: 9]
*   **Application**: This dataset is specifically employed to generate the results presented in **Figure 3** of our study[cite: 9].

To reproduce the findings, please download the raw data from the provided link and place it in the `/data` directory before executing the script[cite: 9].

## Data Preprocessing Pipeline

To ensure consistency and comparability across all simulations and empirical analyses, the raw network undergoes a unified preprocessing pipeline:

1. **Edge List Ingestion**: Load the raw edge list from the `soc-political-retweet` dataset.
2. **Graph Conversion**: Convert the directed network into an undirected graph representation.
3. **Largest Connected Component (LCC)**: Isolate and extract the LCC to remove disconnected components.
4. **Snowball / BFS Subgraph Sampling**: Extract a dense 6,000-node subgraph using Snowball (Breadth-First Search) sampling starting from the highest-degree node.
5. **Structural Metrics Logging**: Log and verify the structural attributes of the sampled subgraph:
   * **Node Count ($N$)**
   * **Edge Count ($E$)**
   * **Mean Degree ($\langle k \rangle$)**
   * **Maximum Degree ($k_{\max}$)**

## Requirements

The code is implemented in Python[cite: 9]. We recommend using a virtual environment and installing the dependencies[cite: 9]:

```bash
pip install -r requirements.txt
