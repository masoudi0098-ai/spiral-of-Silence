# spiral-of-Silence
# Network Modeling and Opinion Dynamics: Implementations and Reproducibility

This repository contains the source code and implementation frameworks for the experiments presented in our research paper. These implementations are provided to ensure the transparency and reproducibility of our results.

## Project Structure

The repository is organized into distinct modules, each corresponding to different stages of our network analysis and opinion dynamics modeling:

*   `Empirical Twitter Network.py`: Implementation for processing and analyzing the empirical Twitter dataset.
*   `Simulation_Models/`: Contains scripts for the agent-based Monte Carlo simulations.
*   `Requirements.txt`: List of necessary Python libraries (e.g., PyTorch, NetworkX, Matplotlib).

## Dataset Information

For the empirical analysis presented in our manuscript, we utilize the real-world political retweet network dataset. 

*   **Dataset Source**: [soc-political-retweet](https://networkrepository.com/soc-political-retweet.php)
*   **Implementation File**: `Empirical Twitter Network.py`
*   **Application**: This dataset is specifically employed to generate the results presented in **Figure 3** of our study.

To reproduce the findings, please download the raw data from the provided link and place it in the `/data` directory before executing the script.

## Requirements

The code is implemented in Python. We recommend using a virtual environment and installing the dependencies:

```bash
pip install -r requirements.txt
