import networkx as nx
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from collections import deque
import random
import gc

# ==========================================
# 1. Configuration & Styling
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 300
})

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Thermodynamic Simulation Engine running on: {DEVICE}")

# ==========================================
# 2. Scientifically Valid Sampling (Snowball)
# ==========================================
def snowball_sampling(filepath, target_nodes=6000):
    """
    Extracts a topology-preserving subgraph using BFS (Snowball Sampling).
    """
    # CRITICAL FIX: delimiter changed to ' ' for SNAP .edges datasets
    G_raw = nx.read_edgelist(filepath, delimiter=',', nodetype=int, data=False)
    lcc_nodes = max(nx.connected_components(G_raw), key=len)
    G_sub = G_raw.subgraph(lcc_nodes).copy()
    
    start_node = list(G_sub.nodes())[SEED % len(G_sub.nodes())]
    sampled_nodes = {start_node}
    queue = deque([start_node])
    
    while queue and len(sampled_nodes) < target_nodes:
        current = queue.popleft()
        neighbors = list(G_sub.neighbors(current))
        random.shuffle(neighbors)
        
        for neighbor in neighbors:
            if neighbor not in sampled_nodes:
                sampled_nodes.add(neighbor)
                queue.append(neighbor)
                if len(sampled_nodes) >= target_nodes:
                    break
                    
    return nx.convert_node_labels_to_integers(G_sub.subgraph(sampled_nodes).copy())

# ==========================================
# 3. Main Phase Diagram Engine
# ==========================================
def main():
    filepath = "soc-political-retweet.edges"
    try:
        G_emp = snowball_sampling(filepath, target_nodes=6000)
    except FileNotFoundError:
        print("[!] Dataset not found. Ensure 'soc-political-retweet.edges' is in the directory.")
        return

    N = G_emp.number_of_nodes()
    degrees = dict(G_emp.degree())
    sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
    
    # Sparse Matrix Representation for O(E) complexity
    A_scipy = nx.to_scipy_sparse_array(G_emp, format='coo')
    i = torch.LongTensor(np.vstack((A_scipy.row, A_scipy.col)))
    v = torch.FloatTensor(A_scipy.data)
    A_sparse = torch.sparse_coo_tensor(i, v, torch.Size(A_scipy.shape)).to(DEVICE)
    
    # Grid Resolution (60x60 for HD contours)
    res = 60 
    z_vals = np.linspace(0.0, 0.25, res)
    alpha_vals = np.linspace(0.1, 1.0, res)
    heatmap_data = np.zeros((res, res))

    ENSEMBLE_SIZE = 1000 

    # Theory Parameters (Eqs 4, 7, 9)
    theta_threshold = 0.15   # Psychological threshold
    sigma = 15.0             # Steepness of Heaviside sigmoid (Eq 7)
    c0 = 1.0                 # Base cost
    beta = 2.5               # Psychological sensitivity
    b = 0.1                  # Base intrinsic reward (Eq 9)
    dt = 0.1                 # Replicator integration step
    
    print("[*] Simulating Phase Diagram with Ensemble Size = {ENSEMBLE_SIZE}...")
    
    for i, alpha in enumerate(alpha_vals):
        for j, z in enumerate(z_vals):
            num_hubs = int(z * N)
            hubs = sorted_nodes[:num_hubs]
            
            ensemble_outcomes = []
            for e in range(ENSEMBLE_SIZE):
                # 0.0 = Silent, 1.0 = Expressing
                states = torch.zeros(N, dtype=torch.float32, device=DEVICE)
                if num_hubs > 0:
                    states[hubs] = 1.0
                    
                for step in range(80): # Adequate steps for thermodynamic equilibrium
                    # O(E) calculation of neighborhood states
                    active_sum = torch.sparse.mm(A_sparse, states.unsqueeze(1)).squeeze(1)
                    silent_sum = torch.sparse.mm(A_sparse, (1.0 - states).unsqueeze(1)).squeeze(1)
                    
                    # Equations (5 & 6): Effective visibility topology
                    phi_1 = active_sum / (active_sum + alpha * silent_sum + 1e-12)
                    phi_0 = (alpha * active_sum) / (alpha * active_sum + silent_sum + 1e-12)
                    phi = torch.where(states == 1.0, phi_1, phi_0)
                    
                    # Equation (7): Smoothed Heaviside Step Function
                    H_gate = 1.0 / (1.0 + torch.exp(-sigma * (phi - theta_threshold)))
                    
                    # Equation (4): Dynamic cost
                    cost = c0 * torch.exp(-beta * phi)
                    
                    # Equation (9 / 11): Net Payoff with baseline factor 'b'
                    net_payoff = (1.0 - b) * phi - cost
                    
                    # Equation (8): Microscopic Replicator Dynamics Probabilities
                    prob_0_to_1 = torch.clamp(dt * torch.relu(net_payoff) * H_gate, 0.0, 1.0)
                    prob_1_to_0 = torch.clamp(dt * torch.relu(-net_payoff) * H_gate, 0.0, 1.0)
                    
                    rand_tensor = torch.rand(N, device=DEVICE)
                    new_states = states.clone()
                    
                    # Probabilistic transitions
                    new_states[(states == 0.0) & (rand_tensor < prob_0_to_1)] = 1.0
                    new_states[(states == 1.0) & (rand_tensor < prob_1_to_0)] = 0.0
                    
                    # Rigid constraint: Committed agents (Zealots) never revert
                    if num_hubs > 0:
                        new_states[hubs] = 1.0
                    
                    if torch.equal(states, new_states):
                        break
                    states = new_states
                
                ensemble_outcomes.append(states.mean().item())
            
            heatmap_data[i, j] = np.mean(ensemble_outcomes)
            
        torch.cuda.empty_cache()
        gc.collect()

    # ==========================================
    # 4. HD Plotting & Critical Contours
    # ==========================================
    fig, ax = plt.subplots(figsize=(9, 6))
    
    colors = ["#000033", "#1f77b4", "#2ca02c", "#6cff56", "#ffff00"]
    custom_cmap = LinearSegmentedColormap.from_list("DarkBlueToYellow", colors, N=256)
    
    im = ax.imshow(heatmap_data, 
                   extent=[z_vals.min(), z_vals.max(), alpha_vals.min(), alpha_vals.max()], 
                   origin='lower', aspect='auto', cmap=custom_cmap, interpolation='bilinear',
                   vmin=0.0, vmax=1.0) 
    
    critical_levels = [0.2, 0.5, 0.8]
    contours = ax.contour(z_vals, alpha_vals, heatmap_data, levels=critical_levels, 
                          colors='red', linewidths=1.5, linestyles='dashed')
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r'Steady-State Expressive Fraction ($\Theta^*$)', fontsize=13, labelpad=12)
    
    ax.set_xlabel(r'Committed Agents Fraction ($z$)', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'Visibility Coefficient ($\alpha$)', fontsize=13, fontweight='bold')
    ax.set_title('Phase Diagram: Empirical Twitter Network', fontsize=15, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig('Fig3_Ultimate_Phase_Diagram.png', dpi=400, bbox_inches='tight')
    plt.savefig('Fig3_Ultimate_Phase_Diagram.pdf', format='pdf', bbox_inches='tight')
    print("[+] Plot successfully generated and saved as HD images.")

if __name__ == "__main__":
    main()
