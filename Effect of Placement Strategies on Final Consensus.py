import networkx as nx
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import random
from collections import deque
import warnings
import time

# Suppress runtime warnings for safe division
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================
# 1. Configuration & Styling
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 400,
})

MASTER_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Simulation Engine running on: {DEVICE}")

# Set deterministic behavior
random.seed(MASTER_SEED)
np.random.seed(MASTER_SEED)
torch.manual_seed(MASTER_SEED)

# ==========================================
# 2. Scientifically Valid Sampling (Snowball)
# ==========================================
def snowball_sampling(filepath, target_nodes=6000, seed=42):
    """
    Unified Graph Extraction Pipeline:
    1. Loads dataset
    2. Enforces Undirected Graph topology
    3. Extracts Largest Connected Component (LCC)
    4. Applies Snowball (BFS) sampling to reach target_nodes
    """
    print(f"[*] Loading and extracting true topology from {filepath}...")
    
    G_raw = nx.read_edgelist(filepath, delimiter=',', nodetype=int, data=False, create_using=nx.Graph())
    lcc_nodes = max(nx.connected_components(G_raw), key=len)
    G_sub = G_raw.subgraph(lcc_nodes).copy()
    
    rng = random.Random(seed)
    start_node = list(G_sub.nodes())[seed % len(G_sub.nodes())]
    sampled_nodes = {start_node}
    queue = deque([start_node])
    
    while queue and len(sampled_nodes) < target_nodes:
        current = queue.popleft()
        neighbors = list(G_sub.neighbors(current))
        rng.shuffle(neighbors)
        
        for neighbor in neighbors:
            if neighbor not in sampled_nodes:
                sampled_nodes.add(neighbor)
                queue.append(neighbor)
                if len(sampled_nodes) >= target_nodes:
                    break
                    
    return nx.convert_node_labels_to_integers(G_sub.subgraph(sampled_nodes).copy())

# ==========================================
# 3. GPU-Accelerated Replicator Dynamics
# ==========================================
def run_dynamics_gpu(A_sparse, is_committed, omega_tensor, N, ENSEMBLE_SIZE, 
                     alpha=0.5, beta=3.0, c0=1.0, b=0.3, theta=0.3, sigma=15.0, max_steps=5000):
    
    states = torch.zeros((N, ENSEMBLE_SIZE), dtype=torch.float32, device=DEVICE)
    states = torch.where(is_committed, 1.0, states)

    # Circular Buffer for m(t) - Table III Stopping Criterion
    m_history = torch.zeros((500, ENSEMBLE_SIZE), dtype=torch.float32, device=DEVICE)

    for t in range(max_steps):
        active_sum = torch.sparse.mm(A_sparse, states)
        silent_sum = torch.sparse.mm(A_sparse, 1.0 - states)

        phi_1 = (active_sum + omega_tensor) / (active_sum + alpha * silent_sum + omega_tensor + 1e-12)
        phi_0 = (alpha * active_sum + omega_tensor) / (alpha * active_sum + silent_sum + omega_tensor + 1e-12)
        phi = torch.where(states == 1.0, phi_1, phi_0)

        F = (1 - b) * phi - c0 * torch.exp(-beta * phi)
        H = 1.0 / (1.0 + torch.exp(-sigma * (phi - theta)))
        prob_express = (1.0 / (1.0 + torch.exp(-sigma * F))) * H

        rand_draw = torch.rand((N, ENSEMBLE_SIZE), device=DEVICE)
        new_states = torch.where(prob_express > rand_draw, 1.0, 0.0)
        
        new_states = torch.where(is_committed, 1.0, new_states)
        states = new_states

        # Calculate Macroscopic State m(t)
        m_t = states.mean(dim=0)
        
        # Check |m(t) - m(t-500)| < 10^-5
        if t >= 500:
            m_past = m_history[t % 500]
            if torch.all(torch.abs(m_t - m_past) < 1e-5):
                break
                
        # Update Circular Buffer
        m_history[t % 500] = m_t

    return states.mean(dim=0).cpu().numpy()

def theoretical_zc(k_hub, k_mean, beta, c0, b, theta):
    M = c0 * np.exp(-beta * theta) - (1 - b) * theta
    if M <= 0:
        return None, M  
    zc = (k_mean * M) / (k_hub + k_mean * M)
    return zc, M

# ==========================================
# 4. Main GPU Workflow
# ==========================================
def main():
    file_path = "soc-political-retweet.edges"
    try:
        G = snowball_sampling(file_path, target_nodes=6000, seed=MASTER_SEED)
    except FileNotFoundError:
        print("[!] Error: Dataset not found. Please ensure 'soc-political-retweet.edges' is in the working directory.")
        return

    N = G.number_of_nodes()
    
    print("[*] Computing Network Centralities... (Please wait)")
    deg_cent = nx.degree_centrality(G)
    bet_cent = nx.betweenness_centrality(G, k=min(100, N), seed=MASTER_SEED)
    pr_cent = nx.pagerank(G)

    nodes_deg = sorted(deg_cent, key=deg_cent.get, reverse=True)
    nodes_bet = sorted(bet_cent, key=bet_cent.get, reverse=True)
    nodes_pr = sorted(pr_cent, key=pr_cent.get, reverse=True)
    nodes_all = list(G.nodes())

    A_scipy = nx.to_scipy_sparse_array(G, format='coo')
    i = torch.LongTensor(np.vstack((A_scipy.row, A_scipy.col)))
    v = torch.FloatTensor(A_scipy.data)
    A_sparse = torch.sparse_coo_tensor(i, v, torch.Size(A_scipy.shape)).to(DEVICE)
    
    deg_tensor = torch.sparse.sum(A_sparse, dim=1).to_dense().unsqueeze(1)
    deg_tensor[deg_tensor == 0] = 1.0
    k_mean = deg_tensor.mean().item()

    # Model Parameters
    ALPHA = 0.5; BETA = 3.0; C0 = 1.0; B_PAYOFF = 0.3; THETA = 0.3; SIGMA = 15.0     
    z_vals = np.linspace(0.0, 0.25, 26) 
    ENSEMBLE_SIZE = 1000 

    master_rng = np.random.default_rng(MASTER_SEED)
    realization_seeds = master_rng.integers(0, 9999999, size=ENSEMBLE_SIZE)

    raw_results = []
    print(f"\n[*] Launching Massively Parallel GPU Simulations (Ensemble = {ENSEMBLE_SIZE})...")
    start_time = time.time()

    for z in z_vals:
        print(f"    -> Processing z = {z:.2f}")
        num_committed = int(z * N)
        omega = (z * deg_tensor) / (k_mean * (1 - z) + 1e-12)
        
        is_committed_deg = torch.zeros((N, ENSEMBLE_SIZE), dtype=torch.bool, device=DEVICE)
        is_committed_bet = torch.zeros((N, ENSEMBLE_SIZE), dtype=torch.bool, device=DEVICE)
        is_committed_pr = torch.zeros((N, ENSEMBLE_SIZE), dtype=torch.bool, device=DEVICE)
        is_committed_rand = torch.zeros((N, ENSEMBLE_SIZE), dtype=torch.bool, device=DEVICE)

        if num_committed > 0:
            is_committed_deg[nodes_deg[:num_committed], :] = True
            is_committed_bet[nodes_bet[:num_committed], :] = True
            is_committed_pr[nodes_pr[:num_committed], :] = True
            
            for e in range(ENSEMBLE_SIZE):
                rng_cpu = np.random.default_rng(realization_seeds[e])
                c_rand = rng_cpu.choice(nodes_all, num_committed, replace=False)
                is_committed_rand[c_rand, e] = True

        theta_deg = run_dynamics_gpu(A_sparse, is_committed_deg, omega, N, ENSEMBLE_SIZE, alpha=ALPHA)
        theta_bet = run_dynamics_gpu(A_sparse, is_committed_bet, omega, N, ENSEMBLE_SIZE, alpha=ALPHA)
        theta_pr = run_dynamics_gpu(A_sparse, is_committed_pr, omega, N, ENSEMBLE_SIZE, alpha=ALPHA)
        theta_rand = run_dynamics_gpu(A_sparse, is_committed_rand, omega, N, ENSEMBLE_SIZE, alpha=ALPHA)

        for e in range(ENSEMBLE_SIZE):
            seed_val = int(realization_seeds[e])
            raw_results.append({'strategy': 'Degree', 'z': z, 'alpha': ALPHA, 'seed': seed_val, 'final_theta': theta_deg[e]})
            raw_results.append({'strategy': 'Betweenness', 'z': z, 'alpha': ALPHA, 'seed': seed_val, 'final_theta': theta_bet[e]})
            raw_results.append({'strategy': 'PageRank', 'z': z, 'alpha': ALPHA, 'seed': seed_val, 'final_theta': theta_pr[e]})
            raw_results.append({'strategy': 'Random', 'z': z, 'alpha': ALPHA, 'seed': seed_val, 'final_theta': theta_rand[e]})

    end_time = time.time()
    print(f"\n[+] Total GPU Simulation Time: {end_time - start_time:.2f} seconds.")

    # ==========================================
    # 5. Data Aggregation & CSV Export
    # ==========================================
    df_raw = pd.DataFrame(raw_results)
    stats_df = df_raw.groupby(['z', 'strategy'])['final_theta'].agg(
        mean_theta='mean', std_theta='std', count='count').reset_index()
    stats_df['CI95'] = 1.96 * (stats_df['std_theta'] / np.sqrt(stats_df['count']))
    stats_df['CI95'] = stats_df['CI95'].fillna(0)

    df_final = pd.merge(df_raw, stats_df[['z', 'strategy', 'mean_theta', 'std_theta', 'CI95']], 
                        on=['z', 'strategy'], how='left')
    df_final = df_final[['strategy', 'z', 'alpha', 'seed', 'final_theta', 'mean_theta', 'std_theta', 'CI95']]
    
    df_final.to_csv('raw_results.csv', index=False)

    # ==========================================
    # 6. Reviewer Analysis Report (Console Print)
    # ==========================================
    z_ref = 0.03
    hub_nodes_ref = nodes_deg[:max(1, int(z_ref * N))]
    k_hub = np.mean([G.degree(n) for n in hub_nodes_ref])
    zc_theory, M_val = theoretical_zc(k_hub, k_mean, BETA, C0, B_PAYOFF, THETA)

    empirical_zc = None
    deg_stats = stats_df[stats_df['strategy'] == 'Degree']
    for z_val, mean_theta in zip(deg_stats['z'], deg_stats['mean_theta']):
        if mean_theta >= 0.5:  # Symmetry breaking threshold
            empirical_zc = z_val
            break

    print("\n" + "="*50)
    print(" 📊 REVIEWER ANALYSIS REPORT (Copy to Paper) ")
    print("="*50)
    print(f"[*] Mean degree of network <k> = {k_mean:.2f}")
    print(f"[*] Mean degree of targeted hubs (k_hub) = {k_hub:.2f}")
    print(f"[*] M = {M_val:.4f}")
    print(f"[*] Theoretical z_c (Eq. 13) = {zc_theory:.4f}")
    
    if empirical_zc is not None and zc_theory is not None:
        rel_error = abs(zc_theory - empirical_zc) / empirical_zc * 100
        print(f"[*] Empirical z_c from simulation = {empirical_zc:.4f}")
        print(f"[*] Relative Error (Theory vs Empiric) = {rel_error:.2f}%")
    print("="*50 + "\n")

    # ==========================================
    # 7. High-Quality Plotting
    # ==========================================
    plot_data = {
        'Degree': {'color': 'red', 'marker': 'o', 'linestyle': '-', 'label': 'Degree Centrality (Hubs)'},
        'Betweenness': {'color': 'blue', 'marker': 's', 'linestyle': '-', 'label': 'Betweenness Centrality'},
        'PageRank': {'color': 'green', 'marker': '^', 'linestyle': '-', 'label': 'PageRank'},
        'Random': {'color': 'black', 'marker': '', 'linestyle': '--', 'label': 'Random Placement'}
    }

    plt.figure(figsize=(7, 5))
    for strat, style in plot_data.items():
        strat_data = stats_df[stats_df['strategy'] == strat]
        z_arr = strat_data['z'].values
        mean_arr = strat_data['mean_theta'].values
        ci_arr = strat_data['CI95'].values
        
        plt.plot(z_arr, mean_arr, color=style['color'], marker=style['marker'], 
                 linestyle=style['linestyle'], linewidth=2.5, markersize=5, label=style['label'])
        plt.fill_between(z_arr, mean_arr - ci_arr, mean_arr + ci_arr, 
                         color=style['color'], alpha=0.15, edgecolor='none')

    plt.axhline(y=0.8, color='gray', linestyle=':', alpha=0.8, linewidth=2, label='Consensus Threshold (0.8)')

    if zc_theory is not None:
        plt.axvline(x=zc_theory, color='purple', linestyle='-.', linewidth=2.5,
                    label=fr'Theoretical $z_c$ $\approx {zc_theory:.3f}$')

    plt.title("Effect of Placement Strategies on Final Consensus", fontweight='bold', pad=15)
    plt.xlabel(r"Fraction of Committed Agents ($z$)", fontweight='bold', fontsize=11)
    plt.ylabel(r"Final Expressive Fraction ($\Theta_\infty$)", fontweight='bold', fontsize=11)
    
    plt.xlim(0, 0.25)
    plt.ylim(-0.02, 1.05)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', framealpha=0.95, edgecolor='black', fancybox=False, fontsize=9.5)
    plt.tight_layout()
    
    plt.savefig('Fig_Placement.png', dpi=400, bbox_inches='tight')
    plt.savefig('Fig_Placement.pdf', format='pdf', bbox_inches='tight')
    print("[+] Success! Figure saved and raw data exported.")

if __name__ == "__main__":
    main()
