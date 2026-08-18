import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})


def load_network_safe(filepath, target_nodes=6000):
    print(f"[*] Loading network from {filepath}...")
    G = nx.Graph()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('%'):
                continue
            parts = line.split(',') if ',' in line else line.split()
            if len(parts) >= 2:
                G.add_edge(parts[0], parts[1])
    G = G.to_undirected()
    lcc = max(nx.connected_components(G), key=len)
    G_sub = G.subgraph(lcc).copy()
    if G_sub.number_of_nodes() > target_nodes:
        degrees = dict(G_sub.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:target_nodes]
        G_final = G_sub.subgraph(top_nodes).copy()
    else:
        G_final = G_sub
    return nx.convert_node_labels_to_integers(G_final)


def compute_omega(z, k, k_mean):
    """
    Eq. (1): encounter frequency of a committed agent in the
    neighborhood of a node of degree k.
        Omega_k(z) = z*k / (<k>*(1-z))
    k may be a scalar or a numpy array (per-node degree).
    """
    z = min(max(z, 0.0), 0.999)  # keep away from the z=1 singularity
    return (z * k) / (k_mean * (1 - z) + 1e-12)


def run_dynamics(A, committed_nodes, z, alpha=0.5, beta=3.0, c0=1.0,
                  b=0.3, theta=0.3, sigma=15.0, max_steps=50, seed=None):
    """
    Agent-based implementation of Eqs. (4), (5), (6), (8):

      - A_eff[i,j] = A[i,j] * V[i,j]     (Eq. 5: V=1 if same strategy,
                                           V=alpha if opinion-discordant)
      - phi_i = (sum_j A_eff[i,j]*x_j + Omega_i*x_committed_proxy)
                / (sum_j A_eff[i,j] + Omega_i)         (Eq. 4)
      - F(phi) = (1-b)*phi - c0*exp(-beta*phi)          (pure payoff, Eq. 9)
      - H(phi-theta) = sigmoid(sigma*(phi-theta))       (Eq. 6, cognitive gate)
      - expression probability = sigmoid(sigma*F) * H   (Eq. 8, discretized)

    theta and b are NOT present in the original repository code and
    must be chosen/justified by the authors (see note below).
    """
    rng = np.random.default_rng(seed)
    N = A.shape[0]
    deg = A.sum(axis=1)
    deg[deg == 0] = 1
    k_mean = deg.mean()

    states = np.zeros(N)
    states[committed_nodes] = 1.0
    is_committed = np.zeros(N, dtype=bool)
    is_committed[committed_nodes] = True

    omega = compute_omega(z, deg, k_mean)  # Eq. (1), per-node using own degree

    for _ in range(max_steps):
        prev_states = states.copy()

        # Eq. (5): effective adjacency via the visibility matrix
        same_strategy = (states[:, None] == states[None, :]).astype(float)
        V = np.where(same_strategy == 1.0, 1.0, alpha)
        A_eff = A * V

        # Eq. (4): local perceptual field, including committed-agent exposure
        numerator = A_eff @ states + omega * 1.0     # committed encounters are always expressive (x=1)
        denominator = A_eff.sum(axis=1) + omega
        denominator[denominator == 0] = 1e-8
        phi = numerator / denominator

        F = (1 - b) * phi - c0 * np.exp(-beta * phi)          # Eq. (9)
        H = 1.0 / (1.0 + np.exp(-sigma * (phi - theta)))       # Eq. (6)
        prob_express = (1.0 / (1.0 + np.exp(-sigma * F))) * H  # discretized Eq. (8)

        rand_draw = rng.random(N)
        new_states = np.where(prob_express > rand_draw, 1.0, 0.0)
        new_states[is_committed] = 1.0
        states = new_states

        if np.array_equal(states, prev_states):
            break

    return np.mean(states)


def theoretical_zc(k_hub, k_mean, beta, c0, b, theta):
    """
    Eq. (12):
        M  = c0*exp(-beta*theta) - (1-b)*theta
        zc = (<k>*M) / (k_hub + <k>*M)
    """
    M = c0 * np.exp(-beta * theta) - (1 - b) * theta
    if M <= 0:
        return None, M  # Eq. (12) requires M > 0 for a meaningful zc in (0,1)
    zc = (k_mean * M) / (k_hub + k_mean * M)
    return zc, M


# =================================================================
# MAIN
# =================================================================
file_path = "soc-political-retweet.edges"
try:

    G = load_network_safe(file_path, target_nodes=6000)
    A = nx.to_numpy_array(G)
    N = G.number_of_nodes()
    k_mean = np.array(A.sum(axis=1)).mean()

    print("[*] Computing Network Centralities... (Please wait)")
    deg_cent = nx.degree_centrality(G)
    bet_cent = nx.betweenness_centrality(G, k=min(100, N))
    pr_cent = nx.pagerank(G)

    nodes_deg = sorted(deg_cent, key=deg_cent.get, reverse=True)
    nodes_bet = sorted(bet_cent, key=bet_cent.get, reverse=True)
    nodes_pr = sorted(pr_cent, key=pr_cent.get, reverse=True)
    nodes_all = list(G.nodes())

    # ---- Model parameters (must match Sec. 4.1 / 4.1.1 of the paper) ----
    ALPHA = 0.5
    BETA = 3.0     # pick the representative value used for Fig. 4 (paper sweeps beta in {1,2,2.5,3,4.5,5})
    C0 = 1.0
    B_PAYOFF = 0.3   # NEW parameter, not in original code -- must be justified/reported
    THETA = 0.3      # NEW parameter, not in original code -- must be justified/reported
    SIGMA = 15.0     # renamed from "gamma" to match paper notation

    z_vals = np.linspace(0.01, 0.25, 20)
    results = {'Degree': [], 'Betweenness': [], 'PageRank': [], 'Random': []}

    print("[*] Running Simulations for different Z values...")
    for z in z_vals:
        num_committed = max(1, int(z * N))
        c_deg = [G.nodes[n] if False else int(n) for n in nodes_deg[:num_committed]]  # keep as int indices
        c_deg = nodes_deg[:num_committed]
        c_bet = nodes_bet[:num_committed]
        c_pr = nodes_pr[:num_committed]
        c_rand = random.sample(nodes_all, num_committed)

        results['Degree'].append(run_dynamics(A, c_deg, z, ALPHA, BETA, C0, B_PAYOFF, THETA, SIGMA))
        results['Betweenness'].append(run_dynamics(A, c_bet, z, ALPHA, BETA, C0, B_PAYOFF, THETA, SIGMA))
        results['PageRank'].append(run_dynamics(A, c_pr, z, ALPHA, BETA, C0, B_PAYOFF, THETA, SIGMA))
        results['Random'].append(run_dynamics(A, c_rand, z, ALPHA, BETA, C0, B_PAYOFF, THETA, SIGMA))

    # ---- Theoretical zc from Eq. (12), evaluated at the mean degree of the
    #      hub set actually targeted by the Degree-Centrality strategy ----
    # Use a representative z near the observed tipping region to pick k_hub,
    # since the hub set size (num_committed) depends on z.
    z_ref = 0.03
    num_committed_ref = max(1, int(z_ref * N))
    hub_nodes_ref = nodes_deg[:num_committed_ref]
    k_hub = np.mean([A[n].sum() for n in hub_nodes_ref])

    zc_theory, M_val = theoretical_zc(k_hub, k_mean, BETA, C0, B_PAYOFF, THETA)
    print(f"[*] Mean degree of network <k> = {k_mean:.2f}")
    print(f"[*] Mean degree of targeted hubs k_hub = {k_hub:.2f}")
    print(f"[*] M = {M_val:.4f}")
    print(f"[*] Theoretical z_c (Eq. 12) = {zc_theory}")

    # ---- Plot ----
    plt.figure(figsize=(6, 4.5))
    plt.plot(z_vals, results['Degree'], 'r-o', linewidth=2, markersize=5, label='Degree Centrality (Hubs)')
    plt.plot(z_vals, results['Betweenness'], 'b-s', linewidth=1.5, markersize=4, label='Betweenness Centrality')
    plt.plot(z_vals, results['PageRank'], 'g-^', linewidth=1.5, markersize=4, label='PageRank')
    plt.plot(z_vals, results['Random'], 'k--', linewidth=1.5, label='Random Placement')
    plt.axhline(y=0.8, color='gray', linestyle=':', alpha=0.7, label='Consensus Threshold')

    if zc_theory is not None:
        plt.axvline(x=zc_theory, color='purple', linestyle='-.', linewidth=1.5,
                    label=fr'Theoretical $z_c$ (Eq. 12) $\approx {zc_theory:.3f}$')

    plt.title("Effect of Placement Strategies on Final Consensus", fontweight='bold')
    plt.xlabel(r"Fraction of Committed Agents ($z$)", fontweight='bold')
    plt.ylabel(r"Final Expressive Fraction ($\Theta_\infty$)", fontweight='bold')
    plt.xlim(0, 0.25)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', framealpha=0.9, edgecolor='black', fancybox=False)
    plt.tight_layout()
    plt.savefig('Fig_Placement_Strategies_with_zc.png')
    print("[+] Success! Figure saved as 'Fig_Placement_Strategies_with_zc.png'")
    plt.show()

except Exception as e:
    print(f"[!] Error: {e}\nfile is not in path.")
