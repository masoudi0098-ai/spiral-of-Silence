import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
})


def load_network_safe(filepath, target_nodes=5000):
    print(f"[*] Loading network from {filepath}...")
    G = nx.Graph()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('%'): continue
            parts = line.split(',') if ',' in line else line.split()
            if len(parts) >= 2: G.add_edge(parts[0], parts[1])
                
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

def run_dynamics(A, committed_nodes, alpha=0.5, beta=3.0, c0=1.0, gamma=15.0, max_steps=50):
    N = A.shape[0]
    states = np.zeros(N)
    states[committed_nodes] = 1.0
    
    deg = np.sum(A, axis=1)
    deg[deg == 0] = 1 
    
    for _ in range(max_steps):
        prev_states = states.copy()
        
        k1 = A @ states
        phi = (alpha * k1) / (alpha * k1 + (deg - k1) + 1e-8)
        
        cost = c0 * np.exp(-beta * phi)
        prob_express = 1.0 / (1.0 + np.exp(-gamma * (phi - cost)))
        
        states = np.where(prob_express > 0.5, 1.0, 0.0)
        
        states[committed_nodes] = 1.0
        
        if np.array_equal(states, prev_states):
            break
            
    return np.mean(states)


file_path = "soc-political-retweet.edges"
try:
    G = load_network_safe(file_path, target_nodes=2000)
    A = nx.to_numpy_array(G)
    N = G.number_of_nodes()
    
    print("[*] Computing Network Centralities... (Please wait)")
    deg_cent = nx.degree_centrality(G)
    bet_cent = nx.betweenness_centrality(G, k=min(100, N))
    pr_cent = nx.pagerank(G)
    
    nodes_deg = sorted(deg_cent, key=deg_cent.get, reverse=True)
    nodes_bet = sorted(bet_cent, key=bet_cent.get, reverse=True)
    nodes_pr = sorted(pr_cent, key=pr_cent.get, reverse=True)
    nodes_all = list(G.nodes())
    
    z_vals = np.linspace(0.01, 0.25, 20)
    results = {'Degree': [], 'Betweenness': [], 'PageRank': [], 'Random': []}
    
    print("[*] Running Simulations for different Z values...")
    for z in z_vals:
        num_committed = int(z * N)
        
        c_deg = nodes_deg[:num_committed]
        c_bet = nodes_bet[:num_committed]
        c_pr = nodes_pr[:num_committed]
        c_rand = random.sample(nodes_all, num_committed)
        
        results['Degree'].append(run_dynamics(A, c_deg))
        results['Betweenness'].append(run_dynamics(A, c_bet))
        results['PageRank'].append(run_dynamics(A, c_pr))
        results['Random'].append(run_dynamics(A, c_rand))

    plt.figure(figsize=(6, 4.5))
    
    plt.plot(z_vals, results['Degree'], 'r-o', linewidth=2, markersize=5, label='Degree Centrality (Hubs)')
    plt.plot(z_vals, results['Betweenness'], 'b-s', linewidth=1.5, markersize=4, label='Betweenness Centrality')
    plt.plot(z_vals, results['PageRank'], 'g-^', linewidth=1.5, markersize=4, label='PageRank')
    plt.plot(z_vals, results['Random'], 'k--', linewidth=1.5, label='Random Placement')
    
    plt.title("Effect of Placement Strategies on Final Consensus", fontweight='bold')
    plt.xlabel(r"Fraction of Committed Agents ($z$)", fontweight='bold')
    plt.ylabel(r"Final Expressive Fraction ($\Theta_\infty$)", fontweight='bold')
    plt.axhline(y=0.8, color='gray', linestyle=':', alpha=0.7, label='Consensus Threshold')
    
    plt.xlim(0, 0.25)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', framealpha=0.9, edgecolor='black', fancybox=False)
    
    plt.tight_layout()
    plt.savefig('Fig_Placement_Strategies.png')
    print("[+] Success! Figure saved as 'Fig_Placement_Strategies.png'")
    plt.show()

except Exception as e:
    print(f"[!] Error: {e}\nfile is not in path.")
