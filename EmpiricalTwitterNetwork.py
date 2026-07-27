import networkx as nx
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import gc

# تنظیمات ژورنال
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "figure.dpi": 300})
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_and_sample_empirical_network(filepath, max_nodes=10):
    G_raw = nx.read_edgelist(filepath, delimiter=',', create_using=nx.DiGraph(), nodetype=int, data=False)
    G_undirected = G_raw.to_undirected()
    lcc_nodes = max(nx.connected_components(G_undirected), key=len)
    G_final = nx.convert_node_labels_to_integers(G_undirected.subgraph(lcc_nodes).copy())
    return G_final

file_path = "soc-political-retweet.edges"
G_emp = load_and_sample_empirical_network(file_path)

if G_emp:
    N = G_emp.number_of_nodes()
    A = torch.tensor(nx.to_numpy_array(G_emp), dtype=torch.float32, device=device)
    degrees = dict(G_emp.degree())
    sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
    
    res = 60 
    z_vals = np.linspace(0.01, 0.25, res)
    alpha_vals = np.linspace(0.1, 1.0, res)
    heatmap_data = np.zeros((res, res))
    
    for i, alpha in enumerate(alpha_vals):
        for j, z in enumerate(z_vals):
            hubs = sorted_nodes[:int(z * N)]
            states = torch.zeros(N, device=device)
            if hubs: states[torch.tensor(hubs, device=device)] = 1.0
            
            for _ in range(50):
                phi = (alpha * torch.matmul(A, states)) / (alpha * torch.matmul(A, states) + torch.matmul(A, 1.0 - states) + 1e-6)
                prob = torch.sigmoid(5.0 * ((1.0 - 0.1) * phi - 0.8 * torch.exp(-2.5 * phi)))
                new_states = torch.where((phi >= 0.2) & (torch.rand(N, device=device) < prob), 1.0, 0.0)
                if hubs: new_states[torch.tensor(hubs, device=device)] = 1.0
                states = new_states
            heatmap_data[i, j] = states.mean().item()
        gc.collect()

    plt.figure(figsize=(8, 6))
    im = plt.imshow(heatmap_data, extent=[z_vals.min(), z_vals.max(), alpha_vals.min(), alpha_vals.max()], 
                    origin='lower', aspect='auto', cmap='viridis', interpolation='bilinear')
    
    critical_levels = [0.2, 0.5, 0.8]
    
    contours = plt.contour(z_vals, alpha_vals, heatmap_data, levels=critical_levels, 
                           colors='red', linewidths=1.5, linestyles='dashed')
    
    print(f"Contour lines successfully drawn for steady-state thresholds: {critical_levels}")
    
    plt.colorbar(im, label=r'Steady-State Expressive Fraction ($\Theta^*$)')
    plt.xlabel('Committed Agents Fraction (z)')
    plt.ylabel('Visibility Coefficient (α)')
    plt.title('Phase Diagram: Empirical Twitter Network')
    plt.tight_layout()
    plt.savefig('Final_Phase_Diagram_Red_Contours.png', dpi=300)
    plt.show()
