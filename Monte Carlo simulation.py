import networkx as nx
import numpy as np
import torch
import matplotlib.pyplot as plt

# ==========================================
# Figure Settings (Times New Roman, 300 DPI)
# ==========================================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 300
})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def simulate_accurate_contagion(topology, N=2000, k_avg=10, z=0.15, alpha=0.6, c0=0.8, beta=2.5, b=0.1, theta=0.20, max_mcs=60):
    if topology == 'BA':
        G = nx.barabasi_albert_graph(N, k_avg // 2)
    elif topology == 'WS':
        G = nx.watts_strogatz_graph(N, k_avg, 0.1)
    else:
        G = nx.erdos_renyi_graph(N, k_avg / N)
        
    degrees = dict(G.degree())
    hubs = sorted(degrees, key=degrees.get, reverse=True)[:int(z * N)]
    
    A = torch.tensor(nx.to_numpy_array(G), dtype=torch.float32, device=device)
    states = torch.zeros(N, dtype=torch.float32, device=device)
    states[hubs] = 1.0
    
    committed_mask = torch.zeros(N, dtype=torch.bool, device=device)
    committed_mask[hubs] = True
    
    history = [states.mean().item()]
    
    for _ in range(max_mcs):
        k_1 = torch.matmul(A, states)
        k_0 = torch.matmul(A, 1.0 - states)
        
        phi = (alpha * k_1) / (alpha * k_1 + k_0 + 1e-6)
        
        c_i = c0 * torch.exp(-beta * phi)
        payoff_express = (1.0 - b) * phi - c_i
        
        threshold_passed = (phi >= theta)
        
        activation_prob = torch.sigmoid(5.0 * payoff_express)
        random_draw = torch.rand(N, device=device)
        
        new_states = torch.where(threshold_passed & (random_draw < activation_prob), 1.0, 0.0)
        states = torch.where(committed_mask, 1.0, new_states)
        
        history.append(states.mean().item())
        
    return history


topologies = ['BA', 'WS', 'ER']
results = {t: [] for t in topologies}
N_runs = 1000 

print("Running Monte Carlo simulation ...")
for t in topologies:
    for run in range(N_runs):
        hist = simulate_accurate_contagion(topology=t, N=2000, z=0.15, alpha=0.6)
        results[t].append(hist)
    print(f"Topology {t} computed successfully.")


plt.figure(figsize=(9, 6))
colors = {'BA': '#d62728', 'WS': '#2ca02c', 'ER': '#1f77b4'}
labels = {'BA': 'Scale-Free (BA)', 'WS': 'Small-World (WS)', 'ER': 'Random (ER)'}
styles = {'BA': '-', 'WS': '--', 'ER': '-.'}

for t in topologies:
    data = np.array(results[t])
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    plt.plot(mean, label=labels[t], color=colors[t], linestyle=styles[t], lw=2.5)
    plt.fill_between(range(len(mean)), mean - std, mean + std, color=colors[t], alpha=0.15, edgecolor='none')
    
    diff = np.diff(mean)
    max_jump_idx = np.argmax(diff)
    
    if diff[max_jump_idx] > 0.02: 
        cp_x = max_jump_idx + 1 # جبران شیفت ایندکس مشتق
        cp_y = mean[cp_x]
        
        offset_y = -0.15 if t == 'WS' else 0.1
        offset_x = 5 if t == 'BA' else -5
        
        
plt.xlabel('Monte Carlo Steps (MCS)', fontsize=11)
plt.ylabel(r'Fraction of Expressive Population ($\Theta$)', fontsize=11)
plt.title('Discontinuous Phase Transitions with Critical Point', pad=15, fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.6)
plt.xlim(0, 50)
plt.ylim(0, 1.05)
plt.legend(loc='lower right', frameon=True, shadow=True)
plt.tight_layout()

plt.savefig('Fig_Accurate_Transitions_Annotated.png', dpi=300, bbox_inches='tight')
plt.show()
