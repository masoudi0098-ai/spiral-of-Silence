import networkx as nx
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from collections import deque
import random
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================
# 1. Configuration
# ==========================================
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12,
    "figure.dpi": 300
})

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    DEVICE = torch.device('cpu')
    torch.set_num_threads(max(1, torch.get_num_threads()))

print(f"[*] Engine running strictly optimized on: {DEVICE}")

# ==========================================
# 2. Network Sampling
# ==========================================
def snowball_sampling(filepath, target_nodes=6000, seed=42):
    G_raw = nx.read_edgelist(filepath, delimiter=',', nodetype=int,
                             data=False, create_using=nx.Graph())
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
# 3. JIT Fused Physics Kernel (ULTRA FAST)
# ==========================================
# این بخش تمام عملیات ریاضی را به یک هسته واحد در GPU تبدیل می‌کند
@torch.jit.script
def update_states_kernel(states: torch.Tensor, 
                         active_sum: torch.Tensor, 
                         degree_tensor: torch.Tensor, 
                         alpha_t: torch.Tensor,
                         c1: torch.Tensor, 
                         alpha_deg: torch.Tensor, 
                         sigma_t: torch.Tensor, 
                         theta_t: torch.Tensor,
                         c0_t: torch.Tensor, 
                         beta_t: torch.Tensor, 
                         one_m_b: torch.Tensor, 
                         dt_t: torch.Tensor,
                         rand_tensor: torch.Tensor) -> torch.Tensor:
    
    # محاسبات phi با کمترین استفاده از حافظه
    denom1 = (active_sum * c1) + alpha_deg + 1e-12
    phi_1 = active_sum / denom1
    
    denom2 = -(active_sum * c1) + degree_tensor + 1e-12
    phi_0 = (active_sum * alpha_t) / denom2
    
    phi = torch.where(states == 1.0, phi_1, phi_0)
    
    # توابع هزینه و احتمالات
    H_gate = torch.sigmoid(sigma_t * (phi - theta_t))
    net_payoff = one_m_b * phi - c0_t * torch.exp(-beta_t * phi)
    net_payoff_H_dt = net_payoff * H_gate * dt_t
    
    # ترکیب احتمالات در یک خط بدون نیاز به ماسک‌گذاری متوالی
    p1_next = torch.where(
        states == 1.0,
        1.0 - torch.clamp(torch.relu(-net_payoff_H_dt), min=0.0, max=1.0),
        torch.clamp(torch.relu(net_payoff_H_dt), min=0.0, max=1.0)
    )
    
    return (rand_tensor < p1_next).float()

# ==========================================
# 4. Asynchronous GPU Engine
# ==========================================
@torch.inference_mode()
def run_simulation(z_vals, alpha_vals, A_sparse, degree_tensor,
                   sorted_nodes_tensor, N, ENSEMBLE_SIZE, params,
                   max_steps=5000, tol=1e-5):
                   
    theta_threshold, sigma, c0, beta, b, dt = params
    res_z = len(z_vals)
    res_alpha = len(alpha_vals)
    heatmap_data = np.zeros((res_alpha, res_z))

    # تعریف ثابت‌ها به عنوان تنسورهای تک‌بعدی برای سرعت در JIT
    sigma_t = torch.tensor(sigma,  dtype=torch.float32, device=DEVICE)
    theta_t = torch.tensor(theta_threshold, dtype=torch.float32, device=DEVICE)
    c0_t    = torch.tensor(c0,     dtype=torch.float32, device=DEVICE)
    beta_t  = torch.tensor(beta,   dtype=torch.float32, device=DEVICE)
    one_m_b = torch.tensor(1.0-b,  dtype=torch.float32, device=DEVICE)
    dt_t    = torch.tensor(dt,     dtype=torch.float32, device=DEVICE)

    states      = torch.zeros((N, ENSEMBLE_SIZE), dtype=torch.float32, device=DEVICE)
    rand_tensor = torch.empty((N, ENSEMBLE_SIZE), dtype=torch.float32, device=DEVICE)
    
    # بافر حلقوی برای معیار توقف (کاهش 98 درصدی Sync های GPU)
    history_len = 11  # (500 steps // 50) + 1
    m_history = torch.zeros((history_len, ENSEMBLE_SIZE), dtype=torch.float32, device=DEVICE)

    for r_idx, alpha in enumerate(alpha_vals):
        t0 = time.time()
        alpha_t = torch.tensor(alpha, dtype=torch.float32, device=DEVICE)
        c1_t = 1.0 - alpha_t
        alpha_deg = degree_tensor * alpha_t

        for j, z in enumerate(z_vals):
            n_h = int(z * N)
            hubs_indices = sorted_nodes_tensor[:n_h]

            states.zero_()
            m_history.zero_()
            
            if n_h > 0:
                states[hubs_indices, :] = 1.0

            for t in range(max_steps):
                # 1. عملیات فوق‌سریع اسپارس
                active_sum = torch.sparse.mm(A_sparse, states)
                
                # 2. هسته محاسبه JIT (شلیک موازی همه اعداد در یک مرحله)
                rand_tensor.uniform_()
                states = update_states_kernel(
                    states, active_sum, degree_tensor, alpha_t, c1_t, alpha_deg,
                    sigma_t, theta_t, c0_t, beta_t, one_m_b, dt_t, rand_tensor
                )
                
                # 3. بازنشانی تعهد هاب‌ها
                if n_h > 0:
                    states[hubs_indices, :] = 1.0

                # 4. ارزیابی شرط توقف صرفاً هر 50 گام
                if t % 50 == 0:
                    m_t = states.mean(dim=0)
                    idx = (t // 50) % history_len
                    m_history[idx] = m_t
                    
                    if t >= 500:
                        past_idx = (idx + 1) % history_len
                        m_past = m_history[past_idx]
                        if torch.max(torch.abs(m_t - m_past)).item() < tol:
                            break

            heatmap_data[r_idx, j] = states.mean().item()

        # زمان‌سنجی برای هر سطر چاپ می‌شود
        elapsed = time.time() - t0
        print(f"    [+] Row {r_idx+1:02d}/{res_alpha} (α={alpha:.2f}) strictly resolved in {elapsed:.2f}s")

    return heatmap_data

# ==========================================
# 5. Main Controller
# ==========================================
def main():
    filepath = "soc-political-retweet.edges"
    try:
        print("[*] Parsing network graph...")
        G_emp = snowball_sampling(filepath, target_nodes=6000, seed=SEED)
    except FileNotFoundError:
        print("[!] FATAL: dataset missing."); return

    N = G_emp.number_of_nodes()
    degrees_dict = dict(G_emp.degree())
    
    # استخراج دقیق هاب‌ها
    sorted_nodes = sorted(degrees_dict, key=degrees_dict.get, reverse=True)
    sorted_nodes_tensor = torch.tensor(sorted_nodes, dtype=torch.long, device=DEVICE)
    
    degree_array = np.array([G_emp.degree(i) for i in range(N)], dtype=np.float32)
    degree_tensor = torch.tensor(degree_array, dtype=torch.float32, device=DEVICE).unsqueeze(1)

    # استفاده از ساختار COO به هم پیوسته (Coalesced) برای نهایت سرعت در PyTorch
    A_scipy = nx.to_scipy_sparse_array(G_emp, format='coo')
    indices = torch.LongTensor(np.vstack((A_scipy.row, A_scipy.col)))
    values = torch.FloatTensor(A_scipy.data)
    A_sparse = torch.sparse_coo_tensor(indices, values, torch.Size(A_scipy.shape), device=DEVICE).coalesce()

    # EXACT ACADEMIC PARAMETERS
    res = 50
    ENSEMBLE_SIZE = 1000
    z_vals     = np.linspace(0.0, 0.25, res)
    alpha_vals = np.linspace(0.1, 1.0,  res)
    params = (0.30, 15.0, 1.0, 3.0, 0.30, 0.1)

    print(f"[*] Starting JIT-Optimized Ultra-Fast Simulation: {res}x{res} Grid | Ensemble: {ENSEMBLE_SIZE}")
    t0 = time.time()
    
    heatmap_data = run_simulation(z_vals, alpha_vals, A_sparse, degree_tensor,
                                  sorted_nodes_tensor, N, ENSEMBLE_SIZE, params,
                                  max_steps=5000, tol=1e-5)
                                  
    total_minutes = (time.time() - t0) / 60
    print(f"[+] Processing Complete. Total Wall Clock Time: {total_minutes:.2f} minutes.")

    # ==========================================
    # 6. Plot Generation
    # ==========================================
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#000033", "#1f77b4", "#2ca02c", "#6cff56", "#ffff00"]
    cmap = LinearSegmentedColormap.from_list("DarkBlueToYellow", colors, N=256)
    
    im = ax.imshow(heatmap_data,
                   extent=[z_vals.min(), z_vals.max(), alpha_vals.min(), alpha_vals.max()],
                   origin='lower', aspect='auto', cmap=cmap,
                   interpolation='bilinear', vmin=0.0, vmax=1.0)
                   
    ax.contour(z_vals, alpha_vals, heatmap_data,
               levels=[0.2, 0.5, 0.8], colors='red',
               linewidths=1.5, linestyles='dashed')
               
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r'Steady-State Expressive Fraction ($\Theta^*$)', fontsize=13, labelpad=12)
    ax.set_xlabel(r'Committed Agents Fraction ($z$)', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'Visibility Coefficient ($\alpha$)', fontsize=13, fontweight='bold')
    ax.set_title('Phase Diagram: Empirical Twitter Network', fontsize=15, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig('Fig_Diagarm.png', dpi=400, bbox_inches='tight')
    plt.savefig('Fig_Diagram.pdf', format='pdf', bbox_inches='tight')
    print("[+] Plot rendering successful.")

if __name__ == "__main__":
    main()
