import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300,
})


alpha = 0.50
c0 = 1.0
gamma = 15.0

def F(theta, beta):
    phi = (alpha * theta) / (alpha * theta + (1 - theta) + 1e-8)
    cost = c0 * np.exp(-beta * phi)
    return 1.0 / (1.0 + np.exp(-gamma * (phi - cost)))

theta_vals = np.linspace(0.0, 1.0, 500)
betas_to_test = [1.0, 2.5, 4.5]
colors = ['#1f77b4', '#9467bd', '#2ca02c']
labels = [r'$\beta = 1.0$ (Linear)', r'$\beta = 2.5$ (Bistable)', r'$\beta = 4.5$ (Avalanche)']

legend_kwargs = {
    'loc': 'best',
    'framealpha': 0.9,
    'edgecolor': 'black',
    'fancybox': False,
    'handlelength': 1.2,
    'handletextpad': 0.4,
    'borderpad': 0.4,
    'labelspacing': 0.3
}

fig_a, ax_a = plt.subplots(figsize=(4.5, 3.5))
ax_a.plot(theta_vals, theta_vals, 'k:', lw=1.2, label=r'Fixed Points ($y=x$)')

for b, color, label in zip(betas_to_test, colors, labels):
    ax_a.plot(theta_vals, F(theta_vals, b), color=color, label=label, lw=1.5)

ax_a.set_title(r"(a) Macroscopic Mapping Function", fontweight='bold')
ax_a.set_xlabel(r'Current State ($\theta$)')
ax_a.set_ylabel(r'Next State $F(\theta)$')
ax_a.set_xlim(0, 1)
ax_a.set_ylim(0, 1)
ax_a.grid(True, linestyle='--', alpha=0.4)
ax_a.legend(**legend_kwargs)

plt.tight_layout()
plt.savefig('Panel_A_Mapping.png')
plt.show()

fig_b, ax_b = plt.subplots(figsize=(4.5, 3.5))

for b, color, label in zip(betas_to_test, colors, labels):
    f_vals = F(theta_vals, b)
    driving_force = f_vals - theta_vals
    V = -cumulative_trapezoid(driving_force, theta_vals, initial=0)
    V_normalized = V - np.min(V)
    ax_b.plot(theta_vals, V_normalized, color=color, label=label, lw=1.5)

ax_b.set_title(r"(b) Potential Landscape $V(\theta)$", fontweight='bold')
ax_b.set_xlabel(r'State ($\theta$)')
ax_b.set_ylabel(r'Pseudo-Potential $V(\theta)$')
ax_b.set_xlim(0, 1)
ax_b.grid(True, linestyle='--', alpha=0.4)
ax_b.legend(**legend_kwargs)

plt.tight_layout()
plt.savefig('Panel_B_Potential.png')
plt.show()


unstable_b, unstable_t = [], []

beta_vals = np.linspace(0.5, 6.0, 800)
t_vals = np.linspace(0, 1, 2000)

for b in beta_vals:
    f_vals = F(t_vals, b)
    diff = f_vals - t_vals
    idx = np.where(np.diff(np.sign(diff)))[0]
    
    for i in idx:
        t_root = t_vals[i] - diff[i] * (t_vals[i+1] - t_vals[i]) / (diff[i+1] - diff[i])
        slope = (f_vals[i+1] - f_vals[i]) / (t_vals[i+1] - t_vals[i])
        
        if slope >= 1.0:
            unstable_b.append(b)
            unstable_t.append(t_root)

fig_c, ax_c = plt.subplots(figsize=(5.5, 4))

ax_c.axvspan(0.5, 1.8, facecolor="#bfbcee", alpha=0.7, label='Absorbing Phase')
ax_c.axvspan(1.8, 4.0, facecolor="#e9a5a5", alpha=0.7, label='Bistable Phase')
ax_c.axvspan(4.0, 6.0, facecolor="#aaf1aa", alpha=0.7, label='Consensus Phase')

sort_idx = np.argsort(unstable_b)
ax_c.plot(np.array(unstable_b)[sort_idx], np.array(unstable_t)[sort_idx], 
        'r--', linewidth=1.5, label='Unstable Repeller', zorder=6)

ax_c.set_title(r"(c) Bifurcation Diagram ($\theta^*$ vs $\beta$)", fontweight='bold')
ax_c.set_xlabel(r'Cost Decay Rate ($\beta$)')
ax_c.set_ylabel(r'Equilibrium States ($\theta^*$)')
ax_c.set_xlim(0.5, 6.0)
ax_c.set_ylim(0, 1.0)
ax_c.grid(True, linestyle='--', alpha=0.4)

ax_c.legend(loc='best', framealpha=0.9, edgecolor='black', 
            fancybox=False, handlelength=1.2, handletextpad=0.4, 
            borderpad=0.4, labelspacing=0.3)

plt.tight_layout()
plt.savefig('Panel_C_Bifurcation_NoAttractor.png')
plt.show()
