import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8, "figure.dpi": 300})

def calculate_psychological_cost(phi, c0, beta):
    return c0 * np.exp(-beta * phi)

phi_values = np.linspace(0, 1, 500)

c0_fixed = 1.0  
beta_variants = [1.0, 2.0, 3.0, 5.0] 
colors = ["#083b60", "#3232cf", "#54d7e8", "#26d719"]


plt.figure(figsize=(8, 6))

for beta, col in zip(beta_variants, colors):
    cost_values = calculate_psychological_cost(phi_values, c0_fixed, beta)
    plt.plot(phi_values, cost_values, 
             label=f'$\\beta = {beta}$', 
             color=col, 
             linewidth=2.5)

plt.title(r'Impact of Social Support on Psychological Cost', 
          fontsize=11, fontweight='bold', pad=10)
plt.xlabel('Local Social Support ($\phi$)', fontsize=11)
plt.ylabel('Resulting Psychological Cost ($C$)', fontsize=11)

plt.xlim(0, 1)
plt.ylim(-0.05, 1.05)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='gainsboro', fontsize=11)

plt.tight_layout()

plt.savefig('Psychological_Cost_vs_Support_Beta.png', dpi=300)
plt.show()
