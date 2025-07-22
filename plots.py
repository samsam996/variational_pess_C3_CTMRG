

import matplotlib.pyplot as plt
import numpy as np


e_monte_carlo = -0.544553
colors = ['b','r','g','m','c']
chi_list = [20]
index_max = [1,1,1,1,1]
D = 4
i = 0
energy_final = []

for k in range(len(chi_list)):

    chi = chi_list[k]
    col1, col2 = [], []
    model = "Heisenberg"
    # Read the log file
    with open(f'data/{model}_D{D}_chi{chi}_float64.log', 'r') as file:
        for line in file:
            # Split each line by whitespace and extract first two columns
            try:
                parts = line.split()
                col1.append(float(parts[0]))  # Convert to float if values are numeric
                col2.append(float(parts[1]))
            except (IndexError, ValueError):
                # Skip lines that don't have enough columns or contain non-numeric values
                continue

    iterations = col1
    energy = col2
    xx, cv  = [], []
    for j in range(len(iterations)-1):
        xx.append((iterations[j] + iterations[j+1])/2)
        cv.append((energy[j] - energy[j+1])/(iterations[j] - iterations[j+1]))


    energy_final.append(energy[index_max[k]])
  
  
    plt.plot(iterations, energy, marker='.', linestyle='', color=colors[k], label=f'chi={chi}')
    plt.xlabel('iteration')
    plt.ylabel('Energy')
    plt.title(f'D={D}')
    plt.grid(True)
    plt.savefig(f'figures_peps/D{D}chi{chi}.png', dpi=300) 
    plt.savefig(f'D{D}chi{chi}.pdf', dpi=300) 



plt.legend()
plt.show()


