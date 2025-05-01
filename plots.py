

import matplotlib.pyplot as plt
import numpy as np


# Lists to store first and second column values

e_monte_carlo = -0.544553
colors = ['b','r','g','m','c']
chi_ = [10,20,30,40,50]
i = 0

for chi in chi_:
    # chi = 30
    D = 3
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

    # Plotting the data
    plt.plot(col1, (col2), marker='.', linestyle='', color=colors[i], label=f'chi={chi}')
    plt.xlabel('iteration')
    plt.ylabel('Energy')
    plt.title(f'D={D}')
    plt.grid(True)
    plt.savefig(f'figures/D{D}chi{chi}.png', dpi=300) 
    # plt.savefig(f'D{D}chi{chi}.pdf', dpi=300) 
    print(i)
    i += 1

plt.legend()
plt.show()
