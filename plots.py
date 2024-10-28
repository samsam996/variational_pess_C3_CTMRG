

import matplotlib.pyplot as plt
import numpy as np


# Lists to store first and second column values
col1, col2 = [], []
e_monte_carlo = -0.544553

chi = 15
D = 2
model = "maple_leaf"
# Read the log file
with open(f'data/{model}_D{D}_chi{chi}_float64.log', 'r') as file:
    for line in file:
        # Split each line by whitespace and extract first two columns
        try:
            parts = line.split()
            col1.append(float(parts[0]))  # Convert to float if values are numeric
            col2.append(float(parts[1])-e_monte_carlo)
        except (IndexError, ValueError):
            # Skip lines that don't have enough columns or contain non-numeric values
            continue

# Plotting the data
plt.plot(col1, np.log10(col2), marker='.', linestyle='-', color='b')
plt.xlabel('iteration')
plt.ylabel('log(E - E_(Monte Carlo))')
plt.title(f'Chi={chi}, D={D}')
plt.grid(True)
plt.savefig('D3chi20.png', dpi=300) 
plt.savefig('D3chi20.pdf', dpi=300) 

plt.show()
