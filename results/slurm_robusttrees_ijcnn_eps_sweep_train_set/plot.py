import matplotlib.pyplot as plt
import json
import glob

# Load and parse data from files
eps_values = []
rob_tp = []
unrob_tp = []

# Process robust model files
for filename in sorted(glob.glob('results/rob-*.json')):
    with open(filename, 'r') as f:
        data = json.load(f)
        eps_values.append(data['eps'])
        rob_tp.append(data['n_true_positives'])

# Process non-robust model files
unrob_eps = []
for filename in sorted(glob.glob('results/unrob-*.json')):
    with open(filename, 'r') as f:
        data = json.load(f)
        unrob_eps.append(data['eps'])
        unrob_tp.append(data['n_true_positives'])

# Ensure we're using the same epsilon values
assert eps_values == unrob_eps, "Epsilon values don't match between robust and non-robust files"

# Create the plot
plt.figure(figsize=(10, 6), dpi=150)
plt.plot(eps_values, rob_tp, label='RobustTrees robustly trained model')
plt.plot(eps_values, unrob_tp, label='Standard model')

plt.yscale('log')
plt.xlabel('Neighborhood Size (ε)')
plt.ylabel('Unfair pairs')
plt.title('Number of unfair pairs in epsilon')
plt.legend()
# Logscale grid
plt.grid(True, which="both", ls="-")
plt.grid(True, which="minor", ls=":")

plt.savefig('plot.png')
plt.close()
