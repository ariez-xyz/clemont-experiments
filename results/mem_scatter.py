import matplotlib.pyplot as plt
import sys

methods = {'bf': [], 'kdtree': [], 'snn': [], 'bdd': []}
epss = {'bf': [], 'kdtree': [], 'snn': [], 'bdd': []}

with open(sys.argv[1], 'r') as f:
    for line in f:
        if not line.strip():
            continue
        # Parse the tuple-like string
        parts = line.strip()[1:-1].split(',')
        method = parts[3].strip()[1:-1].split('@')[0]  # Extract bf/kdtree/snn
        threads = "1t"
        if "@" in line:
            threads = parts[3].strip()[1:-1].split('@')[1]  
        memory = float(parts[2])  # Second last number
        eps = float(parts[1])  # Second last number
        if "dl2" in line.lower() and "compas" in line.lower() and threads == "1t":
            methods[method].append(memory)
            epss[method].append(eps*100)
            print(line)

plt.figure(figsize=(10, 6))

y_positions = {'bf': 1, 'kdtree': 2, 'snn': 3, 'bdd': 4}

for method, values in methods.items():
    plt.scatter([v/1000 for v in values], [y_positions[method]] * len(values), 
                alpha=0.5, label=method, s=epss[method])

plt.yticks([1, 2, 3, 4], ['bf', 'kdtree', 'snn', 'bdd'])
plt.xlabel('Memory Usage (MB)')
plt.xlim(left=0)
plt.ylabel('Method')
plt.title('Memory Usage Distribution by Method')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
