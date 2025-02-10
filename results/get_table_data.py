import sys
from collections import defaultdict

"""
Point it at a performance.txt file holding info gathered by get_stats.py
"""

methods_mem = defaultdict(lambda: defaultdict(list))
methods_rt  = defaultdict(lambda: defaultdict(list))
experiments = set()

def fingerprint(experiment):
    if "-local-" in experiment or "-global-" in experiment or "-base-" in experiment:
        prefix = "Certifair"
        if "adult" in experiment:
            return prefix + " Adult (31 cols, 46033 rows, 2 classes, Linf norm)"
        elif "german" in experiment:
            return prefix + " German (18 cols, 1000 rows, 2 classes, Linf norm)"
        elif "compas" in experiment:
            return prefix + " COMPAS (15 cols, 5278 rows, 2 classes, Linf norm)"
    elif "_dl2-" in experiment:
        prefix = "Lcifr"
        if "adult" in experiment:
            return prefix + " Adult (104 cols, 45222 rows, 2 classes, Linf norm)"
        elif "german" in experiment:
            return prefix + " German (59 cols, 1000 rows, 2 classes, Linf norm)"
        elif "compas" in experiment:
            return prefix + " COMPAS (14 cols, 5278 rows, 2 classes, Linf norm)"
    else:
        prefix = ""
    if "cifar10-" in experiment:
        return prefix + " CIFAR-10 (3072 cols, 20k rows, 10 classes, Linf norm)"
    elif "cifar100-" in experiment:
        return prefix + " CIFAR-100 (3072 cols, 20k rows, 100 classes, Linf norm)"
    elif "cifar10c-" in experiment:
        return prefix + " CIFAR-10C (384 cols, 20k rows, 10 classes, L2 norm)"
    elif "cifar100c-" in experiment:
        return prefix + " CIFAR-100C (384 cols, 20k rows, 100 classes, L2 norm)"
    elif "imagenet3dcc-" in experiment:
        return prefix + " Imagenet-3DCC (384 cols, 10k rows, 1000 classes, L2 norm)"
    elif "imagenet-" in experiment:
        return prefix + " Imagenet (150528 cols, 10k rows, 1000 classes, Linf norm)"

with open(sys.argv[1], 'r') as f:
    for line in f:
        if not line.strip():
            continue
        # Parse the tuple-like string
        parts = line.strip()[1:-1].split(',')
        experiment = parts[0]
        method = parts[3].strip()[1:-1]
        threads = "1t"
        if "@" in line:
            threads = parts[3].strip()[1:-1].split('@')[1]  
        if "P1" in experiment or ("cifar" not in experiment and threads != "1t"):
            print(f"skipping {line}", file=sys.stderr, end="")
            continue
        memory = int(parts[-2])  
        eps = float(parts[1])  
        if eps >= 0.16: continue
        avgtime = float(parts[-4])  

        fp = fingerprint(experiment)
        methods_mem[method][fp].append(memory)
        methods_rt[method][fp].append(avgtime)
        experiments.add(fp)

for experiment in sorted(list(experiments)):
    print(experiment)
    for method in methods_mem.keys():
        mem = methods_mem[method][experiment]
        rt = methods_rt[method][experiment]
        if len(mem) == 0: continue
        print(f"\t{method.ljust(10)}\t{round(sum(mem)/len(mem)/1024)}MB\t{round(sum(rt)/len(rt)*1000, 4)}ms")


