# Clemont Experiments

Code to reproduce experiments for [Clemont](https://github.com/ariez-xyz/clemont).


## Structure

* `data/` contains various data sources used to create the input data for monitoring.
* `experiments/` holds a number of scripts, each of which corresponds to an experiment
* `results/` is the place for all experimental outputs
* `lib/` holds dd (and Clemont with the `--dev` switch)


## Usage

Use the `setup.sh` script to create a Mamba environemnt with the required dependencies. With the `--dev` switch, Clemont will be cloned to `lib/clemont` with an editable install for development purposes.

Once complete, we can conduct an experiment as follows.

```bash
# STEP 1: obtaining input data
cd data/RobustBench/
./setup.sh                          # Install RobustBench
./slurm_submit_cifar10_standard.sh  # Compute CIFAR10 predictions

# STEP 2: monitoring
cd ../../experiments
./example_experiment.sh             # Run experiment (script adapted from submit_adversarial_cifar10c_standard.sh for local execution)

# STEP 3: inspecting results
cd ../results
python get_stats.py adversarial/
```

The final command should print output close to this:

```
# Input file                 eps       avg time   backend  mem      #positives
('cifar10-Standard-adv.csv', 0.031373, 0.0037135, 'bf@1t', 2250484, 9477)
```


## Experimental procedure

### 1. Input data

In order to reproduce the experiments, first the input data to Clemont must be created by setting up the data sources [Certifair](https://github.com/rcpsl/Certifair), [lcifr](https://github.com/eth-sri/lcifr), [RobustTrees](https://github.com/chenhongge/RobustTrees) and [RobustBench](https://github.com/RobustBench/robustbench). We provide scripts and individual documentation for each source in the `data/` directory.

### 2. Monitoring the data

Once the input data has been obtained, it is possible to run the scripts in the `experiments/` directory. Because of the requirements in terms of hardware (up to 96c, 512GB RAM, GPUs) and runtime (up to 48h), the scripts are designed for a Slurm cluster environment. They can be adapted to run locally, if so desired.

### 3. Inspecting results

The experiments scripts will save their results to the `results/` directory. This directory also contains additional documentation as well as scripts to process the results further, e.g. creating plots or human readable tabular data.
