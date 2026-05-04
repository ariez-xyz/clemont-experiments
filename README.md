# Clemont Experiments

Code to reproduce experiments for [Clemont](https://github.com/ariez-xyz/clemont).


## Structure

* `data/` contains various data sources used to create the input data for monitoring.
* `experiments/` holds a number of scripts, each of which corresponds to an experiment
* `results/` is the place for all experimental outputs
* `lib/` holds dd (and Clemont when running `setup.sh` with the `--dev` switch)


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

### Quantitative experiments

The quantitative monitor has its own runner script `experiments/quant_runner.py` that saves runs to `results/quantitative`, which contains scripts to generate various plots.

### Text LLM judge experiments

The text experiments are reproducible through fixed scripts. The preparation
scripts generate deterministic monitor-ready CSV/JSON pairs whose filenames
encode the judge model, embedding model, temperature, class count, and sample
size. The monitor scripts in `experiments/` hardcode those exact filenames.

```bash
# STEP 1: prepare OpenRouter judge outputs and embeddings
data/text/amazon/prepare.sh
data/text/toxic-chat/prepare.sh

# These generate the CSVs expected by the 4b text monitor experiments:
# data/text/amazon/amazon-judge-gemma-4-26b-a4b-it_embed-pplx-embed-v1-4b_temp-t0_5class_n2000.csv
# data/text/toxic-chat/toxic-chat-judge-gemma-4-26b-a4b-it_embed-pplx-embed-v1-4b_temp-t0_5class_n2000.csv

# STEP 2: run the monitor experiments
experiments/run_quant_text_amazon_5class_4b_probs.sh
experiments/run_quant_text_amazon_5class_4b_argmax.sh
experiments/run_quant_text_toxic_chat_5class_4b_probs.sh
experiments/run_quant_text_toxic_chat_5class_4b_argmax.sh

# STEP 3: browse results
./serve_viewers.sh
```

Each 4b monitor script writes a `quant_run_*.json` under
`results/quantitative/text_*/*` and then calls
`results/quantitative/report_text_monitor.py` to generate a human-readable PDF
report for the same run. The viewer landing page served by `serve_viewers.sh`
links directly to dataset viewers and monitor result viewers.

OpenRouter credentials are read from `.env` via `OPENROUTER_API_KEY`. The
OpenRouter client defaults are centralized in `data/text/openrouter_client.py`;
the current text experiments use temperature `0.0` for reproducible logprob
outputs.


## Experimental procedure

### 1. Input data

In order to reproduce the experiments, first the input data to Clemont must be created by setting up the data sources [Certifair](https://github.com/rcpsl/Certifair), [lcifr](https://github.com/eth-sri/lcifr), [RobustTrees](https://github.com/chenhongge/RobustTrees) and [RobustBench](https://github.com/RobustBench/robustbench). We provide scripts and individual documentation for each source in the `data/` directory.

### 2. Monitoring the data

Once the input data has been obtained, it is possible to run the scripts in the `experiments/` directory. Because of the requirements in terms of hardware (up to 96c, 512GB RAM, GPUs) and runtime (up to 48h), the scripts are designed for a Slurm cluster environment. They can be adapted to run locally, if so desired.

### 3. Inspecting results

The experiments scripts will save their results to the `results/` directory. This directory also contains additional documentation as well as scripts to process the results further, e.g. creating plots or human readable tabular data.
