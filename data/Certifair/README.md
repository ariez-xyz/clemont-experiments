# Certifair

[Certifair](https://github.com/rcpsl/Certifair) predictions are provided pre-computed in `predictions.zip`

This is because the setup procedure provided by the Certifair authors does not work when using mamba instead of conda. The Microsoft tempeh package, required by a dependency of Certifair, seems to be unavailable outside Conda default channels. This is not an option due to licensing.

To reproduce the precomputed predictions:

1. clone the Certifair [repository](https://github.com/rcpsl/Certifair)
2. follow the installation instructions (requires Anaconda)
3. copy `certifair.py` and `makepreds.sh` into the cloned repository, overwriting the originals. (The file `certifair.py` was adapted to save the predictions in CSV format, and `makepreds.sh` proceeds in the same way as [the author's provided evaluation script](https://github.com/rcpsl/Certifair/blob/main/scripts/tab2.sh) but handles naming of the resulting CSV files.)
4. activate the environment created during installation, and run `./makepreds.sh`.
5. use the combine.sh script to create combined predictions files for test+train sets.
