# Certifair

[Certifair](https://github.com/rcpsl/Certifair) predictions are provided pre-computed in `predictions.zip`

This is because the provided setup procedure does not work when using mamba instead of conda. The Microsoft tempeh package, required by a dependency of Certifair, seems to be available in conda-forge only. This is not an option due to licensing.

To reproduce the provided predictions:

1. clone the Certifair [repository](https://github.com/rcpsl/Certifair) repository 
2. follow the installation instructions
3. copy `certifair.py` and `makepreds.sh` into the cloned repository, overwriting the originals
4. activate the environment created during installation, and run `./makepreds.sh`.

