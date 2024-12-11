#!/bin/bash
trap 'exit' SIGINT

if [ ! -d "./miniconda3" ]; then # Setup miniconda in cwd
    mkdir -p ./miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./miniconda3/miniconda.sh
    bash ./miniconda3/miniconda.sh -b -s -u -p ./miniconda3
    rm ./miniconda3/miniconda.sh
fi

source activate.sh

conda install -y python=3.11
conda install -y pytorch::faiss-cpu
pip install --upgrade pybind11
pip install --verbose 'nmslib @ git+https://github.com/nmslib/nmslib.git#egg=nmslib&subdirectory=python_bindings'
pip install pandas matplotlib

pushd lib/
###################################################
# From tulip-control/dd script install_dd_cudd.sh #
###################################################
#
# Install `dd`, including the modules
# `dd.cudd` and `dd.cudd_zdd`

set -v
set -e
pip install dd
    # to first install
    # dependencies of `dd`
pip uninstall -y dd
pip download \
    --no-deps dd \
    --no-binary dd
tar -xzf dd-*.tar.gz
pushd dd-*/
export DD_FETCH=1 DD_CUDD=1 DD_CUDD_ZDD=1
pip install . \
    -vvv \
    --use-pep517 \
    --no-build-isolation
# confirm that `dd.cudd` did get installed
pushd tests/
python -c 'import dd.cudd'
popd
popd
##################################################
# End tulip-control/dd script install_dd_cudd.sh #
##################################################
popd



