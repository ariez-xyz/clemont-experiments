#!/usr/bin/env bash
trap 'exit' SIGINT
set -e

DEV=false
ASSUME_YES=false
SHOW_HELP=false

for arg in "$@"; do
  case "$arg" in
    --dev) DEV=true ;;
    --yes|-y) ASSUME_YES=true ;;
    --help|-h) SHOW_HELP=true ;;
  esac
done

print_plan() {
  cat <<EOF
This script will create a micromamba environment with all required 
dependencies to reproduce the experiments. It will:

  1) download micromamba into ./micromamba/
  2) create a file ./activate.fish (convenience for fish shell users.)
  3) install Python 3.11, pandas, matplotlib, xgboost, scikit-learn,
     torch, torchvision, torchaudio
  4) install clemont (see note)
  5) install the dd Python package with CUDD support.

Notes:
  - Supported platforms: macOS and Linux x86_64. On Windows, use WSL.
  - The --dev switch will clone clemont to lib/clemont and do an
    editable install for development purposes.

Current mode:
  --dev:     ${DEV}
  --yes:     ${ASSUME_YES}
  --help:    ${SHOW_HELP}

EOF
}

if [ "$SHOW_HELP" = true ]; then
  print_plan
  exit 0
fi

print_plan
if [ "$ASSUME_YES" != true ]; then
  read -r -p "Proceed with these actions? [y/N] " ans
  case "$ans" in
    y|Y|yes|YES) ;;
    *) echo "Aborted."; exit 0 ;;
  esac
else
  echo "Proceeding without confirmation (--yes)."
fi

check_dependencies() {
    local missing_tools=()
    if ! command -v wget &> /dev/null; then
        missing_tools+=("wget")
    fi
    if ! command -v tar &> /dev/null; then
        missing_tools+=("tar")
    fi
    if ! command -v git &> /dev/null; then
        missing_tools+=("git")
    fi
    if [ ${#missing_tools[@]} -ne 0 ]; then
        echo "Error: The following required tools are not installed:" >&2
        printf "  - %s\n" "${missing_tools[@]}" >&2
        echo "Please install the missing tools and try again." >&2
        exit 1
    fi
}

check_dependencies

# Detect architecture and OS (for micromamba)
PLATFORM=""
if [[ "$OSTYPE" == "darwin"* ]]; then
  if [[ $(uname -m) == "arm64" ]]; then
    PLATFORM="osx-arm64"
  else
    PLATFORM="osx-64"
  fi
elif [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "linux"* ]]; then
  PLATFORM="linux-64"
else
  echo "Unsupported OSTYPE: $OSTYPE"
  exit 1
fi

# Install micromamba if missing
if [ ! -d "./micromamba" ]; then
    mkdir -p ./micromamba
    wget -qO- "https://micro.mamba.pm/api/micromamba/${PLATFORM}/latest" | tar -xvj bin/micromamba
    mv bin micromamba
fi

# Initialize shell env
source activate.sh

# Prepare fish activation hook if needed
if [ ! -f "activate.fish" ]; then
    echo "set -gx MAMBA_ROOT_PREFIX ./micromamba/" > activate.fish
    ./micromamba/bin/micromamba shell hook -s fish >> activate.fish
    echo "micromamba activate" >> activate.fish
fi

# Create micromamba environment and install Python 3.11
micromamba install -y python=3.11

# Core Python deps
pip install pandas matplotlib xgboost scikit-learn
pip install torch torchvision torchaudio

mkdir -p lib
pushd lib

# clemont: PyPI by default, dev mode: clone and editable install
if [ "$DEV" = true ]; then
  git clone https://github.com/ariez-xyz/clemont
  pip install -e ./clemont
else
  pip install clemont==0.1.0
fi

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

set +v
popd >/dev/null

echo "setup.sh complete."
echo "To verify the install: source activate.sh && python experiments/test.py"

