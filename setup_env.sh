#!/usr/bin/env bash
set -euo pipefail

# Smart Traffic Light Optimization - environment setup
# Usage:
#   bash setup_env.sh
#   bash setup_env.sh my_env_name
# Optional:
#   PYTHON_VERSION=3.11 bash setup_env.sh trafficopt

ENV_NAME="${1:-trafficopt}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$PROJECT_ROOT"

echo "============================================================"
echo "Smart Traffic Light Optimization - Environment Setup"
echo "============================================================"
echo "Project root : $PROJECT_ROOT"
echo "Environment  : $ENV_NAME"
echo "Python       : $PYTHON_VERSION"
echo "============================================================"

# Create required output folders used by the CLI/dashboard.
mkdir -p outputs/charts
mkdir -p tests

# Write requirements.txt so a fresh clone has all required packages.
cat > requirements.txt <<'REQ'
# Core data stack
numpy>=1.24,<3
pandas>=2.0,<3

# Machine learning and optimization
scikit-learn>=1.3,<2
scipy>=1.10,<2

# Simulation utilities
networkx>=3.1,<4

# Visualization and dashboard
matplotlib>=3.7,<4
plotly>=5.18,<7
streamlit>=1.32,<2

# Testing
pytest>=8,<10
REQ

echo "requirements.txt created/updated."

install_with_conda() {
    echo "Conda detected. Setting up conda environment: $ENV_NAME"

    # Create the environment only if it does not already exist.
    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        echo "Conda environment '$ENV_NAME' already exists. Reusing it."
    else
        echo "Creating conda environment '$ENV_NAME'..."
        conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip
    fi

    echo "Installing Python packages..."
    conda run -n "$ENV_NAME" python -m pip install --upgrade pip setuptools wheel
    conda run -n "$ENV_NAME" python -m pip install -r requirements.txt

    echo "Validating imports..."
    conda run -n "$ENV_NAME" python - <<'PY'
import numpy
import pandas
import sklearn
import scipy
import matplotlib
import plotly
import streamlit
import networkx
print("All required packages imported successfully.")
PY

    echo ""
    echo "Setup complete. To use the environment, run:"
    echo "  conda activate $ENV_NAME"
    echo "  pytest"
    echo "  streamlit run streamlit_app.py"
}

install_with_venv() {
    echo "Conda not detected. Falling back to Python venv: .venv"

    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        echo "ERROR: Python was not found. Install Python $PYTHON_VERSION or Anaconda/Miniconda first."
        exit 1
    fi

    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment .venv..."
        "$PYTHON_BIN" -m venv .venv
    else
        echo "Virtual environment .venv already exists. Reusing it."
    fi

    # shellcheck disable=SC1091
    source .venv/bin/activate

    echo "Installing Python packages..."
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r requirements.txt

    echo "Validating imports..."
    python - <<'PY'
import numpy
import pandas
import sklearn
import scipy
import matplotlib
import plotly
import streamlit
import networkx
print("All required packages imported successfully.")
PY

    echo ""
    echo "Setup complete. To use the environment, run:"
    echo "  source .venv/bin/activate"
    echo "  pytest"
    echo "  streamlit run streamlit_app.py"
}

if command -v conda >/dev/null 2>&1; then
    install_with_conda
else
    install_with_venv
fi

echo "============================================================"
echo "Environment setup finished successfully."
echo "============================================================"
