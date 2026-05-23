#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reproducible setup script for Automatic Kernel Discovery.
#
# Creates a conda env named `bo` with the exact versions used for the paper,
# installs LassoBench (editable), and (optionally) MuJoCo for Humanoid.
#
# Usage:
#   bash setup.sh                # creates env + installs Python deps
#   WITH_MUJOCO=1 bash setup.sh  # also installs mujoco210 + mujoco-py (needs sudo)
# ---------------------------------------------------------------------------
set -euo pipefail

ENV_NAME=${ENV_NAME:-bo}
WITH_MUJOCO=${WITH_MUJOCO:-0}

# --- Conda init ---
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda not found. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# --- Env creation ---
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -n "$ENV_NAME" python=3.10 -y
fi
conda activate "$ENV_NAME"

# --- Python deps (order matters) ---
python -m pip install -U pip wheel
python -m pip install setuptools==80.9.0
python -m pip install numpy==1.26.4 scipy==1.12.0 scikit-learn==1.7.2
python -m pip install gym==0.26.2
python -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121
python -m pip install gpytorch==1.15.2 botorch==0.16.1
python -m pip install hyperopt==0.2.7 python-dotenv openai
python -m pip install matplotlib joblib

# LassoBench (vendored; needed for Lasso-DNA benchmark)
python -m pip install -e ./hdbo/benchsuite/LassoBench

# --- Optional MuJoCo (needs sudo for apt) ---
if [[ "$WITH_MUJOCO" == "1" ]]; then
    if [[ "$(id -u)" -ne 0 ]] && ! command -v sudo >/dev/null 2>&1; then
        echo "[WARN] no sudo; skipping apt step — install build deps manually."
    else
        SUDO=$([[ "$(id -u)" -ne 0 ]] && echo "sudo" || echo "")
        $SUDO apt-get update
        $SUDO apt-get install -y \
            build-essential gcc patchelf wget \
            libosmesa6-dev libgl1-mesa-dev libglew-dev libglfw3
    fi

    mkdir -p "$HOME/.mujoco"
    if [[ ! -d "$HOME/.mujoco/mujoco210" ]]; then
        (cd "$HOME/.mujoco" && \
         wget -q https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz && \
         tar -xzf mujoco210-linux-x86_64.tar.gz && \
         rm mujoco210-linux-x86_64.tar.gz)
    fi
    grep -qxF 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin' ~/.bashrc \
        || echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin' >> ~/.bashrc
    grep -qxF 'export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210' ~/.bashrc \
        || echo 'export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210' >> ~/.bashrc
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin"
    export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"

    python -m pip install "Cython<3" "setuptools<81"
    python -m pip install --no-build-isolation "mujoco-py==2.1.2.14"
    python -c "import mujoco_py; print('mujoco_py OK')"
fi

echo
echo "==============================================================="
echo "Setup complete.  Activate with:   conda activate $ENV_NAME"
echo "==============================================================="
