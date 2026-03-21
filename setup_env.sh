#!/bin/bash
# Python dev environment setup for RLix (assumes CUDA drivers already installed).
# Creates conda env "rlix" with Python 3.10 + CUDA toolkit 12.4,
# clones ROLL from GitHub, then installs all Python dependencies.
# Use -eo but not -u because conda's shell init references unset variables like PS1
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_ENV_NAME="rlix"
PYTHON_VERSION="3.10"
ROLL_REPO="https://github.com/rlops/ROLL.git"
ROLL_BRANCH="rlix"
ROLL_DIR="${SCRIPT_DIR}/external/ROLL"
CUDA_CHANNEL_LABEL="cuda-12.4.1"
CUDA_NVCC_VERSION="12.4.131"
CUDA_CUDART_DEV_VERSION="12.4.127"
CUDA_NVRTC_DEV_VERSION="12.4.127"
CUDA_CUBLAS_DEV_VERSION="12.4.5.8"
CUDA_CUSPARSE_DEV_VERSION="12.3.1.170"
CUDA_CUSOLVER_DEV_VERSION="11.6.1.9"
CUDA_NVTX_VERSION="12.4.127"
CUDNN_VERSION="9.1.1.17"

print_usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  -h, --help   Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option '$1'" >&2
      print_usage >&2
      exit 1
      ;;
  esac
  shift
done

# --- GPU check: abort early if no working NVIDIA GPU is detected ---
if ! nvidia-smi &> /dev/null; then
  echo "ERROR: 'nvidia-smi' failed. No working NVIDIA GPU detected or driver is misconfigured." >&2
  echo "This setup requires a machine with functional NVIDIA GPUs and matching drivers." >&2
  exit 1
fi
echo "GPU check passed:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# --- Install Miniconda if conda is not available ---
if ! command -v conda &> /dev/null; then
  if [[ -d "$HOME/miniconda3" ]]; then
    # Miniconda dir exists but conda not on PATH, just init it
    echo "Found existing Miniconda install, initializing..."
  else
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
  fi
  # Initialize conda for future bash sessions
  "$HOME/miniconda3/bin/conda" init bash
fi
# Source conda into current shell
# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# Accept conda default channel TOS (required for non-interactive use)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# --- Create conda env with Python 3.10 and the minimal CUDA stack needed by Transformer Engine ---
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
  echo "Conda env '${CONDA_ENV_NAME}' already exists, ensuring the required CUDA build packages are installed."
  conda install -n "${CONDA_ENV_NAME}" \
    "cuda-nvcc=${CUDA_NVCC_VERSION}" \
    "cuda-cudart-dev=${CUDA_CUDART_DEV_VERSION}" \
    "cuda-nvrtc-dev=${CUDA_NVRTC_DEV_VERSION}" \
    "libcublas-dev=${CUDA_CUBLAS_DEV_VERSION}" \
    "libcusparse-dev=${CUDA_CUSPARSE_DEV_VERSION}" \
    "libcusolver-dev=${CUDA_CUSOLVER_DEV_VERSION}" \
    "cuda-nvtx=${CUDA_NVTX_VERSION}" \
    "cudnn=${CUDNN_VERSION}" \
    -c "nvidia/label/${CUDA_CHANNEL_LABEL}" -c defaults \
    --strict-channel-priority -y
else
  echo "Creating conda env '${CONDA_ENV_NAME}' with Python ${PYTHON_VERSION}, CUDA nvcc ${CUDA_NVCC_VERSION}, CUDA runtime headers ${CUDA_CUDART_DEV_VERSION}, NVRTC headers ${CUDA_NVRTC_DEV_VERSION}, cuBLAS ${CUDA_CUBLAS_DEV_VERSION}, cuSPARSE ${CUDA_CUSPARSE_DEV_VERSION}, cuSOLVER ${CUDA_CUSOLVER_DEV_VERSION}, NVTX ${CUDA_NVTX_VERSION}, and cuDNN ${CUDNN_VERSION}..."
  # Transformer Engine needs CUDA headers, nvcc, NVRTC, cuBLAS, cuSPARSE,
  # cuSOLVER, NVTX, and cuDNN, but not the full toolkit metapackage. Keeping
  # this solve minimal avoids the earlier clobber conflicts from the broader
  # CUDA transaction.
  conda create -n "${CONDA_ENV_NAME}" python="${PYTHON_VERSION}" \
    "cuda-nvcc=${CUDA_NVCC_VERSION}" \
    "cuda-cudart-dev=${CUDA_CUDART_DEV_VERSION}" \
    "cuda-nvrtc-dev=${CUDA_NVRTC_DEV_VERSION}" \
    "libcublas-dev=${CUDA_CUBLAS_DEV_VERSION}" \
    "libcusparse-dev=${CUDA_CUSPARSE_DEV_VERSION}" \
    "libcusolver-dev=${CUDA_CUSOLVER_DEV_VERSION}" \
    "cuda-nvtx=${CUDA_NVTX_VERSION}" \
    "cudnn=${CUDNN_VERSION}" \
    -c "nvidia/label/${CUDA_CHANNEL_LABEL}" -c defaults \
    --strict-channel-priority -y
fi

# Activate the conda env
conda activate "${CONDA_ENV_NAME}"

# Set CUDA_HOME so build tools (e.g. transformer-engine, apex) find the toolkit
# Derive the toolkit root from the active nvcc path because CONDA_PREFIX is not
# reliable in this non-interactive script path.
CUDA_HOME_DIR="$(dirname "$(dirname "$(command -v nvcc)")")"
# Export it in the current shell and persist it for future activations.
export CUDA_HOME="$CUDA_HOME_DIR"
conda env config vars set CUDA_HOME="$CUDA_HOME_DIR"
# Re-activate to pick up persisted env vars in this shell too.
conda deactivate
conda activate "${CONDA_ENV_NAME}"
export CUDA_HOME="$CUDA_HOME_DIR"

echo "Active env: $(conda info --envs | grep '*')"
echo "Python: $(python --version)"
echo "nvcc: $(nvcc --version | tail -1)"
echo "CUDA_HOME: ${CUDA_HOME}"

# --- Install uv if not available ---
if ! command -v uv &> /dev/null; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# --- Clone ROLL if not already present ---
if [[ -d "${ROLL_DIR}" ]]; then
  echo "ROLL already cloned at ${ROLL_DIR}, pulling latest..."
  git -C "${ROLL_DIR}" pull
else
  echo "Cloning ROLL (branch ${ROLL_BRANCH}) into ${ROLL_DIR}..."
  mkdir -p "$(dirname "${ROLL_DIR}")"
  git clone --branch "${ROLL_BRANCH}" "${ROLL_REPO}" "${ROLL_DIR}"
fi

# --- Install Python dependencies ---
cd "${ROLL_DIR}"

uv pip install -r requirements_torch260_vllm.txt
if python -c "import transformer_engine" >/dev/null 2>&1; then
  echo "transformer-engine is already installed, skipping reinstall."
else
  uv pip install --no-build-isolation "transformer-engine[pytorch]==2.2.0"
fi

# Install ROLL
uv pip install "${ROLL_DIR}"

# --- System packages for tracing ---
# Must run before 'rlix' install: rlix depends on tg4perfetto which
# requires protoc (protobuf-compiler) at build time.
# Wait for any running apt/dpkg locks (e.g. unattended-upgrades) before installing
while fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do echo "Waiting for dpkg lock..."; sleep 5; done
apt-get update && apt-get install -y protobuf-compiler libprotobuf-dev nvtop

# Use the pure-Python protobuf backend so tg4perfetto's generated stubs work
# with any protobuf version. This avoids the <3.21.0 pin and conflicts with
# wandb/other packages that require a newer protobuf.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
conda env config vars set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
conda deactivate
conda activate "${CONDA_ENV_NAME}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Install tg4perfetto with --no-deps to skip its protobuf<3.21.0 constraint;
# compatibility is handled via the pure-Python backend above.
uv pip install --no-deps "tg4perfetto>=0.0.6"

# Install RLix
uv pip install "${SCRIPT_DIR}"

echo "Done! Run 'conda activate ${CONDA_ENV_NAME}' to use the environment."