#!/bin/bash -l
#SBATCH --job-name=ctf-smoke
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="${SCRATCH:-/scratch/$USER}/symbolic-ctf-agents/cache"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export UV_PYTHON_INSTALL_DIR="${SCRATCH:-/scratch/$USER}/symbolic-ctf-agents/python"
export PATH="$HOME/.local/bin:$PATH"

if ! command -v git >/dev/null 2>&1; then
    module load git
fi
git --version

if ! command -v uv >/dev/null 2>&1; then
    curl -fsSL https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$PWD/.local/bin" sh
    export PATH="$PWD/.local/bin:$PATH"
fi
uv sync --locked --no-dev

if [[ $# -gt 0 ]]; then
    device=$1
    episodes=$2
    checkpoint=$3
    run=$4
    seed=20261104
    training=(--spec "$run/capture.json" --output "$run/training"
        --opponent chase --device "$device" --episodes "$episodes" --no-tensorboard-server)
    if [[ -n "$checkpoint" ]]; then
        seed=$((SLURM_JOB_ID % 4294967296))
        training+=(--resume "$checkpoint")
    fi
    if [[ "$device" == cuda ]]; then
        nvidia-smi -L
    fi
    mkdir -p "$run"
    .venv/bin/mctf translate strategies/capture.txt --output "$run/capture.json"
    .venv/bin/mctf train "${training[@]}" --seed "$seed"
    cat "$run/training/training_complete.json"
    .venv/bin/mctf evaluate "$run/training/policy.zip" \
        --episodes 5 --output "$run/evaluation.csv"
    exit 0
fi

run="runs/smoke-$SLURM_JOB_ID"
mkdir -p "$run"
.venv/bin/mctf translate strategies/capture.txt --output "$run/capture.json"
.venv/bin/mctf train-team --spec "$run/capture.json" --output "$run/training" \
    --team-size 3 --opponent chase --episodes 10 --seed 20261104 \
    --no-reward-shaping --no-temporal-state --no-tensorboard-server
.venv/bin/mctf evaluate-team "$run/training/policy.zip" \
    --episodes 1 --output "$run/evaluation.csv"
cat "$run/training/training_budget.json"
