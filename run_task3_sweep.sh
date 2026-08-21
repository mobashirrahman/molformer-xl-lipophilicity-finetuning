#!/bin/bash
# -----------------------------------------------------------------------------
# run_task3_sweep.sh - HTCondor job for running a WandB hyperparameter sweep for Task 3
# -----------------------------------------------------------------------------





# Check if a sweep ID was provided as the first argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <SWEEP_ID>"
    exit 1
fi
SWEEP_ID=$1

echo "=== Starting Hyperparameter Sweep Job with wandb agent ==="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "User: $(id -un 2>/dev/null || echo 'Unknown')"
echo "Process ID: $$"
echo "Allocated GPUs (nvidia-smi):"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not found."
fi
echo "Memory info (free -h):"
free -h

echo "Installed Python version:"
which python3 && python3 --version
echo "Installed pip version:"
which pip3 && pip3 --version

# -----------------------------------------------------------------------------
# Install dependencies (only once)
# -----------------------------------------------------------------------------
echo "Installing Python dependencies from requirements.txt..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
echo "Installed package versions:"
python3 -m pip freeze

# -----------------------------------------------------------------------------
# Set up Weights & Biases environment variables for automatic sweep tracking
# -----------------------------------------------------------------------------
# export WANDB_API_KEY="REDACTED_WANDB_API_KEY"
# export WANDB_ENTITY="mobashirrahman-saarland-university"
# export WANDB_PROJECT="Task3"
# export WANDB_SWEEP=FALSE


echo "WandB environment variables set:"
echo "  WANDB_API_KEY: ${WANDB_API_KEY}"
echo "  WANDB_ENTITY: ${WANDB_ENTITY}"
echo "  WANDB_PROJECT: ${WANDB_PROJECT}"
echo "  WANDB_SWEEP: ${WANDB_SWEEP}"

# -----------------------------------------------------------------------------
# Launch wandb agent with the given sweep id
# -----------------------------------------------------------------------------
echo "Starting wandb agent with sweep id: ${SWEEP_ID}"
python3 -m wandb agent ${WANDB_ENTITY}/${WANDB_PROJECT}/${SWEEP_ID}

# -----------------------------------------------------------------------------
# Logging sweep results
# -----------------------------------------------------------------------------
log_dir="sweep_logs"
mkdir -p "$log_dir"
echo "=== Hyperparameter Sweep Finished at $(date) ==="
echo "Logs saved in: $log_dir"
