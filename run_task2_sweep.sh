#!/bin/bash
# -----------------------------------------------------------------------------
# run_task2.sh - HTCondor job for Task 2: Influence Function-based Data Selection
# -----------------------------------------------------------------------------

set -e  # Exit immediately if any command fails

# Create directories for logs and models
mkdir -p logs
mkdir -p saved_models

# Create the folder structure required by task2.py.
# task2.py expects the external dataset at "../tasks/External-Dataset_for_Task2.csv"
mkdir -p tasks
if [ -f External-Dataset_for_Task2.csv ]; then
    cp External-Dataset_for_Task2.csv tasks/
    echo "Copied External-Dataset_for_Task2.csv to tasks/"
else
    echo "Warning: External-Dataset_for_Task2.csv not found in the current directory."
fi

# Manually import environment (uncomment if needed)

# export WANDB_API_KEY="REDACTED_WANDB_API_KEY"
# export WANDB_ENTITY="mobashirrahman-saarland-university"
# export WANDB_PROJECT="Task2InfluenceFunctions"
# export WANDB_SWEEP=TRUE

echo "=== Starting Task 2: Influence Function-based Data Selection ==="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
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

echo "Installing Python dependencies from requirements.txt..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

echo "Installed package versions:"
python3 -m pip freeze

# Check if a sweep ID is provided.
if [ -z "$SWEEP_ID" ]; then
    echo "No SWEEP_ID provided. Running standard task2.py run."
    python3 task2.py 2>&1 | tee logs/task2_run.log
else
    echo "Running wandb sweep agent with SWEEP_ID: $SWEEP_ID"
    python3 -m wandb agent $SWEEP_ID 2>&1 | tee logs/task2_run.log
fi

echo "Task 2 completed at $(date)"
