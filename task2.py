#!/usr/bin/env python3
"""
Task 2: Influence Function-based Data Selection (Robust Production Implementation)

This script performs the following:
1. Loads the best model from Task 1.
2. Loads the Lipophilicity dataset (with train/validation/test split) and an external dataset.
3. Computes the average test gradient over the test set.
4. Uses a robust LiSSA approximation to compute the inverse Hessian–vector product (iHVP)
   with respect to the training loss.
5. For each external sample, computes its gradient and the influence score:
      influence(z) = - (grad(z))^T * iHVP(test_grad)
6. Selects the top-X% most influential external samples.
7. Combines the selected external samples with the original training set.
8. Fine-tunes the model on the combined dataset with early stopping (using a dedicated validation set).
9. Evaluates on the test set and saves all important artifacts.

Key parameters (adjustable via command-line arguments):
    - LiSSA recursion depth, damping, scaling factor.
    - Proportion of external samples to include (e.g., top 10%).
    - Fine-tuning epochs and learning rate.
    
References:
    - Koh & Liang (2017): https://arxiv.org/abs/1703.04730
    - LiSSA approximation: Agarwal et al. (2016) https://arxiv.org/abs/1602.03943
"""

import os
import argparse
import logging
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import wandb

from common import LipoDataset, MolFormerRegressor

# -----------------------------------------------------------------------------
# Reproducibility: Set random seeds and deterministic computation
# -----------------------------------------------------------------------------
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Default seed
SEED = 42
seed_all(SEED)

# -----------------------------------------------------------------------------
# Setup logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Argument Parsing for Hyperparameters and Configurations
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Task 2: Influence Function-based Data Selection")
    parser.add_argument("--lissa_recursion_depth", type=int, default=100, help="LiSSA recursion depth")
    parser.add_argument("--lissa_damping", type=float, default=0.01, help="LiSSA damping factor")
    parser.add_argument("--lissa_scale", type=float, default=25.0, help="LiSSA scaling factor")
    parser.add_argument("--external_top_percent", type=float, default=0.10, help="Top percent of external samples to select")
    parser.add_argument("--fine_tune_epochs", type=int, default=10, help="Number of fine-tuning epochs")
    parser.add_argument("--fine_tune_lr", type=float, default=2e-5, help="Fine-tuning learning rate")
    parser.add_argument("--batch_size_train", type=int, default=16, help="Training batch size")
    parser.add_argument("--batch_size_eval", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--max_length", type=int, default=128, help="Max tokenization length")
    parser.add_argument("--project", type=str, default="Task2_InfluenceFunctions", help="wandb project name")
    parser.add_argument("--model_revision", type=str, default="7b12d946c181a37f6012b9dc3b002275de070314",
                        help="Model repo revision to load (see finetune_lipophilicity.py for why this is pinned)")
    args = parser.parse_args()
    return args

# -----------------------------------------------------------------------------
# Dataset Classes
# -----------------------------------------------------------------------------
class ExternalDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128):
        df = pd.read_csv(csv_file)
        self.smiles_list = df['SMILES'].tolist()
        df.rename(columns={'Label': 'label'}, inplace=True)
        self.targets = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_df = df  # keep original DataFrame for later logging

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        target = self.targets[idx]
        encoding = self.tokenizer(
            smiles,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(target, dtype=torch.float)
        return item

# -----------------------------------------------------------------------------
# Gradient & Hessian-Vector Product (HVP) Computation
# -----------------------------------------------------------------------------
def compute_gradients(model, loss_fn, batch, device):
    """
    Compute gradients for a given batch.
    Returns a list of gradient tensors.
    """
    model.zero_grad()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].unsqueeze(1).to(device)
    outputs = model(input_ids, attention_mask)
    loss = loss_fn(outputs, labels)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
    return grads

def flatten_gradients(grad_list):
    """Flatten a list of gradients into a single vector."""
    return torch.cat([g.contiguous().view(-1) for g in grad_list])

def hvp(model, loss_fn, batch, vec, device):
    """
    Compute Hessian-vector product for a batch.
    vec: list of tensors (same shape as model.parameters())
    Returns a list of tensors representing the HVP.
    """
    model.zero_grad()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].unsqueeze(1).to(device)
    outputs = model(input_ids, attention_mask)
    loss = loss_fn(outputs, labels)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    dot = sum(torch.sum(g * v) for g, v in zip(grads, vec))
    hv = torch.autograd.grad(dot, model.parameters(), retain_graph=True)
    return hv

def lissa(model, train_loader, loss_fn, v, device, damping, scale, recursion_depth, tol):
    """
    LiSSA approximation for the inverse Hessian-vector product.
    v: list of tensors (test gradient vector)
    Returns approximation of H^{-1}v as a list of tensors.
    """
    h_est = [vi.clone().detach() for vi in v]
    for i in tqdm(range(recursion_depth), desc="LiSSA Recursion"):
        try:
            batch = next(train_loader)
        except StopIteration:
            train_loader = iter(train_loader)
            batch = next(train_loader)
        hv = hvp(model, loss_fn, batch, h_est, device)
        h_est = [vi + (1 - damping) * hi - scale * hv_i for vi, hi, hv_i in zip(v, h_est, hv)]
        if i % 10 == 0:
            norm = sum(torch.norm(hi).item() for hi in h_est)
            logger.info("LiSSA iteration %d: norm=%.6f", i, norm)
            wandb.log({"lissa_iteration": i, "lissa_norm": norm})
            if norm < tol:
                logger.info("Converged at iteration %d with norm %.6f", i, norm)
                break
    return h_est

def compute_average_gradient(model, data_loader, loss_fn, device):
    """
    Compute the average gradient over an entire dataset.
    """
    model.eval()
    total_grads = None
    count = 0
    for batch in tqdm(data_loader, desc="Computing Average Gradient"):
        grads = compute_gradients(model, loss_fn, batch, device)
        if total_grads is None:
            total_grads = [g.clone().detach() for g in grads]
        else:
            total_grads = [tg + g for tg, g in zip(total_grads, grads)]
        count += 1
    avg_grads = [tg / count for tg in total_grads]
    return avg_grads

# -----------------------------------------------------------------------------
# Main Function for Task 2
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    # Initialize wandb
    wandb.init(project=args.project, config={
        "lissa_recursion_depth": args.lissa_recursion_depth,
        "lissa_damping": args.lissa_damping,
        "lissa_scale": args.lissa_scale,
        "external_top_percent": args.external_top_percent,
        "fine_tune_epochs": args.fine_tune_epochs,
        "fine_tune_lr": args.fine_tune_lr,
        "batch_size_train": args.batch_size_train,
        "batch_size_eval": args.batch_size_eval,
        "max_length": args.max_length,
        "seed": SEED,
        "model": "ibm/MoLFormer-XL-both-10pct"
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Create directories for artifacts
    os.makedirs("logs", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

    # Load tokenizer and instantiate model
    model_name = "ibm/MoLFormer-XL-both-10pct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, revision=args.model_revision)
    base_model = AutoModel.from_pretrained(model_name, trust_remote_code=True, revision=args.model_revision)
    model = MolFormerRegressor(base_model)
    model.to(device)
    loss_fn = nn.MSELoss()

    # Load best model from Task1 if available; otherwise, use base model.
    best_model_path = "task1_best_model.pt"
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        logger.info("Loaded best Task1 model from %s", best_model_path)
    else:
        logger.warning("Task1 best model not found. Proceeding with base model.")

    # -----------------------------------------------------------------------------
    # Load Lipophilicity dataset and perform train/validation/test split
    # -----------------------------------------------------------------------------
    dataset_path = "scikit-fingerprints/MoleculeNet_Lipophilicity"
    data = load_dataset(dataset_path)
    df = pd.DataFrame(data['train'])
    df['bin'] = pd.qcut(df['label'], q=10, duplicates='drop')
    # First split: train_val and test
    train_val_df, test_df = train_test_split(df, test_size=0.2, stratify=df['bin'], random_state=SEED)
    # Second split: train and validation from train_val
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, stratify=train_val_df['bin'], random_state=SEED)
    for split_name, split_df in zip(["train", "val", "test"], [train_df, val_df, test_df]):
        logger.info("%s dataset size: %d", split_name, len(split_df))
        wandb.log({f"{split_name}_dataset_size": len(split_df)})
    # Drop bin column
    train_df = train_df.drop(columns=['bin'])
    val_df = val_df.drop(columns=['bin'])
    test_df = test_df.drop(columns=['bin'])
    train_dataset = LipoDataset(train_df['SMILES'].tolist(), train_df['label'].tolist(), tokenizer, max_length=args.max_length)
    val_dataset = LipoDataset(val_df['SMILES'].tolist(), val_df['label'].tolist(), tokenizer, max_length=args.max_length)
    test_dataset = LipoDataset(test_df['SMILES'].tolist(), test_df['label'].tolist(), tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_eval, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size_eval, shuffle=False, num_workers=2)

    # -----------------------------------------------------------------------------
    # Load external dataset for Task2
    # -----------------------------------------------------------------------------
    external_dataset_path = os.path.join("External-Dataset_for_Task2.csv")
    external_dataset = ExternalDataset(external_dataset_path, tokenizer, max_length=args.max_length)
    external_loader = DataLoader(external_dataset, batch_size=1, shuffle=False)

    # -----------------------------------------------------------------------------
    # Compute the average test gradient (v) over the test set
    # -----------------------------------------------------------------------------
    logger.info("Computing average test gradient over test set...")
    test_grad = compute_average_gradient(model, test_loader, loss_fn, device)

    # -----------------------------------------------------------------------------
    # Compute inverse Hessian-vector product (iHVP) using robust LiSSA
    # -----------------------------------------------------------------------------
    train_loader_cycle = iter(train_loader)
    logger.info("Approximating inverse Hessian-vector product using LiSSA...")
    iHVP = lissa(model, train_loader_cycle, loss_fn, test_grad, device,
                 damping=args.lissa_damping, scale=args.lissa_scale,
                 recursion_depth=args.lissa_recursion_depth, tol=1e-3)

    # -----------------------------------------------------------------------------
    # Compute influence scores for each external sample
    # -----------------------------------------------------------------------------
    influence_scores = []
    model.eval()
    for batch in tqdm(external_loader, desc="Computing Influence Scores for External Data"):
        grad_external = compute_gradients(model, loss_fn, batch, device)
        grad_ext_flat = flatten_gradients(grad_external)
        iHVP_flat = flatten_gradients(iHVP)
        influence = - torch.dot(grad_ext_flat, iHVP_flat).item()  # negative dot product as per theory
        influence_scores.append(influence)
    external_df = external_dataset.data_df.copy()
    external_df['influence'] = influence_scores
    influence_csv_path = os.path.join("logs", "external_influence_scores.csv")
    external_df.to_csv(influence_csv_path, index=False)
    # Log the external influence scores as a wandb artifact
    external_artifact = wandb.Artifact("external_influence_scores", type="dataset", description="Influence scores for external data")
    external_artifact.add_file(influence_csv_path)
    wandb.log_artifact(external_artifact)

    # -----------------------------------------------------------------------------
    # Select Top-k External Samples based on Influence Scores
    # -----------------------------------------------------------------------------
    top_percent = args.external_top_percent
    k = int(top_percent * len(external_df))
    top_external_df = external_df.sort_values(by="influence", ascending=False).head(k)
    logger.info("Selected top %d (%.1f%%) external samples.", k, top_percent * 100)
    wandb.log({"selected_external_samples": k})

    # -----------------------------------------------------------------------------
    # Combine external data with original training data
    # -----------------------------------------------------------------------------
    combined_df = pd.concat([train_df, top_external_df[['SMILES', 'label']]], ignore_index=True)
    combined_dataset = LipoDataset(combined_df['SMILES'].tolist(), combined_df['label'].tolist(), tokenizer, max_length=args.max_length)
    combined_loader = DataLoader(combined_dataset, batch_size=args.batch_size_train, shuffle=True, num_workers=2)
    logger.info("Combined training dataset size: %d", len(combined_dataset))
    wandb.log({"combined_dataset_size": len(combined_dataset)})

    # -----------------------------------------------------------------------------
    # Fine-tune the model on the combined dataset (with early stopping on validation set)
    # -----------------------------------------------------------------------------
    optimizer = optim.AdamW(model.parameters(), lr=args.fine_tune_lr)
    best_val_loss = float('inf')
    early_stop_patience = 3
    patience_counter = 0

    logger.info("Starting fine-tuning on combined dataset...")
    for epoch in range(args.fine_tune_epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(combined_loader, desc=f"Fine-tuning Epoch {epoch+1}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].unsqueeze(1).to(device)
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(combined_loader)
        logger.info("Epoch %d: Average Training Loss = %.4f", epoch+1, avg_loss)
        wandb.log({"epoch": epoch+1, "train_loss": avg_loss})

        # Validation for early stopping using the separate validation set
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].unsqueeze(1).to(device)
                preds = model(input_ids, attention_mask)
                loss = loss_fn(preds, labels)
                val_losses.append(loss.item())
        avg_val_loss = np.mean(val_losses)
        logger.info("Epoch %d: Average Validation Loss = %.4f", epoch+1, avg_val_loss)
        wandb.log({"epoch": epoch+1, "val_loss": avg_val_loss})

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_task2_path = os.path.join("saved_models", "best_model_task2.pt")
            torch.save(model.state_dict(), best_model_task2_path)
            logger.info("Saved new best model at epoch %d", epoch+1)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info("Early stopping triggered at epoch %d", epoch+1)
                break

    # -----------------------------------------------------------------------------
    # Evaluate the fine-tuned model on the test set (final evaluation)
    # -----------------------------------------------------------------------------
    model.eval()
    predictions = []
    true_values = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Fine-tuned Model"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            preds = model(input_ids, attention_mask)
            predictions.extend(preds.squeeze(1).cpu().tolist())
            true_values.extend(labels.tolist())
    mse = mean_squared_error(true_values, predictions)
    mae = mean_absolute_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)
    logger.info("Final Fine-tuned Model Performance -- Test MSE: %.4f, MAE: %.4f, R^2: %.4f", mse, mae, r2)
    wandb.log({"final_test_mse": mse, "final_test_mae": mae, "final_test_r2": r2})

    # Save final evaluation results
    eval_results = {
        "Test_MSE": mse,
        "Test_MAE": mae,
        "Test_R2": r2
    }
    eval_csv_path = os.path.join("logs", "evaluation_results_task2.csv")
    pd.DataFrame([eval_results]).to_csv(eval_csv_path, index=False)
    logger.info("Saved evaluation results to %s", eval_csv_path)
    
    # Log final evaluation results as an artifact
    eval_artifact = wandb.Artifact("evaluation_results_task2", type="dataset", description="Final evaluation results for Task 2")
    eval_artifact.add_file(eval_csv_path)
    wandb.log_artifact(eval_artifact)
    
    # Upload best model artifact to wandb
    model_artifact = wandb.Artifact("best_model_task2", type="model", description="Best fine-tuned model from Task2")
    model_artifact.add_file(best_model_task2_path)
    wandb.log_artifact(model_artifact)

    wandb.finish()

if __name__ == '__main__':
    main()
