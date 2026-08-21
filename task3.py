#!/usr/bin/env python3
"""
Task 3: Exploration of Data Selection and Fine-Tuning Methods

This script performs the following:
1. Data selection on an external dataset using multiple strategies:
   - Target Distribution Alignment
   - Random Sampling
   - Clustering-based Selection
2. Combines the selected external data with the default Lipophilicity dataset.
3. Applies three parameter-efficient fine-tuning strategies:
   - BitFit: Only updates bias parameters.
   - LoRA: Injects low-rank adapters into selected linear layers.
   - iA3: Adds learned scaling vectors to selected modules.
4. Trains and evaluates the model on the combined dataset.
5. Logs performance metrics, training time, and trainable parameter counts.
6. Saves the best-performing model checkpoints and uploads them as WandB artifacts.

Author: Md Mobashir Rahman
License: MIT License
"""

import os
import time
import logging
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb

# For clustering-based selection
from sklearn.cluster import KMeans

from common import LipoDataset, MolFormerRegressor

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# -----------------------------------------------------------------------------
# Setup logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Embedding Computation (Batch-based)
# -----------------------------------------------------------------------------
def compute_embeddings_batch(texts, model, tokenizer, device, max_length=128, batch_size=16):
    """Compute embeddings for a list of texts using batch processing."""
    model.to(device)
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
            batch_texts = texts[i:i+batch_size]
            encoding = tokenizer(
                batch_texts,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                embeddings = outputs.last_hidden_state[:, 0]
            all_embeddings.append(embeddings.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)

# -----------------------------------------------------------------------------
# Data Selection Strategies
# -----------------------------------------------------------------------------
def select_data_subset(
    df, tokenizer, molformer, device, selection_percent=0.20, batch_size=16, target_avg=None
):
    """Select a subset of data closest in distribution to the target."""
    logger.info("Starting data selection using target distribution alignment...")
    if target_avg is None:
        target_sample = df.sample(n=min(100, len(df)), random_state=SEED)
        target_texts = target_sample['SMILES'].tolist()
        target_embeddings = compute_embeddings_batch(target_texts, molformer, tokenizer, device, batch_size=batch_size)
        target_avg = np.mean(target_embeddings, axis=0)
    candidate_texts = df['SMILES'].tolist()
    candidate_embeddings = compute_embeddings_batch(candidate_texts, molformer, tokenizer, device, batch_size=batch_size)
    distances = np.linalg.norm(candidate_embeddings - target_avg, axis=1)
    df = df.copy()
    df['distance'] = distances
    num_select = int(len(df) * selection_percent)
    selected_df = df.nsmallest(num_select, 'distance')
    logger.info("Selected %d samples out of %d (%.1f%%) using target alignment.",
                len(selected_df), len(df), selection_percent * 100)
    return selected_df

def select_data_subset_random(df, selection_percent=0.20):
    """Select a random subset of data."""
    logger.info("Selecting data using random sampling...")
    selected_df = df.sample(frac=selection_percent, random_state=SEED)
    logger.info("Selected %d samples out of %d (%.1f%%) using random sampling.",
                len(selected_df), len(df), selection_percent * 100)
    return selected_df

def select_data_subset_clustering(
    df, tokenizer, molformer, device, selection_percent=0.20, batch_size=16
):
    """Select a subset of data based on clustering in the embedding space."""
    logger.info("Starting data selection using clustering-based method...")
    texts = df['SMILES'].tolist()
    embeddings = compute_embeddings_batch(texts, molformer, tokenizer, device, batch_size=batch_size)
    n_select = int(len(df) * selection_percent)
    kmeans = KMeans(n_clusters=n_select, random_state=SEED)
    kmeans.fit(embeddings)
    selected_indices = []
    for cluster in range(kmeans.n_clusters):
        cluster_indices = np.where(kmeans.labels_ == cluster)[0]
        if len(cluster_indices) > 0:
            cluster_embeddings = embeddings[cluster_indices]
            center = kmeans.cluster_centers_[cluster]
            distances = np.linalg.norm(cluster_embeddings - center, axis=1)
            best_index = cluster_indices[np.argmin(distances)]
            selected_indices.append(best_index)
    selected_df = df.iloc[selected_indices]
    logger.info("Selected %d samples out of %d using clustering-based selection.", len(selected_df), len(df))
    return selected_df

# -----------------------------------------------------------------------------
# Fine-Tuning Strategies
# -----------------------------------------------------------------------------
def apply_bitfit(model):
    """Freeze all weights except biases."""
    logger.info("Applying BitFit: freezing all parameters except biases.")
    for name, param in model.named_parameters():
        if "bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model

class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=8):
        super(LoRALinear, self).__init__()
        self.original_linear = original_linear
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.r = r
        for param in self.original_linear.parameters():
            param.requires_grad = False
        self.A = nn.Parameter(torch.randn(self.r, self.in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(self.out_features, self.r) * 0.01)
    
    def forward(self, x):
        return self.original_linear(x) + (x @ self.A.t() @ self.B.t())

def apply_lora(model, target_module_names=["attention", "intermediate"], r=8):
    """Inject low-rank adapters into Linear layers."""
    logger.info("Applying LoRA: injecting low-rank adapters into selected layers.")
    replaced_modules = []
    
    def replace_module(module, prefix=""):
        for name, child in module.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and any(t in child_prefix.lower() for t in target_module_names):
                setattr(module, name, LoRALinear(child, r=r))
                replaced_modules.append(child_prefix)
                logger.info("Replaced %s with LoRALinear.", child_prefix)
            else:
                replace_module(child, child_prefix)
    
    replace_module(model.base_model)
    logger.info("LoRA: Total replaced modules: %d", len(replaced_modules))
    return model

class IA3Module(nn.Module):
    def __init__(self, original_module):
        super(IA3Module, self).__init__()
        self.original_module = original_module
        if hasattr(original_module, 'out_features'):
            self.scale = nn.Parameter(torch.ones(original_module.out_features))
        else:
            self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        out = self.original_module(x)
        return out * self.scale

def apply_ia3(model, target_module_names=["attention", "intermediate"]):
    """Inject learned scaling vectors into selected layers."""
    logger.info("Applying iA3: injecting learned scaling vectors into selected layers.")
    wrapped_modules = []
    
    def wrap_module(module, prefix=""):
        for name, child in module.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and any(t in child_prefix.lower() for t in target_module_names):
                setattr(module, name, IA3Module(child))
                wrapped_modules.append(child_prefix)
                logger.info("Wrapped %s with IA3 scaling.", child_prefix)
            else:
                wrap_module(child, child_prefix)
    
    wrap_module(model.base_model)
    logger.info("iA3: Total wrapped modules: %d", len(wrapped_modules))
    return model

# -----------------------------------------------------------------------------
# Utility: Count trainable parameters
# -----------------------------------------------------------------------------
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -----------------------------------------------------------------------------
# Training and Evaluation Functions
# -----------------------------------------------------------------------------
def train_model(model, train_loader, test_loader, device, epochs=5, lr=2e-5, patience=2):
    model.to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience, verbose=True)
    loss_fn = nn.MSELoss()
    best_val_loss = float('inf')
    best_model_state = None
    early_stopping_counter = 0
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].unsqueeze(1).to(device)
            preds = model(input_ids, attention_mask)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        logger.debug("Epoch %d: Training Loss = %.4f", epoch+1, avg_loss)
        
        model.eval()
        val_losses = []
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].unsqueeze(1).to(device)
                preds = model(input_ids, attention_mask)
                loss = loss_fn(preds, labels)
                val_losses.append(loss.item())
                all_preds.extend(preds.squeeze(1).cpu().numpy().tolist())
                all_labels.extend(labels.squeeze(1).cpu().numpy().tolist())
        avg_val_loss = np.mean(val_losses)
        logger.info("Epoch %d: Validation Loss = %.4f", epoch+1, avg_val_loss)
        wandb.log({"epoch": epoch+1, "train_loss": avg_loss, "val_loss": avg_val_loss})
        
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                logger.info("Early stopping triggered at epoch %d", epoch+1)
                break

    train_time = time.time() - start_time
    if best_model_state is None:
        best_model_state = model.state_dict()
    model.load_state_dict(best_model_state)
    model.eval()
    final_preds = []
    final_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].unsqueeze(1).to(device)
            preds = model(input_ids, attention_mask)
            final_preds.extend(preds.squeeze(1).cpu().numpy().tolist())
            final_labels.extend(labels.squeeze(1).cpu().numpy().tolist())
    mse = mean_squared_error(final_labels, final_preds)
    mae = mean_absolute_error(final_labels, final_preds)
    r2 = r2_score(final_labels, final_preds)
    return model, mse, mae, r2, train_time

# -----------------------------------------------------------------------------
# Main Experiment Pipeline
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Task 3: Data Selection & Fine-Tuning Exploration")
    parser.add_argument("--selection_percent", type=float, default=0.50, help="Fraction of external data to select")
    parser.add_argument("--data_selection_method", type=str, default="target_alignment",
                        choices=["target_alignment", "random", "clustering"],
                        help="Data selection method: target_alignment, random, clustering")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr_bitfit", type=float, default=5e-3, help="Learning rate for BitFit")
    parser.add_argument("--lr_lora", type=float, default=2e-5, help="Learning rate for LoRA")
    parser.add_argument("--lr_ia3", type=float, default=1e-4, help="Learning rate for iA3")
    parser.add_argument("--patience", type=int, default=2, help="Patience for early stopping")
    parser.add_argument("--wandb_project", type=str, default="nnti-project", help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default="mobashirrahman-saarland-university", help="WandB entity")
    parser.add_argument("--model_name", type=str, default="ibm/MoLFormer-XL-both-10pct", help="Pre-trained model name")
    parser.add_argument("--model_revision", type=str, default="7b12d946c181a37f6012b9dc3b002275de070314",
                        help="Model repo revision to load (see finetune_lipophilicity.py for why this is pinned)")
    parser.add_argument("--dataset_path", type=str, default="scikit-fingerprints/MoleculeNet_Lipophilicity", help="Default dataset path")
    parser.add_argument("--external_dataset_path", type=str, default="External-Dataset_for_Task2.csv", help="External dataset CSV path")
    parser.add_argument("--experiments", type=str, default="bitfit,lora,ia3",
                        help="Comma-separated list of experiments: bitfit, lora, ia3")
    args = parser.parse_args()
    
    experiments_to_run = [exp.strip().lower() for exp in args.experiments.split(",")]
    
    # Initialize WandB
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), mode="online")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, revision=args.model_revision)
    molformer = AutoModel.from_pretrained(args.model_name, trust_remote_code=True, revision=args.model_revision)
    custom_model = MolFormerRegressor(molformer)
    

    # Load Task 1 checkpoint (strict=False: PEFT adapters add params not present in the checkpoint)
    checkpoint = torch.load("task1_best_model.pt", map_location=device)
    custom_model.load_state_dict(checkpoint, strict=False)
    logger.info("Loaded checkpoint from task1_best_model.pt")
    
    # Load default dataset
    data = load_dataset(args.dataset_path)
    df = pd.DataFrame(data['train'])
    df['bin'] = pd.qcut(df['label'], q=10, duplicates='drop')
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['bin'], random_state=SEED)
    train_df = train_df.drop(columns=['bin'])
    test_df = test_df.drop(columns=['bin'])
    logger.info("Default training set size: %d", len(train_df))
    
    # Load external dataset
    external_df = pd.read_csv(args.external_dataset_path)
    if 'Label' in external_df.columns:
        external_df.rename(columns={'Label': 'label'}, inplace=True)
    logger.info("External dataset size: %d", len(external_df))
    
    # Data Selection
    if args.data_selection_method == "random":
        selected_external_df = select_data_subset_random(external_df, selection_percent=args.selection_percent)
    elif args.data_selection_method == "clustering":
        selected_external_df = select_data_subset_clustering(
            external_df, tokenizer, molformer, device,
            selection_percent=args.selection_percent, batch_size=args.batch_size
        )
    else:  # target_alignment
        target_sample = train_df.sample(n=min(100, len(train_df)), random_state=SEED)
        target_texts = target_sample['SMILES'].tolist()
        target_embeddings = compute_embeddings_batch(
            target_texts, molformer, tokenizer, device, batch_size=args.batch_size
        )
        target_avg = np.mean(target_embeddings, axis=0)
        selected_external_df = select_data_subset(
            external_df, tokenizer, molformer, device,
            selection_percent=args.selection_percent, batch_size=args.batch_size, target_avg=target_avg
        )
    logger.info("Selected external subset size: %d", len(selected_external_df))
    
    # Combine datasets
    combined_train_df = pd.concat([train_df, selected_external_df], ignore_index=True)
    logger.info("Combined training set size: %d", len(combined_train_df))
    
    # Create datasets and loaders
    combined_train_dataset = LipoDataset(
        combined_train_df['SMILES'].tolist(),
        combined_train_df['label'].tolist(),
        tokenizer
    )
    test_dataset = LipoDataset(
        test_df['SMILES'].tolist(),
        test_df['label'].tolist(),
        tokenizer
    )
    combined_train_loader = DataLoader(combined_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size*2, shuffle=False, num_workers=2)
    
    results = {}
    os.makedirs("saved_models", exist_ok=True)
    
    # Experiment 1: BitFit
    if "bitfit" in experiments_to_run:
        logger.info("Starting fine-tuning with BitFit...")
        model_bitfit = copy.deepcopy(custom_model)
        model_bitfit = apply_bitfit(model_bitfit)
        num_params_bitfit = count_trainable_parameters(model_bitfit)
        logger.info("BitFit: Trainable parameters: %d", num_params_bitfit)
        model_bitfit, mse_b, mae_b, r2_b, time_b = train_model(
            model_bitfit, combined_train_loader, test_loader, device,
            epochs=args.epochs, lr=args.lr_bitfit, patience=args.patience
        )
        results['BitFit'] = {
            'MSE': mse_b, 'MAE': mae_b, 'R2': r2_b,
            'TrainTime(s)': time_b, 'TrainableParams': num_params_bitfit
        }
        bitfit_path = os.path.join("saved_models", "model_bitfit.pt")
        torch.save(model_bitfit.state_dict(), bitfit_path)
        wandb.log({
            "BitFit_MSE": mse_b, "BitFit_MAE": mae_b, "BitFit_R2": r2_b,
            "BitFit_TrainTime": time_b, "BitFit_TrainableParams": num_params_bitfit
        })
    
    # Experiment 2: LoRA
    if "lora" in experiments_to_run:
        logger.info("Starting fine-tuning with LoRA...")
        model_lora = copy.deepcopy(custom_model)
        model_lora = apply_lora(model_lora, target_module_names=["attention", "intermediate"], r=8)
        num_params_lora = count_trainable_parameters(model_lora)
        logger.info("LoRA: Trainable parameters: %d", num_params_lora)
        model_lora, mse_l, mae_l, r2_l, time_l = train_model(
            model_lora, combined_train_loader, test_loader, device,
            epochs=args.epochs, lr=args.lr_lora, patience=args.patience
        )
        results['LoRA'] = {
            'MSE': mse_l, 'MAE': mae_l, 'R2': r2_l,
            'TrainTime(s)': time_l, 'TrainableParams': num_params_lora
        }
        lora_path = os.path.join("saved_models", "model_lora.pt")
        torch.save(model_lora.state_dict(), lora_path)
        wandb.log({
            "LoRA_MSE": mse_l, "LoRA_MAE": mae_l, "LoRA_R2": r2_l,
            "LoRA_TrainTime": time_l, "LoRA_TrainableParams": num_params_lora
        })
    
    # Experiment 3: iA3
    if "ia3" in experiments_to_run:
        logger.info("Starting fine-tuning with iA3...")
        model_ia3 = copy.deepcopy(custom_model)
        model_ia3 = apply_ia3(model_ia3, target_module_names=["attention", "intermediate"])
        num_params_ia3 = count_trainable_parameters(model_ia3)
        logger.info("iA3: Trainable parameters: %d", num_params_ia3)
        model_ia3, mse_i, mae_i, r2_i, time_i = train_model(
            model_ia3, combined_train_loader, test_loader, device,
            epochs=args.epochs, lr=args.lr_ia3, patience=args.patience
        )
        results['iA3'] = {
            'MSE': mse_i, 'MAE': mae_i, 'R2': r2_i,
            'TrainTime(s)': time_i, 'TrainableParams': num_params_ia3
        }
        ia3_path = os.path.join("saved_models", "model_ia3.pt")
        torch.save(model_ia3.state_dict(), ia3_path)
        wandb.log({
            "iA3_MSE": mse_i, "iA3_MAE": mae_i, "iA3_R2": r2_i,
            "iA3_TrainTime": time_i, "iA3_TrainableParams": num_params_ia3
        })
    
    # Log and save results
    logger.info("=== Experiment Results ===")
    for method, res in results.items():
        logger.info(
            "%s: MSE=%.4f, MAE=%.4f, R2=%.4f, TrainTime=%.2fs, TrainableParams=%d",
            method, res['MSE'], res['MAE'], res['R2'],
            res['TrainTime(s)'], res['TrainableParams']
        )
    
    results_df = pd.DataFrame(results).T
    os.makedirs("logs", exist_ok=True)
    results_csv_path = os.path.join("logs", "task3_results.csv")
    results_df.to_csv(results_csv_path)
    wandb.save(results_csv_path)
    logger.info("Saved results to %s", results_csv_path)
    wandb.finish()

if __name__ == '__main__':
    main()