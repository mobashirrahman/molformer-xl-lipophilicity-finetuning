#!/usr/bin/env python3
"""
Title: Fine-Tune Chemical Language Model on Lipophilicity with Real-Time Monitoring
Description:
    This script fine-tunes a pre-trained chemical language model (MoLFormer-XL)
    on the Lipophilicity dataset with both supervised regression and unsupervised
    masked language modeling (MLM) fine-tuning. It integrates Weights & Biases
    (wandb) for real-time monitoring and enhanced logging of training progress,
    GPU memory usage, and benchmark timings. The best models are saved as checkpoints.
    
Usage:
    python finetune_lipophilicity.py [--epochs E] [--mlm_epochs M] [--train_batch_size B] [--lr LR]
        [--wandb_project PROJECT] [--wandb_api_key YOUR_API_KEY] [--output_dir OUTPUT_DIR]

"""
import os
import argparse
import logging
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModel, AutoModelForMaskedLM,
                          DataCollatorForLanguageModeling)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import wandb

from common import LipoDataset, MolFormerRegressor

# -----------------------------------------------------------------------------
# Set up logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Dataset classes
# -----------------------------------------------------------------------------
class SmilesDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, max_length=128):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        enc = self.tokenizer(
            smiles,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return enc['input_ids'].squeeze(0)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def load_and_split_data(dataset_path, test_size=0.2, random_state=42, num_bins=10):
    logger.info("Loading dataset from %s", dataset_path)
    data = load_dataset(dataset_path)
    df = pd.DataFrame(data['train'])
    logger.info("Dataset columns: %s", df.columns.tolist())
    df['bin'] = pd.qcut(df['label'], q=num_bins, duplicates='drop')
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['bin'], random_state=random_state)
    train_df = train_df.drop(columns=['bin'])
    test_df = test_df.drop(columns=['bin'])
    logger.info("Train size: %d, Test size: %d", len(train_df), len(test_df))
    return train_df, test_df

def evaluate_model(model, dataset, device, batch_size=32):
    model.eval()
    predictions = []
    true_values = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            preds = model(input_ids, attention_mask)
            predictions.extend(preds.squeeze(1).cpu().tolist())
            true_values.extend(labels.tolist())
    mse = mean_squared_error(true_values, predictions)
    mae = mean_absolute_error(true_values, predictions)
    r2  = r2_score(true_values, predictions)
    return mse, mae, r2

def supervised_train(model, train_loader, val_loader, device, loss_fn, optimizer,
                     epochs=5, accumulation_steps=2, scheduler=None, early_stopping_patience=2, output_dir="checkpoints"):
    best_val_loss = float('inf')
    patience_counter = 0
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_batches = len(train_loader)
        epoch_start = time.time()
        logger.info("Epoch %d/%d", epoch+1, epochs)
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)
        optimizer.zero_grad()
        for i, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].unsqueeze(1).to(device)
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss = loss / accumulation_steps
            loss.backward()
            running_loss += loss.item()

            if torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
                progress_bar.set_postfix(loss=loss.item()*accumulation_steps, gpu_mem=f"{mem_alloc:.1f} MB")

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        epoch_duration = time.time() - epoch_start
        avg_train_loss = running_loss * accumulation_steps / n_batches
        logger.info("Epoch %d completed in %.2fs, Avg Training Loss: %.4f", epoch+1, epoch_duration, avg_train_loss)
        wandb.log({"epoch": epoch+1, "avg_train_loss": avg_train_loss, "epoch_time_sec": epoch_duration})

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].unsqueeze(1).to(device)
                preds = model(input_ids, attention_mask)
                val_loss = loss_fn(preds, labels)
                val_losses.append(val_loss.item())
        avg_val_loss = sum(val_losses) / len(val_losses)
        logger.info("Epoch %d Validation Loss: %.4f", epoch+1, avg_val_loss)
        wandb.log({"epoch": epoch+1, "avg_val_loss": avg_val_loss})

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            checkpoint_path = os.path.join(output_dir, f"best_model_epoch_{epoch+1}.pt")
            torch.save(best_model_state, checkpoint_path)
            logger.info("Saved best model checkpoint to %s", checkpoint_path)
            wandb.save(checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info("Early stopping triggered at epoch %d", epoch+1)
                break

    if 'best_model_state' in locals():
        model.load_state_dict(best_model_state)
    return model

def train_mlm(mlm_model, mlm_loader, device, lr=5e-5, mlm_epochs=1):
    logger.info("Starting MLM fine-tuning for %d epoch(s)...", mlm_epochs)
    mlm_model.train()
    optimizer = torch.optim.AdamW(mlm_model.parameters(), lr=lr)
    for epoch in range(mlm_epochs):
        epoch_start = time.time()
        progress_bar = tqdm(mlm_loader, desc=f"MLM Epoch {epoch+1}", leave=False)
        epoch_loss = 0.0
        n_batches = len(mlm_loader)
        for batch in progress_bar:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = mlm_model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        epoch_duration = time.time() - epoch_start
        avg_loss = epoch_loss / n_batches
        logger.info("MLM Epoch %d completed in %.2fs, Avg Loss: %.4f", epoch+1, epoch_duration, avg_loss)
        wandb.log({"mlm_epoch": epoch+1, "mlm_avg_loss": avg_loss, "mlm_epoch_time_sec": epoch_duration})
    return mlm_model

def main(args):
    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb.login()
        run = wandb.init(project=args.wandb_project, config=vars(args))
    else:
        os.environ["WANDB_MODE"] = "offline"
        run = wandb.init(project=args.wandb_project, config=vars(args), anonymous="allow")
    wandb.run.name = args.run_name if args.run_name else f"finetune_{int(time.time())}"

    if args.hf_token:
        os.environ['HF_TOKEN'] = args.hf_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    wandb.log({"device": str(device)})

    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
        logger.info("Initial GPU memory allocated: %.1f MB", mem_alloc)
        wandb.log({"initial_gpu_mem_MB": mem_alloc})

    os.makedirs(args.output_dir, exist_ok=True)

    train_df, test_df = load_and_split_data(args.dataset_path,
                                              test_size=args.test_size,
                                              random_state=args.random_state,
                                              num_bins=args.num_bins)

    logger.info("Loading tokenizer from %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, revision=args.model_revision)
    
    sample_smiles = train_df.iloc[0]['SMILES']
    tokens = tokenizer.tokenize(sample_smiles)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    logger.info("Sample SMILES: %s", sample_smiles)
    logger.info("Tokens: %s", tokens)
    logger.info("Token IDs: %s", ids)
    
    train_dataset = LipoDataset(train_df['SMILES'].tolist(),
                                train_df['label'].tolist(),
                                tokenizer,
                                max_length=args.max_length)
    test_dataset = LipoDataset(test_df['SMILES'].tolist(),
                               test_df['label'].tolist(),
                               tokenizer,
                               max_length=args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)

    logger.info("Loading base model from %s", args.model_name)
    base_model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True, revision=args.model_revision)
    model = MolFormerRegressor(base_model, dropout_rate=args.dropout)
    model.to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    logger.info("Starting supervised regression fine-tuning...")
    model = supervised_train(model, train_loader, val_loader, device, loss_fn,
                             optimizer, epochs=args.epochs,
                             accumulation_steps=args.accumulation_steps,
                             scheduler=scheduler,
                             early_stopping_patience=args.patience,
                             output_dir=args.output_dir)
    mse, mae, r2 = evaluate_model(model, test_dataset, device, batch_size=args.eval_batch_size)
    logger.info("After supervised fine-tuning: Test MSE: %.4f, MAE: %.4f, R^2: %.4f", mse, mae, r2)
    wandb.log({"supervised_test_mse": mse, "supervised_test_mae": mae, "supervised_test_r2": r2})

    logger.info("Starting unsupervised MLM fine-tuning...")
    mlm_model = AutoModelForMaskedLM.from_pretrained(args.model_name, trust_remote_code=True, revision=args.model_revision)
    mlm_model.to(device)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    unlabeled_smiles = train_df['SMILES'].tolist()
    unlabeled_dataset = SmilesDataset(unlabeled_smiles, tokenizer, max_length=args.max_length)
    mlm_loader = DataLoader(unlabeled_dataset, batch_size=args.mlm_batch_size, shuffle=True, collate_fn=data_collator)
    mlm_model = train_mlm(mlm_model, mlm_loader, device, lr=args.mlm_lr, mlm_epochs=args.mlm_epochs)


    # Save the MLM model checkpoint after MLM training is complete
    mlm_checkpoint_path = os.path.join(args.output_dir, "mlm_model_checkpoint.pt")
    torch.save(mlm_model.state_dict(), mlm_checkpoint_path)
    logger.info("Saved MLM model checkpoint to %s", mlm_checkpoint_path)
    
    logger.info("Re-initializing regression model with MLM-fine-tuned base...")
    model.base_model = mlm_model.base_model
    model.regressor = nn.Linear(model.base_model.config.hidden_size, 1)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    logger.info("Starting second round of regression fine-tuning...")
    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.time()
        progress_bar = tqdm(train_loader, desc=f"Re-training Epoch {epoch+1}", leave=False)
        optimizer.zero_grad()
        for i, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].unsqueeze(1).to(device)
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss = loss / args.accumulation_steps
            loss.backward()
            if (i + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            if torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
                progress_bar.set_postfix(loss=loss.item()*args.accumulation_steps, gpu_mem=f"{mem_alloc:.1f} MB")
        epoch_duration = time.time() - epoch_start
        logger.info("Re-training Epoch %d completed in %.2fs", epoch+1, epoch_duration)
        wandb.log({"retrain_epoch": epoch+1, "retrain_loss": loss.item()*args.accumulation_steps, "retrain_epoch_time_sec": epoch_duration})

    mse, mae, r2 = evaluate_model(model, test_dataset, device, batch_size=args.eval_batch_size)
    logger.info("After MLM fine-tuning and re-training: Test MSE: %.4f, MAE: %.4f, R^2: %.4f", mse, mae, r2)
    wandb.log({"final_test_mse": mse, "final_test_mae": mae, "final_test_r2": r2})
    logger.info("Training completed.")
    wandb.finish()

def sweep_train():
    wandb.init()
    args = parse_args()
    config = wandb.config

    if hasattr(config, 'lr'):
        args.lr = config.lr
    if hasattr(config, 'train_batch_size'):
        args.train_batch_size = config.train_batch_size
    if hasattr(config, 'accumulation_steps'):
        args.accumulation_steps = config.accumulation_steps
    if hasattr(config, 'epochs'):
        args.epochs = config.epochs
    if hasattr(config, 'mlm_epochs'):
        args.mlm_epochs = config.mlm_epochs
    if hasattr(config, 'weight_decay'):
        args.weight_decay = config.weight_decay
    if hasattr(config, 'dropout'):
        args.dropout = config.dropout

    main(args)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune MoLFormer-XL on Lipophilicity Dataset with Monitoring and Checkpointing")
    parser.add_argument("--dataset_path", type=str, default="scikit-fingerprints/MoleculeNet_Lipophilicity",
                        help="Path or name of the dataset on Hugging Face")
    parser.add_argument("--model_name", type=str, default="ibm/MoLFormer-XL-both-10pct",
                        help="Pre-trained model name")
    parser.add_argument("--model_revision", type=str, default="7b12d946c181a37f6012b9dc3b002275de070314",
                        help="Model repo revision to load. Pinned to the pre-refactor commit: "
                             "the model's trust_remote_code modeling file has since been rewritten "
                             "upstream against a newer, not-yet-released transformers API and no "
                             "longer imports cleanly on any released transformers version.")
    parser.add_argument("--hf_token", type=str, default="",
                        help="Hugging Face token (if needed for authentication)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of supervised training epochs")
    parser.add_argument("--mlm_epochs", type=int, default=1, help="Number of MLM fine-tuning epochs")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size for regression")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--mlm_batch_size", type=int, default=32, help="Batch size for MLM training")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for regression training")
    parser.add_argument("--mlm_lr", type=float, default=5e-5, help="Learning rate for MLM fine-tuning")
    parser.add_argument("--accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum tokenization sequence length")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of dataset to use as test set")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for train-test split")
    parser.add_argument("--num_bins", type=int, default=10, help="Number of bins for stratification in train-test split")
    parser.add_argument("--wandb_project", type=str, default="Lipophilicity-FineTuning", help="wandb project name")
    parser.add_argument("--run_name", type=str, default="", help="Optional run name for wandb")
    parser.add_argument("--wandb_api_key", type=str, default="", help="wandb API key")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for optimizer")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for the model")
    return parser.parse_args()

if __name__ == '__main__':
    if os.environ.get("WANDB_SWEEP") == "true":
        sweep_train()
    else:
        args = parse_args()
        main(args)
