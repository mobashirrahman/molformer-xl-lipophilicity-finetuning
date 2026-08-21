"""Training loop and evaluation protocol.

Model selection and early stopping run against the validation split; the test
split is evaluated exactly once, after the best-validation weights have been
restored. The original code early-stopped on the test set and then reported
metrics on it, so its "test" numbers were optimistically biased by selection.

Mixed precision is available but defaults to OFF, because neither path works
for this model on this hardware:
  * fp16 autocast overflows the linear-attention block completely - every
    element of the backbone's hidden states comes back non-finite.
  * bf16 stays finite but is unusable here: outputs deviate from fp32 by ~0.79
    on a prediction range of ~0.8, and it runs ~25% *slower* because Turing
    (SM 7.5) has no native bf16 tensor cores.
Runs are therefore fp32. Batch size 64 is the throughput sweet spot.
"""
import copy
import logging
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


@torch.no_grad()
def evaluate(model, loader, device, amp=False):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            out = model(input_ids, attention_mask)
        preds.extend(out.float().squeeze(-1).cpu().tolist())
        trues.extend(batch["labels"].tolist())
    return {
        "mse": mean_squared_error(trues, preds),
        "mae": mean_absolute_error(trues, preds),
        "r2": r2_score(trues, preds),
    }


def train(
    model,
    train_loader,
    val_loader,
    device,
    epochs=10,
    lr=2e-5,
    weight_decay=0.0,
    patience=3,
    amp=False,
    max_grad_norm=1.0,
    log_every=None,
):
    """Train, early-stopping on validation loss. Returns (model, history)."""
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, patience - 1)
    )
    loss_fn = nn.MSELoss()
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    stale = 0
    history = []
    start = time.time()

    for epoch in range(epochs):
        model.train()
        running = 0.0
        n_batches = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = loss_fn(model(input_ids, attention_mask).squeeze(-1), labels)

            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at epoch {epoch}")

            scaler.scale(loss).backward()
            if max_grad_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            n_batches += 1

        train_loss = running / max(1, n_batches)
        val = evaluate(model, val_loader, device, amp=use_amp)
        scheduler.step(val["mse"])
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_mse": val["mse"],
                        "val_mae": val["mae"], "val_r2": val["r2"]})

        if log_every and (epoch + 1) % log_every == 0:
            logger.info("epoch %d/%d train=%.4f val_mse=%.4f val_r2=%.4f",
                        epoch + 1, epochs, train_loss, val["mse"], val["r2"])

        if val["mse"] < best_val - 1e-6:
            best_val = val["mse"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                logger.info("early stopping at epoch %d (best epoch %d)", epoch + 1, best_epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "history": history,
        "best_val_mse": best_val,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "train_seconds": time.time() - start,
    }


def train_and_test(model, train_loader, val_loader, test_loader, device, **kwargs):
    """Train with validation-based selection, then touch the test set once."""
    model, info = train(model, train_loader, val_loader, device, **kwargs)
    test_metrics = evaluate(model, test_loader, device, amp=kwargs.get("amp", False))
    info.update({f"test_{k}": v for k, v in test_metrics.items()})
    return model, info
