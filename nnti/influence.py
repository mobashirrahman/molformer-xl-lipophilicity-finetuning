"""Influence functions via the LiSSA inverse-Hessian-vector product.

The recursion follows Koh & Liang (2017) / Agarwal et al. (2016):

    h_0     = v
    h_{j+1} = v + (1 - damping) * h_j - (H h_j) / scale
    H^-1 v ~= h_J / scale

The original implementation multiplied by `scale` instead of dividing, and
never applied the final division. On a synthetic quadratic with a known
ground-truth H^-1 v that recursion reaches inf within 10 steps; at the default
recursion_depth of 100 every influence score came out NaN. Those NaNs were
written to CSV unchecked, and the subsequent `sort_values(...).head(k)` on an
all-NaN column silently degenerates to "take the first k rows of the file", so
selection was effectively arbitrary. Hence the explicit finiteness guards here.
"""
import logging

import torch

logger = logging.getLogger(__name__)


class InfluenceDivergedError(RuntimeError):
    """Raised when the LiSSA recursion fails to stay finite/bounded."""


def select_parameters(model, scope="all"):
    """Choose which parameters the influence computation runs over.

    Influence estimates on full deep networks are known to be unstable, and
    restricting to the head or the top block is a common practical choice, so
    the scope is exposed as an experimental variable rather than hardcoded.
    """
    if scope == "all":
        params = [p for p in model.parameters() if p.requires_grad]
    elif scope == "head":
        params = [p for p in model.regressor.parameters() if p.requires_grad]
    elif scope == "head+pooler":
        params = [p for n, p in model.named_parameters() if p.requires_grad
                  and ("regressor" in n or "pooler" in n)]
    else:
        raise ValueError(f"unknown parameter scope: {scope}")
    if not params:
        raise ValueError(f"parameter scope {scope!r} selected no trainable parameters")
    return params


def _flatten(tensors):
    return torch.cat([t.reshape(-1) for t in tensors])


def batch_loss(model, loss_fn, batch, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    labels = batch["labels"].to(device)
    preds = model(input_ids, attention_mask).squeeze(-1)
    return loss_fn(preds, labels)


def compute_gradient(model, loss_fn, batch, device, params):
    model.zero_grad(set_to_none=True)
    loss = batch_loss(model, loss_fn, batch, device)
    grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
    return [g.detach() for g in grads]


def hessian_vector_product(model, loss_fn, batch, vec, device, params):
    model.zero_grad(set_to_none=True)
    loss = batch_loss(model, loss_fn, batch, device)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    dot = sum((g * v).sum() for g, v in zip(grads, vec))
    hv = torch.autograd.grad(dot, params)
    return [h.detach() for h in hv]


def average_gradient(model, loader, loss_fn, device, params):
    """Mean gradient of the loss over a dataset.

    This should be computed on the *validation* set. The original computed it
    on the test set and then reported final metrics on that same test set,
    which leaks the evaluation target into the data-selection step.
    """
    model.eval()
    total = None
    n = 0
    for batch in loader:
        grads = compute_gradient(model, loss_fn, batch, device, params)
        total = grads if total is None else [a + b for a, b in zip(total, grads)]
        n += 1
    if n == 0:
        raise ValueError("empty loader passed to average_gradient")
    return [t / n for t in total]


def lissa_inverse_hvp(
    model,
    loader,
    loss_fn,
    v,
    device,
    params,
    damping=0.01,
    scale=25.0,
    recursion_depth=100,
    divergence_factor=1e6,
):
    """Stochastic estimate of H^-1 v. Raises if the recursion diverges."""
    model.eval()
    h = [vi.clone().detach() for vi in v]
    v0_norm = _flatten(v).norm().item()
    it = iter(loader)

    for step in range(recursion_depth):
        try:
            batch = next(it)
        except StopIteration:
            # The original re-called iter() on the already-exhausted *iterator*,
            # which returns the same exhausted object and cannot cycle.
            it = iter(loader)
            batch = next(it)

        hv = hessian_vector_product(model, loss_fn, batch, h, device, params)
        h = [vi + (1.0 - damping) * hi - hv_i / scale for vi, hi, hv_i in zip(v, h, hv)]

        norm = _flatten(h).norm().item()
        if not torch.isfinite(torch.tensor(norm)):
            raise InfluenceDivergedError(
                f"LiSSA produced a non-finite estimate at step {step} "
                f"(damping={damping}, scale={scale})"
            )
        if v0_norm > 0 and norm > divergence_factor * v0_norm:
            raise InfluenceDivergedError(
                f"LiSSA diverging at step {step}: ||h||={norm:.4e} exceeds "
                f"{divergence_factor:g}x ||v||={v0_norm:.4e}. Increase `scale` "
                f"so the Neumann series contracts."
            )
        if step % 20 == 0:
            logger.debug("LiSSA step %d: ||h||=%.6e", step, norm)

    ihvp = [hi / scale for hi in h]
    flat = _flatten(ihvp)
    if not torch.isfinite(flat).all():
        raise InfluenceDivergedError("LiSSA returned non-finite values after scaling")
    logger.info("LiSSA converged: ||H^-1 v||=%.6e over %d steps", flat.norm().item(), recursion_depth)
    return ihvp


def influence_scores(model, external_loader, loss_fn, ihvp, device, params):
    """Score each external sample: influence(z) = -grad(z)^T H^-1 v.

    A more positive score means adding z is predicted to reduce the loss.
    """
    model.eval()
    ihvp_flat = _flatten(ihvp)
    scores = []
    for batch in external_loader:
        grad = compute_gradient(model, loss_fn, batch, device, params)
        scores.append(-torch.dot(_flatten(grad), ihvp_flat).item())

    tensor = torch.tensor(scores)
    if not torch.isfinite(tensor).all():
        n_bad = int((~torch.isfinite(tensor)).sum())
        raise InfluenceDivergedError(
            f"{n_bad}/{len(scores)} influence scores are non-finite; refusing to "
            f"write them. This is the failure mode that silently produced an "
            f"all-NaN score column in the original run."
        )
    if tensor.std().item() == 0.0:
        raise InfluenceDivergedError(
            "all influence scores are identical; ranking would be meaningless"
        )
    return scores
