"""Parameter-efficient fine-tuning wrappers: BitFit, LoRA, iA3.

The LoRA implementation here differs from the original in two ways that matter:

  * B is initialized to zeros, not to random values. Hu et al. (2021) initialize
    the up-projection to zero specifically so the adapter is an exact no-op at
    step 0 and training starts from the pretrained model's true behavior. The
    original initialized both A and B randomly, injecting a ~1.3% relative
    perturbation into every wrapped layer before any gradient step.
  * The standard alpha/r scaling is applied. alpha defaults to r, giving a
    scaling of 1.0, so this changes nothing unless explicitly configured.
"""
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

DEFAULT_TARGETS = ("attention", "intermediate")


def apply_bitfit(model):
    """Freeze everything except bias terms (and the regression head, which is
    the task-specific output layer and must remain trainable)."""
    for name, param in model.named_parameters():
        param.requires_grad = ("bias" in name) or name.startswith("regressor.")
    return model


class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=8, alpha=None):
        super().__init__()
        self.original_linear = original_linear
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.r = r
        self.alpha = r if alpha is None else alpha
        self.scaling = self.alpha / self.r

        for param in self.original_linear.parameters():
            param.requires_grad = False

        self.A = nn.Parameter(torch.empty(r, self.in_features))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        # Zero init: the adapter contributes exactly nothing until it is trained.
        self.B = nn.Parameter(torch.zeros(self.out_features, r))

    def forward(self, x):
        return self.original_linear(x) + self.scaling * (x @ self.A.t() @ self.B.t())


class IA3Module(nn.Module):
    def __init__(self, original_module):
        super().__init__()
        self.original_module = original_module
        width = getattr(original_module, "out_features", None)
        self.scale = nn.Parameter(torch.ones(width if width else 1))

    def forward(self, x):
        return self.original_module(x) * self.scale


def _wrap_targets(root, target_names, make_wrapper):
    wrapped = []

    def recurse(module, prefix=""):
        for name, child in module.named_children():
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and any(t in path.lower() for t in target_names):
                setattr(module, name, make_wrapper(child))
                wrapped.append(path)
            else:
                recurse(child, path)

    recurse(root)
    return wrapped


def apply_lora(model, target_module_names=DEFAULT_TARGETS, r=8, alpha=None):
    for param in model.base_model.parameters():
        param.requires_grad = False
    wrapped = _wrap_targets(
        model.base_model, target_module_names, lambda lin: LoRALinear(lin, r=r, alpha=alpha)
    )
    logger.info("LoRA: wrapped %d linear layers (r=%d)", len(wrapped), r)
    return model


def apply_ia3(model, target_module_names=DEFAULT_TARGETS):
    for param in model.base_model.parameters():
        param.requires_grad = False
    wrapped = _wrap_targets(model.base_model, target_module_names, IA3Module)
    logger.info("iA3: wrapped %d linear layers", len(wrapped))
    return model


def apply_full_finetune(model):
    """Reference arm: everything trainable. The original project never compared
    the PEFT methods against plain full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    return model


METHODS = {
    "bitfit": apply_bitfit,
    "lora": apply_lora,
    "ia3": apply_ia3,
    "full": apply_full_finetune,
}


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
