"""Model definition and checkpoint I/O.

The checkpoint helpers exist because the original code loaded Task 1 weights
with load_state_dict(..., strict=False), which silently tolerates a total
failure to load. Pointing it at the wrong file in the same output directory
matched 0 of 209 parameters and raised nothing; training then proceeded from a
randomly initialized regression head, which converges to predicting the label
mean (R^2 ~ 0). load_checkpoint below fails loudly instead.
"""
import logging

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "ibm/MoLFormer-XL-both-10pct"

# The model ships its architecture as trust_remote_code from the Hub, and that
# code was rewritten upstream in 2026 against an unreleased transformers API.
# Pin to the last commit that imports cleanly on transformers 4.x.
DEFAULT_REVISION = "7b12d946c181a37f6012b9dc3b002275de070314"


class MolFormerRegressor(nn.Module):
    def __init__(self, base_model, dropout_rate=0.0):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        if getattr(outputs, "pooler_output", None) is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]
        return self.regressor(self.dropout(pooled))


def load_backbone(model_name=DEFAULT_MODEL, revision=DEFAULT_REVISION):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, revision=revision
    )
    model_kwargs = {"trust_remote_code": True, "revision": revision}
    if "molformer" in str(model_name).casefold():
        # MoLFormer's random-feature attention redraws its projection matrix on
        # every forward pass unless this remote-code option is enabled.  The
        # model still redraws while training; only model.eval() becomes stable.
        model_kwargs["deterministic_eval"] = True
    base = AutoModel.from_pretrained(model_name, **model_kwargs)
    return tokenizer, base


def build_regressor(model_name=DEFAULT_MODEL, revision=DEFAULT_REVISION, dropout_rate=0.0):
    tokenizer, base = load_backbone(model_name, revision)
    return tokenizer, MolFormerRegressor(base, dropout_rate=dropout_rate)


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, map_location="cpu", allow_missing=()):
    """Load weights, raising unless every parameter is accounted for.

    allow_missing names prefixes that may legitimately be absent (e.g. loading
    a backbone-only checkpoint into a model that will get a fresh head).
    """
    state = torch.load(path, map_location=map_location)
    if not isinstance(state, dict):
        raise TypeError(f"{path} does not contain a state_dict (got {type(state).__name__})")

    result = model.load_state_dict(state, strict=False)
    unexpected = list(result.unexpected_keys)
    missing = [
        k for k in result.missing_keys if not any(k.startswith(p) for p in allow_missing)
    ]

    total = len(model.state_dict())
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint {path} does not match the model: "
            f"{total - len(result.missing_keys)}/{total} parameters matched, "
            f"{len(missing)} unexpected-missing, {len(unexpected)} unexpected keys. "
            f"first missing={missing[:3]} first unexpected={unexpected[:3]}"
        )

    logger.info("loaded %d/%d parameters from %s", total, total, path)
    return model
