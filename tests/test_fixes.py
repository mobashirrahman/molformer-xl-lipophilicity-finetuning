"""Regression tests for the defects found in the original implementation.

Each test corresponds to a specific bug that silently corrupted published
results, so they assert the *fix*, not merely that the code runs.
"""
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from nnti import influence as inf
from nnti.data import stratified_three_way_split
from nnti.models import MolFormerRegressor, load_backbone, load_checkpoint
from nnti.peft import LoRALinear, apply_lora


# --------------------------------------------------------------------------
# MoLFormer evaluation must not redraw its random attention features.
# --------------------------------------------------------------------------
def test_load_backbone_enables_deterministic_molformer_evaluation(monkeypatch):
    calls = {}
    tokenizer = object()
    backbone = object()

    def fake_tokenizer(model_name, **kwargs):
        calls["tokenizer"] = (model_name, kwargs)
        return tokenizer

    def fake_model(model_name, **kwargs):
        calls["model"] = (model_name, kwargs)
        return backbone

    monkeypatch.setattr("nnti.models.AutoTokenizer.from_pretrained", fake_tokenizer)
    monkeypatch.setattr("nnti.models.AutoModel.from_pretrained", fake_model)

    loaded_tokenizer, loaded_backbone = load_backbone(
        "ibm/MoLFormer-XL-both-10pct", revision="pinned-revision"
    )

    assert loaded_tokenizer is tokenizer
    assert loaded_backbone is backbone
    assert calls["model"][1]["deterministic_eval"] is True
    assert calls["model"][1]["trust_remote_code"] is True
    assert calls["model"][1]["revision"] == "pinned-revision"


def test_load_backbone_does_not_leak_molformer_option_to_other_models(monkeypatch):
    model_kwargs = {}

    monkeypatch.setattr(
        "nnti.models.AutoTokenizer.from_pretrained", lambda *args, **kwargs: object()
    )

    def fake_model(model_name, **kwargs):
        model_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr("nnti.models.AutoModel.from_pretrained", fake_model)

    load_backbone("seyonec/ChemBERTa-zinc-base-v1", revision=None)

    assert "deterministic_eval" not in model_kwargs


# --------------------------------------------------------------------------
# LiSSA: the recursion must recover a known inverse-Hessian-vector product.
# --------------------------------------------------------------------------
class _QuadraticModel(nn.Module):
    """Linear model whose MSE loss has an exactly computable Hessian."""

    def __init__(self, w0):
        super().__init__()
        self.w = nn.Parameter(w0.clone())

    def forward(self, input_ids, attention_mask=None):
        return (input_ids.float() @ self.w).unsqueeze(-1)


def _quadratic_setup(n=256, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    y = torch.randn(n, generator=g)
    model = _QuadraticModel(torch.zeros(d))
    # MSE loss = mean((Xw - y)^2)  =>  H = (2/n) X^T X, independent of w.
    H = 2.0 / n * (X.t() @ X)
    batches = [
        {"input_ids": X[i : i + 64], "attention_mask": None, "labels": y[i : i + 64]}
        for i in range(0, n, 64)
    ]
    return model, H, batches


def test_lissa_recovers_known_inverse_hvp():
    model, H, batches = _quadratic_setup()
    params = list(model.parameters())
    v = [torch.randn(8, generator=torch.Generator().manual_seed(1))]
    truth = torch.linalg.solve(H, v[0])

    ihvp = inf.lissa_inverse_hvp(
        model, batches, nn.MSELoss(), v, torch.device("cpu"), params,
        damping=0.0, scale=10.0, recursion_depth=3000,
    )
    rel = (ihvp[0] - truth).norm() / truth.norm()
    assert rel < 0.05, f"LiSSA failed to recover H^-1 v (relative error {rel:.4f})"


def test_lissa_raises_instead_of_returning_nan():
    """The original silently produced NaN scores; divergence must now raise."""
    model, _, batches = _quadratic_setup()
    params = list(model.parameters())
    v = [torch.ones(8)]
    with pytest.raises(inf.InfluenceDivergedError):
        # scale far too small => the Neumann series cannot contract.
        inf.lissa_inverse_hvp(
            model, batches, nn.MSELoss(), v, torch.device("cpu"), params,
            damping=0.0, scale=1e-3, recursion_depth=100,
        )


def test_influence_scores_reject_non_finite():
    model, _, batches = _quadratic_setup()
    params = list(model.parameters())
    bad_ihvp = [torch.full((8,), float("nan"))]
    with pytest.raises(inf.InfluenceDivergedError):
        inf.influence_scores(model, batches, nn.MSELoss(), bad_ihvp, torch.device("cpu"), params)


def test_nan_scores_would_have_degenerated_to_file_order():
    """Documents the original silent failure: sorting an all-NaN column
    returns the first k rows, so 'top-k most influential' was file order."""
    df = pd.DataFrame({"smiles": [f"C{i}" for i in range(50)], "influence": [np.nan] * 50})
    top = df.sort_values("influence", ascending=False).head(10)
    assert top.index.tolist() == list(range(10))


# --------------------------------------------------------------------------
# LoRA must be an exact no-op at initialization.
# --------------------------------------------------------------------------
def test_lora_is_exact_noop_at_init():
    torch.manual_seed(0)
    linear = nn.Linear(64, 64)
    lora = LoRALinear(linear, r=8)
    x = torch.randn(16, 64)
    with torch.no_grad():
        assert torch.equal(lora(x), linear(x)), "LoRA perturbs the model before training"


def test_lora_becomes_active_once_b_is_trained():
    torch.manual_seed(0)
    lora = LoRALinear(nn.Linear(64, 64), r=8)
    x = torch.randn(16, 64)
    with torch.no_grad():
        before = lora(x).clone()
        lora.B.add_(0.1)
        assert not torch.equal(lora(x), before)


def test_apply_lora_freezes_backbone_but_trains_adapters():
    class _Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(32, 32)
            self.config = type("cfg", (), {"hidden_size": 32})()

        def forward(self, input_ids, attention_mask):  # pragma: no cover
            raise NotImplementedError

    model = apply_lora(MolFormerRegressor(_Base()), r=4)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any(n.endswith(".A") for n in trainable)
    assert any(n.endswith(".B") for n in trainable)
    assert not any("original_linear" in n for n in trainable)


# --------------------------------------------------------------------------
# Checkpoint loading must fail loudly on a mismatch.
# --------------------------------------------------------------------------
def test_load_checkpoint_raises_on_wrong_file(tmp_path):
    class _Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(16, 16)
            self.config = type("cfg", (), {"hidden_size": 16})()

    model = MolFormerRegressor(_Base())
    wrong = tmp_path / "wrong.pt"
    torch.save({"totally.unrelated.key": torch.zeros(3)}, wrong)

    with pytest.raises(RuntimeError, match="does not match the model"):
        load_checkpoint(model, str(wrong))


def test_load_checkpoint_accepts_matching_file(tmp_path):
    class _Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(16, 16)
            self.config = type("cfg", (), {"hidden_size": 16})()

    model = MolFormerRegressor(_Base())
    good = tmp_path / "good.pt"
    torch.save(model.state_dict(), good)
    load_checkpoint(model, str(good))  # must not raise


# --------------------------------------------------------------------------
# Splits must be disjoint: the test set is no longer reused for validation.
# --------------------------------------------------------------------------
def test_three_way_split_is_disjoint_and_stratified():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"SMILES": [f"C{i}" for i in range(1000)], "label": rng.normal(2, 1.2, 1000)})
    train, val, test = stratified_three_way_split(df, split_seed=0)

    assert len(train) + len(val) + len(test) == len(df)
    for a, b in [(train, val), (train, test), (val, test)]:
        assert not set(a["SMILES"]) & set(b["SMILES"])
    # stratification should keep the target distributions close
    assert abs(train["label"].mean() - test["label"].mean()) < 0.15


def test_split_is_deterministic_per_seed_and_varies_across_seeds():
    df = pd.DataFrame({"SMILES": [f"C{i}" for i in range(500)],
                       "label": np.linspace(0, 4, 500)})
    a1, _, _ = stratified_three_way_split(df, split_seed=1)
    a2, _, _ = stratified_three_way_split(df, split_seed=1)
    b1, _, _ = stratified_three_way_split(df, split_seed=2)
    assert a1["SMILES"].tolist() == a2["SMILES"].tolist()
    assert a1["SMILES"].tolist() != b1["SMILES"].tolist()
