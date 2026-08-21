"""External-data selection strategies.

Every strategy takes the same budget (n_select) so that comparisons are
matched on sample count. The original compared an influence-selected subset
against a no-external-data baseline, which confounds "does influence ranking
work" with "does adding any data help". The `influence_bottom` arm is the
sharpest control available: if the influence ranking carries signal, the
bottom-k must underperform the top-k at identical budget.
"""
import logging

import numpy as np
import torch
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

STRATEGIES = (
    "none",
    "all",
    "random",
    "target_alignment",
    "clustering",
    "influence_top",
    "influence_bottom",
)


@torch.no_grad()
def embed_smiles(smiles, backbone, tokenizer, device, batch_size=64, amp=True):
    """Mean [CLS]/pooled embedding for each SMILES string."""
    backbone.to(device).eval()
    out = []
    for i in range(0, len(smiles), batch_size):
        enc = tokenizer(
            smiles[i : i + batch_size], padding=True, truncation=True,
            max_length=256, return_tensors="pt",
        )
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            res = backbone(input_ids=ids, attention_mask=mask)
        pooled = getattr(res, "pooler_output", None)
        if pooled is None:
            pooled = res.last_hidden_state[:, 0]
        out.append(pooled.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def select(
    strategy,
    external_df,
    n_select,
    seed,
    influence=None,
    external_embeddings=None,
    target_embedding=None,
):
    """Return the selected subset of external_df.

    influence: array of per-row scores, required by the influence_* strategies.
    external_embeddings / target_embedding: required by embedding-based ones.
    """
    if strategy == "none":
        return external_df.iloc[0:0].copy()
    if strategy == "all":
        return external_df.copy()

    n_select = int(min(max(n_select, 0), len(external_df)))
    if n_select == 0:
        return external_df.iloc[0:0].copy()

    df = external_df.copy().reset_index(drop=True)

    if strategy == "random":
        return df.sample(n=n_select, random_state=seed).copy()

    if strategy in ("influence_top", "influence_bottom"):
        if influence is None:
            raise ValueError(f"{strategy} requires influence scores")
        scores = np.asarray(influence, dtype=float)
        if len(scores) != len(df):
            raise ValueError(f"influence length {len(scores)} != external rows {len(df)}")
        if not np.isfinite(scores).all():
            raise ValueError("influence scores contain non-finite values")
        df["influence"] = scores
        ascending = strategy == "influence_bottom"
        return df.sort_values("influence", ascending=ascending, kind="mergesort").head(n_select).copy()

    if strategy == "target_alignment":
        if external_embeddings is None or target_embedding is None:
            raise ValueError("target_alignment requires embeddings")
        d = np.linalg.norm(external_embeddings - target_embedding, axis=1)
        df["_distance"] = d
        return df.nsmallest(n_select, "_distance").drop(columns=["_distance"]).copy()

    if strategy == "clustering":
        if external_embeddings is None:
            raise ValueError("clustering requires embeddings")
        km = KMeans(n_clusters=n_select, random_state=seed, n_init=10)
        km.fit(external_embeddings)
        picks = []
        for c in range(km.n_clusters):
            idx = np.where(km.labels_ == c)[0]
            if len(idx) == 0:
                continue
            d = np.linalg.norm(external_embeddings[idx] - km.cluster_centers_[c], axis=1)
            picks.append(idx[int(np.argmin(d))])
        return df.iloc[sorted(set(picks))].copy()

    raise ValueError(f"unknown selection strategy: {strategy}")
