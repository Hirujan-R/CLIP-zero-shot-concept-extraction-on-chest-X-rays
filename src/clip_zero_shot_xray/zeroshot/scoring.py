"""Scoring functions: pos/neg concept scores, prompt ranking, and AUROC."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _softmax(logits: np.ndarray, scale: float) -> np.ndarray:
    """Numerically stable softmax over the last axis, scaled by ``scale``."""
    scaled = np.asarray(logits, dtype=np.float64) * float(scale)
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=-1, keepdims=True)


def _pair_embeddings(
    text_embs: np.ndarray, prompts_df: pd.DataFrame, concept: str
) -> tuple[np.ndarray, np.ndarray]:
    sub = prompts_df[prompts_df["concept"] == concept]
    pos_ids = sub.loc[sub["polarity"] == "pos", "prompt_id"].to_numpy()
    neg_ids = sub.loc[sub["polarity"] == "neg", "prompt_id"].to_numpy()
    pos = text_embs[pos_ids].mean(axis=0)
    neg = text_embs[neg_ids].mean(axis=0)
    pos = pos / np.linalg.norm(pos)
    neg = neg / np.linalg.norm(neg)
    return pos, neg


def posneg_scores(
    image_embs: np.ndarray,
    text_embs: np.ndarray,
    prompts_df: pd.DataFrame,
    logit_scale: float,
) -> pd.DataFrame:
    """Return an (n_images x n_concepts) matrix of pos-vs-neg softmax scores."""
    image_embs = np.atleast_2d(image_embs)
    out: dict[str, np.ndarray] = {}
    for concept in prompts_df["concept"].unique():
        pos, neg = _pair_embeddings(text_embs, prompts_df, concept)
        logits = np.stack([image_embs @ pos, image_embs @ neg], axis=1)
        out[concept] = _softmax(logits, logit_scale)[:, 0]
    return pd.DataFrame(out)


def concept_scores(
    image_embs: np.ndarray,
    text_embs: np.ndarray,
    prompts_df: pd.DataFrame,
    logit_scale: float,
    raw: bool = False,
) -> pd.DataFrame:
    """Score a single image per concept, optionally returning raw cosine margins."""
    image_embs = np.atleast_2d(image_embs)
    rows: list[dict[str, float | str]] = []
    for concept in prompts_df["concept"].unique():
        pos, neg = _pair_embeddings(text_embs, prompts_df, concept)
        cos_pos = float((image_embs @ pos)[0])
        cos_neg = float((image_embs @ neg)[0])
        if raw:
            score = cos_pos - cos_neg
        else:
            score = float(_softmax(np.array([[cos_pos, cos_neg]]), logit_scale)[0, 0])
        rows.append(
            {
                "concept": concept,
                "score": score,
                "cos_pos": cos_pos,
                "cos_neg": cos_neg,
            }
        )
    return pd.DataFrame(rows)


def rank_prompts(
    image_embs: np.ndarray,
    texts: list[str],
    text_embs: np.ndarray,
    logit_scale: float,
    raw: bool = False,
) -> pd.DataFrame:
    """Rank arbitrary prompts per image via multi-class softmax (classic CLIP)."""
    image_embs = np.atleast_2d(image_embs)
    logits = image_embs @ np.asarray(text_embs).T
    if raw:
        probs = logits
    else:
        probs = _softmax(logits, logit_scale)
    return pd.DataFrame(probs, columns=list(texts))


def cosine_sim(image_embs: np.ndarray, text_embs: np.ndarray) -> np.ndarray:
    """Return the raw cosine similarity matrix between images and texts."""
    return np.atleast_2d(image_embs) @ np.asarray(text_embs).T


def auroc_from_scores(
    scores: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    """Compute per-concept AUROC and prevalence against ground-truth labels."""
    rows: list[dict[str, float | str]] = []
    for concept in scores.columns:
        if concept not in labels.columns:
            continue
        y = labels[concept].to_numpy()
        if y.min() == y.max():
            auroc = float("nan")
        else:
            auroc = roc_auc_score(y, scores[concept].to_numpy())
        rows.append(
            {"concept": concept, "auroc": auroc, "prevalence": float(y.mean())}
        )
    return (
        pd.DataFrame(rows)
        .sort_values("auroc", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
