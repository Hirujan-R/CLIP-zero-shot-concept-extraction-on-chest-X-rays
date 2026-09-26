"""Tests for the zero-shot core (no model download required)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from clip_zero_shot_xray.zeroshot import (
    ALL_CONCEPTS,
    PROMPT_TEMPLATES,
    auroc_from_scores,
    build_prompt_pairs,
    concept_for_prompt,
    concept_scores,
    prompts_dataframe,
    rank_prompts,
    select_concepts,
)

ONE_TEMPLATE = ["a chest X-ray with {}."]
DECISION_THRESHOLD = 0.5


def test_build_prompt_pairs_counts_and_polarity() -> None:
    pairs = build_prompt_pairs(["Cardiomegaly", "Edema"])
    expected = 2 * len(PROMPT_TEMPLATES) * 2  # 2 concepts x 4 templates x 2 polarities
    assert len(pairs) == expected
    polarities = {row["polarity"] for row in pairs}
    assert polarities == {"pos", "neg"}
    positive = next(r for r in pairs if r["concept"] == "Edema" and r["polarity"] == "pos")
    assert "pulmonary edema" in positive["prompt"]


def test_prompts_dataframe_has_unique_ids() -> None:
    df = prompts_dataframe(["Cardiomegaly"], ONE_TEMPLATE)
    assert list(df["prompt_id"]) == [0, 1]
    assert set(df["polarity"]) == {"pos", "neg"}


def test_select_concepts_drops_rare() -> None:
    labels = pd.DataFrame(
        {"Cardiomegaly": [1, 0, 1, 0], "Fracture": [0, 0, 0, 0]}
    )
    kept = select_concepts(labels, min_prevalence=0.05)
    assert "Cardiomegaly" in kept
    assert "Fracture" not in kept


def test_concept_for_prompt_matches_phrase() -> None:
    assert concept_for_prompt("a chest X-ray with pleural effusion.", ALL_CONCEPTS) == (
        "Pleural Effusion"
    )
    assert concept_for_prompt("a cat", ALL_CONCEPTS) is None


def test_posneg_concept_scores_prefer_positive() -> None:
    pdf = prompts_dataframe(["Cardiomegaly"], ONE_TEMPLATE)
    text_emb = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    image_emb = np.array([[1.0, 0.0, 0.0, 0.0]])
    df = concept_scores(image_emb, text_emb, pdf, logit_scale=10.0)
    assert df.loc[0, "score"] > DECISION_THRESHOLD
    assert df.loc[0, "cos_pos"] > df.loc[0, "cos_neg"]


def test_concept_scores_raw_margin() -> None:
    pdf = prompts_dataframe(["Cardiomegaly"], ONE_TEMPLATE)
    text_emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    image_emb = np.array([[1.0, 0.0]])
    df = concept_scores(image_emb, text_emb, pdf, logit_scale=10.0, raw=True)
    assert df.loc[0, "score"] == pytest.approx(1.0)


def test_rank_prompts_is_distribution() -> None:
    texts = ["a", "b", "c"]
    text_emb = np.eye(3)
    image_emb = np.array([[0.2, 0.5, 0.3]])
    probs = rank_prompts(image_emb, texts, text_emb, logit_scale=5.0)
    assert list(probs.columns) == texts
    assert probs.to_numpy().sum() == pytest.approx(1.0)


def test_rank_prompts_raw_returns_logits() -> None:
    texts = ["a", "b"]
    text_emb = np.eye(2)
    image_emb = np.array([[0.2, 0.8]])
    probs = rank_prompts(image_emb, texts, text_emb, logit_scale=5.0, raw=True)
    assert probs.loc[0, "b"] == pytest.approx(0.8)


def test_auroc_from_scores_perfect_separation() -> None:
    scores = pd.DataFrame({"Cardiomegaly": [0.9, 0.8, 0.2, 0.1]})
    labels = pd.DataFrame({"Cardiomegaly": [1.0, 1.0, 0.0, 0.0]})
    result = auroc_from_scores(scores, labels)
    assert result.loc[0, "auroc"] == pytest.approx(1.0)
    assert result.loc[0, "prevalence"] == pytest.approx(0.5)


def test_auroc_from_scores_single_class_is_nan() -> None:
    scores = pd.DataFrame({"Cardiomegaly": [0.9, 0.8]})
    labels = pd.DataFrame({"Cardiomegaly": [1.0, 1.0]})
    result = auroc_from_scores(scores, labels)
    assert np.isnan(result.loc[0, "auroc"])
