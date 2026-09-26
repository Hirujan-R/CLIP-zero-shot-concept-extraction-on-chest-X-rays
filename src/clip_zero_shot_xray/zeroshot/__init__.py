"""Zero-shot concept extraction core: model, prompts, scoring, and data helpers."""
from __future__ import annotations

from .data import find_repo_root, labels_matrix, load_valid, metadata, sample_images
from .model import DEFAULT_MODEL, ZeroShotModel, get_model, pick_device
from .prompts import (
    ALL_CONCEPTS,
    MIN_PREVALENCE,
    POSITIVE_PHRASES,
    PROMPT_TEMPLATES,
    build_prompt_pairs,
    concept_for_prompt,
    prompts_dataframe,
    select_concepts,
)
from .scoring import (
    auroc_from_scores,
    concept_scores,
    cosine_sim,
    posneg_scores,
    rank_prompts,
)

__all__ = [
    "ALL_CONCEPTS",
    "DEFAULT_MODEL",
    "MIN_PREVALENCE",
    "POSITIVE_PHRASES",
    "PROMPT_TEMPLATES",
    "ZeroShotModel",
    "auroc_from_scores",
    "build_prompt_pairs",
    "concept_for_prompt",
    "concept_scores",
    "cosine_sim",
    "find_repo_root",
    "get_model",
    "labels_matrix",
    "load_valid",
    "metadata",
    "pick_device",
    "posneg_scores",
    "prompts_dataframe",
    "rank_prompts",
    "sample_images",
    "select_concepts",
]
