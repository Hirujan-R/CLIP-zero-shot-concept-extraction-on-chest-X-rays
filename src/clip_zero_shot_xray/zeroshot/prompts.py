"""Concept vocabulary and prompt templates for zero-shot CheXpert concepts."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Concept vocabulary aligned to CheXpert labels (no manual annotation needed).
ALL_CONCEPTS = [
    "Cardiomegaly",
    "Pleural Effusion",
    "Edema",
    "Consolidation",
    "Atelectasis",
    "Pneumonia",
    "Pneumothorax",
    "Lung Opacity",
    "Enlarged Cardiomediastinum",
    "Support Devices",
    "Fracture",
    "Lung Lesion",
]

POSITIVE_PHRASES = {
    "Cardiomegaly": "cardiomegaly",
    "Pleural Effusion": "pleural effusion",
    "Edema": "pulmonary edema",
    "Consolidation": "consolidation",
    "Atelectasis": "atelectasis",
    "Pneumonia": "pneumonia",
    "Pneumothorax": "pneumothorax",
    "Lung Opacity": "lung opacity",
    "Enlarged Cardiomediastinum": "an enlarged cardiomediastinum",
    "Support Devices": "support devices",
    "Fracture": "a fracture",
    "Lung Lesion": "a lung lesion",
}

PROMPT_TEMPLATES = [
    "a chest X-ray with {}.",
    "a chest radiograph showing {}.",
    "the chest X-ray demonstrates {}.",
    "a frontal chest X-ray with {}.",
]

# Concepts with fewer positives than this are dropped from AUROC evaluation.
MIN_PREVALENCE = 0.05


def positive_phrase(concept: str) -> str:
    """Return the positive phrase for a concept (defaults to lower-cased name)."""
    return POSITIVE_PHRASES.get(concept, concept.lower())


def negative_phrase(concept: str) -> str:
    """Return the negated phrase for a concept."""
    return f"no {positive_phrase(concept)}"


def build_prompt_pairs(
    concepts: list[str], templates: list[str] | None = None
) -> list[dict[str, str]]:
    """Build positive/negative prompt pairs for each concept and template."""
    templates = templates or PROMPT_TEMPLATES
    rows: list[dict[str, str]] = []
    for concept in concepts:
        for template in templates:
            rows.append(
                {
                    "concept": concept,
                    "polarity": "pos",
                    "prompt": template.format(positive_phrase(concept)),
                }
            )
            rows.append(
                {
                    "concept": concept,
                    "polarity": "neg",
                    "prompt": template.format(negative_phrase(concept)),
                }
            )
    return rows


def prompts_dataframe(
    concepts: list[str], templates: list[str] | None = None
) -> pd.DataFrame:
    """Return a prompt table with a stable ``prompt_id`` aligned to embeddings."""
    df = pd.DataFrame(build_prompt_pairs(concepts, templates))
    df["prompt_id"] = np.arange(len(df))
    return df


def select_concepts(
    labels: pd.DataFrame, min_prevalence: float = MIN_PREVALENCE
) -> list[str]:
    """Keep concepts with enough positives for a stable AUROC, in vocabulary order."""
    prevalence = labels.mean()
    return [
        c
        for c in ALL_CONCEPTS
        if c in prevalence.index and prevalence[c] >= min_prevalence
    ]


def concept_for_prompt(prompt: str, concepts: list[str]) -> str | None:
    """Return the concept a prompt refers to, if its positive phrase is present."""
    text = prompt.lower()
    for concept in concepts:
        if positive_phrase(concept).lower() in text:
            return concept
    return None
