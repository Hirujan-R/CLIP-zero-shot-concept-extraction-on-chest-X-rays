"""Gradio dashboard for zero-shot chest X-ray concept extraction.

Run from the repo root with::

    uv pip install -e ".[app]"
    python dashboard/app.py
"""
from __future__ import annotations

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from sklearn.metrics import roc_auc_score

from clip_zero_shot_xray.zeroshot import (
    DEFAULT_MODEL,
    POSITIVE_PHRASES,
    PROMPT_TEMPLATES,
    concept_for_prompt,
    concept_scores,
    get_model,
    labels_matrix,
    load_valid,
    pick_device,
    prompts_dataframe,
    rank_prompts,
    sample_images,
    select_concepts,
)

ACCENT = "#2E6FB7"
POSITIVE_THRESHOLD = 0.5
DISCLAIMER = (
    "**Research and educational use only.** Zero-shot concept scores are model "
    "similarities, not diagnoses, and are not calibrated probabilities. Do not "
    "use for clinical decision-making. Uploaded images are processed locally."
)


def empty_fig(message: str = "Nothing to show yet") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper"
    )
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def bar_fig(labels: list[str], values: list[float], title: str, x_title: str) -> go.Figure:
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=ACCENT))
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        height=max(300, 28 * len(labels) + 140),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def box_fig(prompts: list[str], matrix: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure()
    for i, prompt in enumerate(prompts):
        fig.add_trace(go.Box(y=matrix[:, i], name=prompt, boxpoints="all"))
    fig.update_layout(title=title, yaxis_title="score", height=460)
    return fig


def parse_prompts(text: str | None) -> list[str]:
    if not text:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def model():
    return get_model()


try:
    DF = load_valid()
    LABELS = labels_matrix(DF)
    KEPT = select_concepts(LABELS)
except FileNotFoundError:
    DF = None
    LABELS = None
    KEPT = []


def score_uploaded(image, concepts, custom_text, raw):
    if image is None:
        raise gr.Error("Upload a chest X-ray image first.")
    emb = model().encode_images(image.convert("RGB"))

    concept_df = pd.DataFrame()
    concept_figure = empty_fig("Select at least one concept")
    if concepts:
        pdf = prompts_dataframe(list(concepts), PROMPT_TEMPLATES)
        text_emb = model().encode_texts(pdf["prompt"].tolist())
        concept_df = concept_scores(
            emb, text_emb, pdf, model().logit_scale, raw=raw
        ).sort_values("score", ascending=False)
        concept_figure = bar_fig(
            concept_df["concept"].tolist(),
            concept_df["score"].tolist(),
            "Per-concept score",
            "cosine margin" if raw else "P(cardiomegaly-like)",
        )

    custom_df = pd.DataFrame()
    custom_figure = empty_fig("Add custom prompts to rank")
    texts = parse_prompts(custom_text)
    if texts:
        text_emb = model().encode_texts(texts)
        ranked = rank_prompts(emb, texts, text_emb, model().logit_scale, raw=raw)
        custom_df = (
            ranked.T.reset_index()
            .rename(columns={"index": "prompt", 0: "score"})
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )
        custom_figure = bar_fig(
            custom_df["prompt"].tolist(),
            custom_df["score"].tolist(),
            "Custom prompt ranking",
            "cosine" if raw else "softmax prob",
        )

    status = f"Scored {len(concepts)} concepts and {len(texts)} custom prompts."
    return concept_figure, concept_df, custom_figure, custom_df, status


def load_random(concepts, raw):
    if DF is None:
        raise gr.Error(
            "CheXpert valid data not found under data/01_raw/chexpert."
        )
    row = DF.sample(n=1, random_state=int(np.random.randint(0, 1_000_000))).iloc[0]
    image = Image.open(row["image_path"]).convert("RGB")
    emb = model().encode_images(image)

    concepts = list(concepts)
    table = pd.DataFrame()
    figure = empty_fig("Select at least one concept")
    if concepts:
        pdf = prompts_dataframe(concepts, PROMPT_TEMPLATES)
        text_emb = model().encode_texts(pdf["prompt"].tolist())
        df = concept_scores(emb, text_emb, pdf, model().logit_scale, raw=raw)
        df["ground_truth"] = [float(row[c]) for c in df["concept"]]
        if not raw:
            df["correct"] = (df["score"] >= POSITIVE_THRESHOLD) == (
                df["ground_truth"] == 1.0
            )
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        table = df
        figure = bar_fig(
            df["concept"].tolist(),
            df["score"].tolist(),
            "Prediction vs ground truth",
            "cosine margin" if raw else "score",
        )
    status = f"Sampled `{row['Path']}`"
    return image, figure, table, status


def run_over_dataset(prompts_text, n, raw):
    if DF is None:
        raise gr.Error(
            "CheXpert valid data not found under data/01_raw/chexpert."
        )
    prompts = parse_prompts(prompts_text)
    if not prompts:
        prompts = [f"a chest X-ray with {POSITIVE_PHRASES[c]}." for c in KEPT]
    if not prompts:
        raise gr.Error("Enter at least one prompt.")

    sample = sample_images(DF, n=n, seed=int(np.random.randint(0, 1_000_000)))
    images = [Image.open(p).convert("RGB") for p in sample["image_path"]]
    image_emb = model().encode_images(images)
    text_emb = model().encode_texts(prompts)
    probs = rank_prompts(image_emb, prompts, text_emb, model().logit_scale, raw=raw)

    rows = []
    for prompt in prompts:
        concept = concept_for_prompt(prompt, KEPT)
        auroc = np.nan
        prevalence = np.nan
        if concept is not None and concept in LABELS.columns:
            y = sample[concept].to_numpy()
            prevalence = float(y.mean())
            if y.min() != y.max():
                auroc = roc_auc_score(y, probs[prompt].to_numpy())
        rows.append(
            {
                "prompt": prompt,
                "concept": concept or "",
                "mean_score": float(probs[prompt].mean()),
                "std_score": float(probs[prompt].std()),
                "auroc": auroc,
                "prevalence": prevalence,
            }
        )
    table = pd.DataFrame(rows).sort_values(
        "mean_score", ascending=False
    ).reset_index(drop=True)

    mean_fig = bar_fig(
        table["prompt"].tolist(),
        table["mean_score"].tolist(),
        f"Mean score over {len(sample)} sampled images",
        "cosine" if raw else "mean softmax prob",
    )
    dist_fig = box_fig(
        prompts,
        probs[prompts].to_numpy(),
        f"Score distribution over {len(sample)} sampled images",
    )
    status = f"Scored {len(prompts)} prompts over {len(sample)} random valid images."
    return mean_fig, dist_fig, table, status


def build_app() -> gr.Blocks:
    device = pick_device()
    with gr.Blocks(title="CheXpert Zero-Shot Concept Explorer") as demo:
        gr.Markdown("# CheXpert Zero-Shot Concept Explorer")
        gr.Markdown(
            f"Model: `{DEFAULT_MODEL}`  \nDevice: `{device}`  \n"
            f"Eval set: {0 if DF is None else len(DF)} frontal CheXpert valid images"
        )

        with gr.Tab("X-ray -> concepts"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_in = gr.Image(
                        type="pil", label="Chest X-ray", sources=["upload", "clipboard"]
                    )
                    concepts_cb = gr.CheckboxGroup(
                        choices=KEPT, value=KEPT, label="Concepts"
                    )
                    custom_tb = gr.Textbox(
                        lines=4,
                        label="Custom prompts (one per line, optional)",
                        placeholder="a chest X-ray with pneumonia.",
                    )
                    raw_cb = gr.Checkbox(
                        value=False,
                        label="Show raw cosine margin instead of softmax",
                    )
                    score_btn = gr.Button("Score uploaded image", variant="primary")
                    sample_btn = gr.Button("Load random valid image (with ground truth)")
                with gr.Column(scale=2):
                    concept_plot = gr.Plot(label="Concept scores")
                    concept_tbl = gr.Dataframe(label="Concept scores", interactive=False)
                    custom_plot = gr.Plot(label="Custom prompt ranking")
                    custom_tbl = gr.Dataframe(
                        label="Custom prompt scores", interactive=False
                    )
                    status = gr.Markdown()
                    sample_img = gr.Image(
                        label="Sampled valid image", interactive=False
                    )
                    sample_plot = gr.Plot(label="Prediction vs ground truth")
                    sample_tbl = gr.Dataframe(
                        label="Prediction vs ground truth", interactive=False
                    )
                    sample_status = gr.Markdown()

            score_btn.click(
                score_uploaded,
                inputs=[image_in, concepts_cb, custom_tb, raw_cb],
                outputs=[concept_plot, concept_tbl, custom_plot, custom_tbl, status],
            )
            sample_btn.click(
                load_random,
                inputs=[concepts_cb, raw_cb],
                outputs=[sample_img, sample_plot, sample_tbl, sample_status],
            )

        with gr.Tab("Prompts over dataset"):
            gr.Markdown(
                "Score free-form prompts across random CheXpert valid images. "
                "Prompts whose phrase matches a concept get an AUROC column."
            )
            prompts_tb = gr.Textbox(
                lines=6,
                label="Prompts (one per line)",
                value="\n".join(
                    f"a chest X-ray with {POSITIVE_PHRASES[c]}." for c in KEPT
                ),
            )
            with gr.Row():
                n_slider = gr.Slider(
                    minimum=5,
                    maximum=max(5, 0 if DF is None else len(DF)),
                    value=20,
                    step=1,
                    label="Number of random images",
                )
                raw_cb2 = gr.Checkbox(
                    value=False, label="Raw cosine instead of softmax"
                )
            run_btn = gr.Button("Run over dataset", variant="primary")
            prompt_plot = gr.Plot(label="Mean score per prompt")
            dist_plot = gr.Plot(label="Score distribution")
            prompt_tbl = gr.Dataframe(label="Prompt summary", interactive=False)
            ds_status = gr.Markdown()

            run_btn.click(
                run_over_dataset,
                inputs=[prompts_tb, n_slider, raw_cb2],
                outputs=[prompt_plot, dist_plot, prompt_tbl, ds_status],
            )

        gr.Markdown(DISCLAIMER)
    return demo


if __name__ == "__main__":
    build_app().launch()
