from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import auc, roc_curve


def load_bundle(model_dir: Path):
    with open(model_dir / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)

    z = np.load(model_dir / "test_eval.npz", allow_pickle=True)

    return (
        metrics,
        z["y_true"].astype(np.int64),
        z["y_pred"].astype(np.int64),
        z["y_score"].astype(np.float64),
        z["source"].astype(str),
    )


# -----------------------------
# FIXED SCORE DISTRIBUTION
# -----------------------------
def fig_score_distribution(y_true, y_score):
    mask_h = y_true == 0
    mask_g = y_true == 1

    threshold = 0.5

    pct_h_misclassified = (y_score[mask_h] > threshold).mean() * 100
    pct_g_correct = (y_score[mask_g] > threshold).mean() * 100

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=y_score[mask_h],
        name="Human",
        histnorm="probability density",
        opacity=0.6,
        nbinsx=40,
    ))

    fig.add_trace(go.Histogram(
        x=y_score[mask_g],
        name="ChatGPT",
        histnorm="probability density",
        opacity=0.6,
        nbinsx=40,
    ))

    # Decision boundary (no text here anymore)
    fig.add_vline(x=threshold, line_dash="dash")

    fig.update_layout(
        title="Model Confidence Distribution",
        xaxis_title="Predicted Probability of ChatGPT",
        yaxis_title="Density",
        barmode="overlay",
        font=dict(size=18),
        legend=dict(
            orientation="h",
            y=1.2,
            x=0
        ),
        margin=dict(l=70, r=50, t=120, b=70),
        height=520,
        width=760,
    )

    # Clean annotations (no overlap)
    fig.add_annotation(
        x=0.5,
        y=1.08,
        xref="x",
        yref="paper",
        text="Decision boundary (0.5)",
        showarrow=False,
        font=dict(size=14)
    )

    fig.add_annotation(
        x=0.98,
        y=0.95,
        xref="paper",
        yref="paper",
        text=f"{pct_g_correct:.1f}% of ChatGPT texts correctly classified",
        showarrow=False,
        align="right"
    )

    fig.add_annotation(
        x=0.98,
        y=0.88,
        xref="paper",
        yref="paper",
        text=f"{pct_h_misclassified:.1f}% of human texts misclassified",
        showarrow=False,
        align="right"
    )

    return fig


# -----------------------------
# FIXED SOURCE ACCURACY
# -----------------------------
def fig_accuracy_by_source(y_true, y_pred, source):
    # Convert to clean Python strings (fixes hidden issues)
    source = np.array([str(s).strip() for s in source])

    unique_sources = sorted(set(source))

    data = []
    for s in unique_sources:
        mask = source == s

        if mask.sum() == 0:
            continue

        acc = (y_true[mask] == y_pred[mask]).mean()
        n = mask.sum()

        data.append((s, acc, n))

    # Sort by accuracy
    data.sort(key=lambda x: x[1], reverse=True)

    names, accs, supports = zip(*data)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=list(names),
        y=list(accs),
        text=[f"{a:.1%}<br>n={n}" for a, n in zip(accs, supports)],
        textposition="outside",
    ))

    fig.update_layout(
        title="Model Accuracy by Source",
        xaxis_title="HC3 Source",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0, 1.05]),
        font=dict(size=18),
        margin=dict(l=70, r=50, t=80, b=160),
        height=520,
        width=850,
    )

    fig.update_xaxes(tickangle=-30)

    return fig


# -----------------------------
# CLEAN ROC CURVE
# -----------------------------
def fig_roc(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name=f"AUC = {roc_auc:.3f}",
        line=dict(width=3)
    ))

    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        line=dict(dash="dash"),
        name="Random"
    ))

    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        font=dict(size=18),
        height=520,
        width=520,
    )

    return fig


# -----------------------------
# MAIN
# -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="train_outputs")
    p.add_argument("--export", action="store_true")
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    out = model_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)

    metrics, y_true, y_pred, y_score, source = load_bundle(model_dir)

    figs = {
        "score_distribution": fig_score_distribution(y_true, y_score),
        "accuracy_by_source": fig_accuracy_by_source(y_true, y_pred, source),
        "roc_curve": fig_roc(y_true, y_score),
    }

    for name, fig in figs.items():
        #fig.write_html(out / f"{name}.html", include_plotlyjs="cdn")
        pass

    if args.export:
        try:
            for name, fig in figs.items():
                #fig.write_image(out / f"{name}.png", scale=3)
                fig.write_image(out / f"{name}.svg")
                #fig.write_image(out / f"{name}.pdf")
        except Exception as e:
            print("Export failed (install kaleido):", e)

    print("Saved figures to:", out.resolve())


if __name__ == "__main__":
    main()