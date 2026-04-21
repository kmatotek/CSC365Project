from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import auc, roc_curve


# -----------------------------
# LOAD
# -----------------------------
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

def fig_feature_importance(model_dir, top_n=6):
    import pickle
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    # Load model + vectorizer
    with open(model_dir / "tfidf_vectorizer.pkl", "rb") as f:
        vec = pickle.load(f)

    with open(model_dir / "logreg.pkl", "rb") as f:
        clf = pickle.load(f)

    feature_names = np.array(vec.get_feature_names_out())
    coefs = clf.coef_[0]

    # Top positive (ChatGPT-like)
    top_pos_idx = np.argsort(coefs)[-top_n:]
    top_neg_idx = np.argsort(coefs)[:top_n]

    top_pos = pd.DataFrame({
        "feature": feature_names[top_pos_idx],
        "coef": coefs[top_pos_idx],
        "label": "ChatGPT-like"
    })

    top_neg = pd.DataFrame({
        "feature": feature_names[top_neg_idx],
        "coef": coefs[top_neg_idx],
        "label": "Human-like"
    })

    # Keep clean separation (no interleaving confusion)
    df = pd.concat([
        top_neg.sort_values("coef"),
        top_pos.sort_values("coef")
    ])

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["coef"],
        y=df["feature"],
        orientation="h",
        text=[f"{c:.2f}" for c in df["coef"]],
        textposition="outside",
    ))

    fig.add_vline(x=0, line_dash="dash")

    fig.update_layout(
        title=dict(
            text="Top Linguistic Features Learned by Model",
            x=0.5,
            xanchor="center"
        ),
        xaxis=dict(
            title="Model Weight (importance)",
            range=[-12, 12],
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="black",
        ),
        yaxis=dict(
            title="Word / n-gram"
        ),
        font=dict(size=16),
        height=600,
        width=900,
        margin=dict(l=140, r=40, t=120, b=60),
    )

    # Side annotations (moved up for clarity)
    fig.add_annotation(
        x=0.98,
        y=1.12,
        xref="paper",
        yref="paper",
        text="More ChatGPT-like →",
        showarrow=False,
        align="right"
    )

    fig.add_annotation(
        x=0.02,
        y=1.12,
        xref="paper",
        yref="paper",
        text="← More Human-like",
        showarrow=False,
        align="left"
    )

    return fig

# -----------------------------
# SCORE DISTRIBUTION (FIXED LABELING)
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
        name="Human text",
        histnorm="probability",  # normalized
        opacity=0.6,
        nbinsx=40,
    ))

    fig.add_trace(go.Histogram(
        x=y_score[mask_g],
        name="ChatGPT text",
        histnorm="probability",
        opacity=0.6,
        nbinsx=40,
    ))

    fig.add_vline(x=threshold, line_dash="dash")

    fig.update_layout(
        title=dict(
            text="Model Confidence Distribution",
            x=0.5,
            xanchor="center"
        ),
        xaxis_title="Predicted Probability of ChatGPT",
        yaxis_title="Proportion of texts (per bin)",
        barmode="overlay",
        font=dict(size=18),
        legend=dict(
            orientation="h",
            y=1.08,
            x=0.5,
            xanchor="center",
            yanchor="bottom"
        ),
        margin=dict(l=70, r=50, t=140, b=70),
        height=520,
        width=780,
    )

    fig.add_annotation(
        x=0.5,
        y=1.08,
        xref="x",
        yref="paper",
        text="Decision boundary at 0.5",
        showarrow=False,
    )

    """
    fig.add_annotation(
        x=0.98,
        y=0.95,
        xref="paper",
        yref="paper",
        text=f"{pct_g_correct:.1f}% ChatGPT correctly classified",
        showarrow=False,
        align="right"
    )

    fig.add_annotation(
        x=0.98,
        y=0.88,
        xref="paper",
        yref="paper",
        text=f"{pct_h_misclassified:.1f}% human misclassified",
        showarrow=False,
        align="right"
    )
    """
    return fig


import plotly.express as px
import pandas as pd

def fig_accuracy_by_source(y_true, y_pred, source):
    source = np.array([str(s).strip() for s in source])

    rows = []
    for s in sorted(set(source)):
        mask = source == s
        if mask.sum() == 0:
            continue
        acc = (y_true[mask] == y_pred[mask]).mean()
        n = mask.sum()
        rows.append({"source": s, "accuracy": acc, "n": n})

    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)

    # Create label text
    df["label"] = df.apply(lambda r: f"{r['accuracy']:.1%}<br>n={int(r['n'])}", axis=1)

    fig = px.bar(
        df,
        x="source",
        y="accuracy",
        text="label",
    )

    fig.update_traces(
        textposition="inside",     # cleaner than inside for high values
        cliponaxis=False,        # 🔥 THIS FIXES CUT-OFF
        textfont=dict(size=14, color="white"),# ✅ slightly smaller + readable
    )

    fig.update_layout(
        title={
        'text': "Model Accuracy by Source",
        'x': 0.5,
        'xanchor': 'center'
        },
        
        xaxis_title="HC3 Source",
        yaxis_title="Accuracy",

        # keep correct bounds
        yaxis=dict(range=[0.8, 1.0]),

        font=dict(size=18),
        margin=dict(l=70, r=50, t=100, b=170),
        height=520,
        width=850,
    )

    fig.update_xaxes(tickangle=-30)

    # 🔥 THIS ensures labels don’t shrink randomly
    fig.update_layout(uniformtext_minsize=12, uniformtext_mode='show')

    return fig


# -----------------------------
# ROC
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
        name="Random baseline"
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
        "feature_importance": fig_feature_importance(model_dir),
    }

    for name, fig in figs.items():
        #fig.write_html(out / f"{name}.html", include_plotlyjs="cdn")
        pass

    if args.export:
        try:
            for name, fig in figs.items():
                #fig.write_image(out / f"{name}.png", scale=3)
                fig.write_image(out / f"{name}.svg")
                fig.write_image(out / f"{name}.pdf")
        except Exception as e:
            print("Export failed (install kaleido):", e)

    print("Saved figures to:", out.resolve())


if __name__ == "__main__":
    main()