"""
HC3 English: train a lightweight ChatGPT vs human detector.

Loads JSONL from the Hugging Face dataset repo (works with datasets>=4 where HC3.py
scripts are disabled). Each row has parallel human_answers and chatgpt_answers;
we flatten to (text, label) with label 0=human, 1=chatgpt.

Split is GROUPED BY ROW so the same question never appears in both train and test
(leakage control). Use --sources to restrict domains for a cleaner story.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


HC3_JSONL = {
    "all": "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl",
    "finance": "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/finance.jsonl",
    "medicine": "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/medicine.jsonl",
    "open_qa": "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/open_qa.jsonl",
    "reddit_eli5": "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/reddit_eli5.jsonl",
    "wiki_csai": "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/wiki_csai.jsonl",
}


def _as_list(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x] if x.strip() else []
    if isinstance(x, (list, tuple)):
        return [str(t).strip() for t in x if str(t).strip()]
    return [str(x).strip()] if str(x).strip() else []


def row_to_examples(
    row: dict,
    prepend_question: bool,
    min_chars: int,
    max_chars: int,
    row_id: int,
) -> list[dict]:
    q = (row.get("question") or "").strip()
    human = _as_list(row.get("human_answers"))
    gpt = _as_list(row.get("chatgpt_answers"))
    source = row.get("source") or "unknown"

    out: list[dict] = []
    for a in human:
        text = _format_text(q, a, prepend_question)
        if len(text) < min_chars or len(text) > max_chars:
            continue
        out.append({"text": text, "label": 0, "source": source, "group": row_id})

    for a in gpt:
        text = _format_text(q, a, prepend_question)
        if len(text) < min_chars or len(text) > max_chars:
            continue
        out.append({"text": text, "label": 1, "source": source, "group": row_id})
    return out


def _format_text(question: str, answer: str, prepend_question: bool) -> str:
    answer = re.sub(r"\s+", " ", answer).strip()
    if not prepend_question or not question:
        return answer
    question = re.sub(r"\s+", " ", question).strip()
    return f"Question: {question} Answer: {answer}"


def load_hc3_flat(
    subset: str,
    sources_allow: set[str] | None,
    prepend_question: bool,
    min_chars: int,
    max_chars: int,
) -> Dataset:
    if subset not in HC3_JSONL:
        raise ValueError(f"subset must be one of {list(HC3_JSONL)}")

    raw = load_dataset("json", data_files=HC3_JSONL[subset], split="train")
    if sources_allow:
        raw = raw.filter(lambda r: r["source"] in sources_allow)

    rows: list[dict] = []
    for i, row in enumerate(raw):
        rows.extend(
            row_to_examples(
                row, prepend_question, min_chars, max_chars, row_id=i
            )
        )
    return Dataset.from_list(rows)


def train_val_test_groups(
    texts: list[str],
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int,
    test_frac: float,
    val_frac: float,
):
    """Hold out test by group; split remaining train pool into train/val by group."""
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train_idx, test_idx = next(
        gss_test.split(np.zeros(len(texts)), labels, groups)
    )

    groups_train = groups[train_idx]
    y_train = labels[train_idx]
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed + 1)
    tr_rel_idx, va_rel_idx = next(
        gss_val.split(np.zeros(len(train_idx)), y_train, groups_train)
    )
    train_idx_final = train_idx[tr_rel_idx]
    val_idx = train_idx[va_rel_idx]
    return train_idx_final, val_idx, test_idx


def main():
    p = argparse.ArgumentParser(description="HC3 poster-friendly detector (English JSONL)")
    p.add_argument(
        "--subset",
        default="all",
        choices=list(HC3_JSONL),
        help="HC3 JSONL slice (English only on this Hub repo).",
    )
    p.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="If set, keep only these source tags (e.g. wiki_csai open_qa).",
    )
    p.add_argument("--prepend-question", action="store_true", help="QA-style input.")
    p.add_argument("--min-chars", type=int, default=80)
    p.add_argument("--max-chars", type=int, default=8000)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-features", type=int, default=50_000)
    p.add_argument("--ngram-max", type=int, default=2)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument(
        "--out-dir",
        type=str,
        default="train_outputs",
        help="Directory for vectorizer + model + metrics JSON.",
    )
    args = p.parse_args()

    sources_allow = set(args.sources) if args.sources else None
    ds = load_hc3_flat(
        args.subset,
        sources_allow,
        args.prepend_question,
        args.min_chars,
        args.max_chars,
    )
    if len(ds) < 200:
        raise SystemExit(f"Too few examples after filters: {len(ds)}")

    texts = ds["text"]
    labels = np.asarray(ds["label"], dtype=np.int64)
    groups = np.asarray(ds["group"], dtype=np.int64)

    tr_idx, va_idx, te_idx = train_val_test_groups(
        texts, labels, groups, args.seed, args.test_frac, args.val_frac
    )

    def take(idxs):
        return [texts[i] for i in idxs], labels[idxs]

    X_tr, y_tr = take(tr_idx)
    X_va, y_va = take(va_idx)
    X_te, y_te = take(te_idx)

    vec = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        min_df=2,
        sublinear_tf=True,
    )
    X_tr_v = vec.fit_transform(X_tr)
    X_va_v = vec.transform(X_va)
    X_te_v = vec.transform(X_te)

    clf = LogisticRegression(
        max_iter=2000,
        C=args.C,
        class_weight="balanced",
        solver="saga",
        n_jobs=-1,
        random_state=args.seed,
    )
    clf.fit(X_tr_v, y_tr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vec, f)
    with open(out_dir / "logreg.pkl", "wb") as f:
        pickle.dump(clf, f)

    def report(name, Xv, y):
        pred = clf.predict(Xv)
        proba = clf.predict_proba(Xv)[:, 1]
        rep = classification_report(
            y, pred, target_names=["human", "chatgpt"], digits=3
        )
        auc = float(roc_auc_score(y, proba))
        acc = float((pred == y).mean())
        return {"name": name, "accuracy": acc, "roc_auc": auc, "report": rep}

    y_te_arr = np.asarray(y_te, dtype=np.int64)
    proba_te = clf.predict_proba(X_te_v)[:, 1]
    pred_te = clf.predict(X_te_v)
    cm = confusion_matrix(y_te_arr, pred_te, labels=[0, 1])

    metrics = {
        "train": report("train", X_tr_v, y_tr),
        "val": report("val", X_va_v, y_va),
        "test": report("test", X_te_v, y_te),
        "n_examples": len(ds),
        "n_train_rows": int(len(tr_idx)),
        "n_val_rows": int(len(va_idx)),
        "n_test_rows": int(len(te_idx)),
        "subset": args.subset,
        "sources_filter": sorted(sources_allow) if sources_allow else None,
        "prepend_question": args.prepend_question,
        "test_confusion_matrix": cm.tolist(),
        "test_confusion_labels": ["human (0)", "chatgpt (1)"],
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    sources_te = [ds[int(i)]["source"] for i in te_idx]
    np.savez_compressed(
        out_dir / "test_eval.npz",
        y_true=y_te_arr,
        y_pred=pred_te.astype(np.int64),
        y_score=proba_te.astype(np.float64),
        source=np.asarray(sources_te, dtype=object),
    )

    print(json.dumps({k: metrics[k] for k in ("subset", "n_examples", "sources_filter")}, indent=2))
    print("--- val ---\n", metrics["val"]["report"])
    print("val auc", metrics["val"]["roc_auc"])
    print("--- test ---\n", metrics["test"]["report"])
    print("test auc", metrics["test"]["roc_auc"])
    print("saved:", out_dir.resolve())


if __name__ == "__main__":
    main()
