"""Score arbitrary English text with a saved TF-IDF + LogisticRegression bundle."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="train_outputs")
    p.add_argument("text", nargs="?", help="If omitted, read stdin.")
    args = p.parse_args()
    d = Path(args.model_dir)
    with open(d / "tfidf_vectorizer.pkl", "rb") as f:
        vec = pickle.load(f)
    with open(d / "logreg.pkl", "rb") as f:
        clf = pickle.load(f)

    text = args.text
    if text is None:
        import sys

        text = sys.stdin.read()
    text = text.strip()
    if not text:
        raise SystemExit("empty text")
    X = vec.transform([text])
    p_human, p_gpt = clf.predict_proba(X)[0]
    label = int(clf.predict(X)[0])
    print(f"P(human)={p_human:.4f} P(chatgpt)={p_gpt:.4f} predicted_label={label} (1=chatgpt)")


if __name__ == "__main__":
    main()
