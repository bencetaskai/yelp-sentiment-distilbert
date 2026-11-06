from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIG_DIR = RESULTS_DIR / "figures"
PRED_DIR = RESULTS_DIR / "predictions"
SPLIT_PATH = DATA_DIR / "yelp_sample_split.parquet"

for _d in (DATA_DIR, METRICS_DIR, FIG_DIR, PRED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
DATASET_ID = "Yelp/yelp_review_full"

LABELS_5 = ["1★", "2★", "3★", "4★", "5★"]
LABELS_3 = ["negatív", "semleges", "pozitív"]
_STAR_TO_3 = np.array([0, 0, 1, 2, 2])


def to_3class(y5) -> np.ndarray:
    return _STAR_TO_3[np.asarray(y5, dtype=int)]


def labels_for(task: str) -> list[str]:
    return LABELS_5 if task == "5class" else LABELS_3


_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = text.replace("\\n", " ").replace('\\"', '"')
    return _WS.sub(" ", text).strip()


def load_split() -> dict[str, pd.DataFrame]:
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"{SPLIT_PATH} not found - run 01_data_and_baseline.ipynb first.")
    df = pd.read_parquet(SPLIT_PATH)
    return {name: df[df["split"] == name].reset_index(drop=True)
            for name in ("train", "val", "test")}


def y_for(df: pd.DataFrame, task: str) -> np.ndarray:
    return df["label5" if task == "5class" else "label3"].to_numpy()


def compute_metrics(y_true, y_pred, labels: list[str]) -> dict:
    idx = list(range(len(labels)))
    p_m, r_m, f_m, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=idx, average="macro", zero_division=0)
    _, _, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=idx, average="weighted", zero_division=0)
    p_c, r_c, f_c, s_c = precision_recall_fscore_support(
        y_true, y_pred, labels=idx, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p_m),
        "macro_recall": float(r_m),
        "macro_f1": float(f_m),
        "weighted_f1": float(f_w),
        "per_class": {lab: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
                      for lab, p, r, f, s in zip(labels, p_c, r_c, f_c, s_c)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=idx).tolist(),
    }


def plot_confusion(cm, labels: list[str], title: str, path: Path | None = None):
    cm = np.asarray(cm)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    annot = np.array([[f"{norm[i, j]:.0%}\n({cm[i, j]})" for j in range(cm.shape[1])]
                      for i in range(cm.shape[0])])
    size = 1.1 * len(labels) + 1.5
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(norm, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1, cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax, annot_kws={"fontsize": 9})
    ax.set_xlabel("Előrejelzett osztály")
    ax.set_ylabel("Valódi osztály")
    ax.set_title(title)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig


def evaluate_and_save(y_true, y_pred, model: str, task: str, extra: dict | None = None) -> dict:
    labels = labels_for(task)
    m = compute_metrics(y_true, y_pred, labels)
    m.update({"model": model, "task": task, "n_test": int(len(y_true))})
    if extra:
        m.update(extra)
    (METRICS_DIR / f"{model}_{task}.json").write_text(json.dumps(m, indent=2, ensure_ascii=False),
                                                      encoding="utf-8")
    plot_confusion(m["confusion_matrix"], labels, f"{model} – {task}",
                   FIG_DIR / f"cm_{model}_{task}.png")
    print(f"{model} / {task}: accuracy={m['accuracy']:.4f}  macro-F1={m['macro_f1']:.4f}")
    return m


def save_predictions(test_df: pd.DataFrame, y_true, y_pred, proba, model: str, task: str) -> pd.DataFrame:
    labels = labels_for(task)
    out = pd.DataFrame({
        "text": test_df["text"].to_numpy(),
        "true": [labels[i] for i in y_true],
        "pred": [labels[i] for i in y_pred],
        "correct": np.asarray(y_true) == np.asarray(y_pred),
        "confidence": np.asarray(proba).max(axis=1),
    })
    for i, lab in enumerate(labels):
        out[f"p_{lab}"] = np.asarray(proba)[:, i]
    out.to_csv(PRED_DIR / f"{model}_{task}.csv", index=False, encoding="utf-8")
    return out


def results_table() -> pd.DataFrame:
    rows = []
    for f in sorted(METRICS_DIR.glob("*.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        rows.append({k: m[k] for k in ("model", "task", "accuracy", "macro_precision",
                                       "macro_recall", "macro_f1", "weighted_f1")})
    return pd.DataFrame(rows).sort_values(["task", "macro_f1"]).reset_index(drop=True)
