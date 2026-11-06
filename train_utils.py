from __future__ import annotations

import gc
import json
import shutil
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments)
from transformers.utils import logging as hf_logging

from common import RESULTS_DIR, ROOT, SEED, evaluate_and_save, labels_for, save_predictions, y_for

MODEL_DIR = ROOT / "models"
TRAIN_LOG_DIR = RESULTS_DIR / "training"
TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class RunConfig:
    name: str
    max_len: int
    epochs: int
    per_device_batches: tuple[int, ...]
    model_id: str = "distilbert-base-uncased"
    batch_size: int = 16
    eval_batch_size: int = 32
    lr: float = 2e-5
    retrain: bool = False


def describe_device() -> None:
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} | "
              f"free GPU memory: {free / 1024**3:.1f} / {total / 1024**3:.1f} GB")
    else:
        print(f"torch {torch.__version__} | FIGYELEM: nincs GPU, a tanítás CPU-n sok órát venne igénybe.")


def load_tokenizer(cfg: RunConfig):
    return AutoTokenizer.from_pretrained(cfg.model_id)


def count_tokens(tokenizer, texts) -> np.ndarray:
    hf_logging.set_verbosity_error()
    lengths = np.array([len(ids) for ids in tokenizer(list(texts))["input_ids"]])
    hf_logging.set_verbosity_warning()
    return lengths


def load_info(run: str, task: str) -> dict:
    return json.loads((TRAIN_LOG_DIR / f"{run}_{task}.json").read_text(encoding="utf-8"))


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def encode(df: pd.DataFrame, task: str, tokenizer, max_len: int) -> Dataset:
    ds = Dataset.from_dict({"text": df["text"].tolist(), "label": y_for(df, task).tolist()})
    ds = ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=max_len),
                batched=True, remove_columns=["text"])
    return ds.map(lambda b: {"length": [len(x) for x in b["input_ids"]]}, batched=True)


def _compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.asarray(logits).argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "macro_f1": f1_score(labels, preds, average="macro")}


def _make_trainer(cfg: RunConfig, task: str, enc: dict, tokenizer, batch_size: int) -> Trainer:
    labels = labels_for(task)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_id, num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id={lab: i for i, lab in enumerate(labels)})
    args = TrainingArguments(
        output_dir=str(MODEL_DIR / f"checkpoints_{cfg.name}_{task}"),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=cfg.batch_size // batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        warmup_steps=0.1,
        weight_decay=0.01,
        train_sampling_strategy="group_by_length",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_only_model=True,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        fp16=False,
        report_to="none",
        seed=SEED,
    )
    return Trainer(model=model, args=args, train_dataset=enc["train"], eval_dataset=enc["val"],
                   data_collator=DataCollatorWithPadding(tokenizer), processing_class=tokenizer,
                   compute_metrics=_compute_metrics)


def train_or_load(cfg: RunConfig, task: str, enc: dict, tokenizer):
    out_dir = MODEL_DIR / f"{cfg.name}_{task}"
    info_path = TRAIN_LOG_DIR / f"{cfg.name}_{task}.json"

    if out_dir.exists() and info_path.exists() and not cfg.retrain:
        print(f"Mentett modell betöltése: {out_dir}")
        return AutoModelForSequenceClassification.from_pretrained(out_dir), load_info(cfg.name, task)

    for batch_size in cfg.per_device_batches:
        trainer = _make_trainer(cfg, task, enc, tokenizer, batch_size)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            t0 = time.perf_counter()
            trainer.train()
            secs = time.perf_counter() - t0
            break
        except torch.cuda.OutOfMemoryError:
            pass
        del trainer
        free_gpu()
        print(f"Elfogyott a GPU-memória batch {batch_size} mellett, újrapróbálás kisebb batch-csel")
    else:
        raise RuntimeError("Out of memory even at the smallest batch size - lower max_len and retry.")

    trainer.save_model(out_dir)
    info = {
        "model_id": cfg.model_id, "max_len": cfg.max_len, "learning_rate": cfg.lr, "epochs": cfg.epochs,
        "batch_size": cfg.batch_size, "per_device_batch_size": batch_size,
        "gradient_accumulation_steps": cfg.batch_size // batch_size, "precision": "fp32",
        "train_seconds": secs, "best_val_macro_f1": trainer.state.best_metric,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "log_history": trainer.state.log_history,
    }
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    shutil.rmtree(MODEL_DIR / f"checkpoints_{cfg.name}_{task}", ignore_errors=True)
    print(f"Tanítás kész: {secs / 60:.1f} perc, legjobb val macro-F1 = {trainer.state.best_metric:.4f}")
    return trainer.model, info


def predict_in_order(model, ds: Dataset, tokenizer, eval_batch_size: int) -> np.ndarray:
    # transformers 5.x: group_by_length also reorders predict() output, so predict with a default Trainer
    trainer = Trainer(model=model,
                      args=TrainingArguments(output_dir=str(MODEL_DIR / "tmp"), report_to="none",
                                             per_device_eval_batch_size=eval_batch_size),
                      data_collator=DataCollatorWithPadding(tokenizer), processing_class=tokenizer)
    out = trainer.predict(ds)
    assert np.array_equal(out.label_ids, np.asarray(ds["label"])), "prediction order does not match the test set"
    return out.predictions


def run_task(cfg: RunConfig, task: str, split: dict, tokenizer):
    enc = {name: encode(df, task, tokenizer, cfg.max_len) for name, df in split.items()}
    model, info = train_or_load(cfg, task, enc, tokenizer)

    logits = predict_in_order(model, enc["test"], tokenizer, cfg.eval_batch_size)
    proba = torch.softmax(torch.as_tensor(logits, dtype=torch.float32), dim=-1).numpy()
    test_df = split["test"]
    y_true, y_pred = y_for(test_df, task), proba.argmax(axis=1)
    evaluate_and_save(y_true, y_pred, cfg.name, task,
                      extra={k: v for k, v in info.items() if k != "log_history"})
    save_predictions(test_df, y_true, y_pred, proba, cfg.name, task)

    del model
    free_gpu()
    return proba, info


def evaluate_5to3(run: str, proba5: np.ndarray, test_df: pd.DataFrame) -> None:
    proba3 = np.column_stack([proba5[:, 0] + proba5[:, 1], proba5[:, 2], proba5[:, 3] + proba5[:, 4]])
    y_true, y_pred = y_for(test_df, "3class"), proba3.argmax(axis=1)
    evaluate_and_save(y_true, y_pred, f"{run}_5to3", "3class")
    save_predictions(test_df, y_true, y_pred, proba3, f"{run}_5to3", "3class")
