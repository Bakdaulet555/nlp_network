"""
Модуль 4. Оценка качества моделей.

Единые функции метрик для обоих модулей, чтобы отчёт был сопоставимым:
Accuracy, Precision, Recall, F1 (macro и weighted), Confusion Matrix.
Для несбалансированных классов главными считаем macro-F1 и Recall по
каждому классу атак: высокая Accuracy может скрывать пропуски атак.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # без GUI: работает на сервере и в тестах
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_classifier(y_true, y_pred, labels: list[str]) -> dict:
    """Возвращает словарь метрик, пригодный для сохранения в JSON."""
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(p_macro), 4),
        "recall_macro": round(float(r_macro), 4),
        "f1_macro": round(float(f_macro), 4),
        "precision_weighted": round(float(p_w), 4),
        "recall_weighted": round(float(r_w), 4),
        "f1_weighted": round(float(f_w), 4),
        "per_class": {
            lbl: {k: round(float(v), 4) for k, v in report[lbl].items()}
            for lbl in labels if lbl in report
        },
        "labels": list(labels),
        "confusion_matrix": cm.tolist(),
        "support": int(len(y_true)),
    }


def false_positive_rate(y_true_binary, y_pred_binary) -> float:
    """FPR = FP / (FP + TN): доля нормальных объектов, ошибочно
    помеченных как подозрительные. Ключевая метрика «шума» для SOC."""
    y_true_binary = np.asarray(y_true_binary, dtype=bool)
    y_pred_binary = np.asarray(y_pred_binary, dtype=bool)
    fp = np.sum(~y_true_binary & y_pred_binary)
    tn = np.sum(~y_true_binary & ~y_pred_binary)
    return float(fp / (fp + tn)) if (fp + tn) else 0.0


def plot_confusion_matrix(cm, labels, title: str, path) -> None:
    """Сохраняет тепловую карту матрицы ошибок в PNG."""
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 2.5, 1.2 * len(labels) + 2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Прогноз модели")
    ax.set_ylabel("Истинная метка")
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
