"""
Обучение всех моделей и подготовка результатов для панели.

  python train_all.py              # оба модуля
  python train_all.py --only network
  python train_all.py --only nlp

Результаты:
    models/*.joblib                    — обученные пайплайны
  outputs/network_events.csv         — события сетевого модуля (единая схема)
  outputs/http_events.csv            — события NLP-модуля (единая схема)
  outputs/reports/metrics.json       — метрики всех моделей
  outputs/reports/cm_*.png           — матрицы ошибок

Модели сохраняются через joblib (pickle). Загружайте ТОЛЬКО модели,
обученные вами: pickle-файл из чужого источника может выполнить код.
"""
from __future__ import annotations

import argparse
import json

import joblib

import config
from src.common.io_utils import save_events
from src.common.metrics import plot_confusion_matrix


def run_network() -> dict:
    from src.network.models import analyze_network, network_events, train_network_models
    from src.network.preprocess import clean_network_data, load_network_csv

    print("== Модуль 1: сетевой трафик ==")
    df = load_network_csv(config.NETWORK_DATA_PATH, require_label=True)
    df, quality = clean_network_data(df)
    print(f"Качество данных: {quality['duplicates']} дубликатов, "
          f"{quality['missing_total']} пропусков, {quality['infinite_values']} inf")

    models, metrics = train_network_models(df)
    joblib.dump(models["isoforest"], config.ISOFOREST_MODEL_PATH)
    joblib.dump(models["random_forest"], config.RF_MODEL_PATH)

    for key, title in [("isolation_forest", "Isolation Forest"), ("random_forest", "Random Forest")]:
        m = metrics[key]
        plot_confusion_matrix(m["confusion_matrix"], m["labels"], title,
                              config.REPORTS_DIR / f"cm_network_{key}.png")
        print(f"{title}: accuracy={m['accuracy']} f1_macro={m['f1_macro']} FPR={m['false_positive_rate']}")

    # События для панели строим по ВСЕМУ набору, но метрики выше — только по test
    analyzed = analyze_network(df, models["isoforest"], models["random_forest"])
    save_events(network_events(analyzed), config.NETWORK_EVENTS_PATH)
    metrics["data_quality"] = quality
    return metrics


def run_nlp() -> dict:
    from src.nlp.models import classify_requests, http_events, train_nlp_models
    from src.nlp.preprocess import clean_text_data, load_text_csv

    print("== Модуль 2: HTTP-запросы (NLP) ==")
    df = load_text_csv(config.TEXT_DATA_PATH, require_label=True)
    df, quality = clean_text_data(df)
    print(f"После очистки: {quality['rows_after_cleaning']} строк, "
          f"удалено дубликатов: {quality['duplicates_removed']}")

    models, metrics = train_nlp_models(df)
    joblib.dump(models["sgd_classifier"], config.NLP_MODEL_PATH)
    joblib.dump(models["complement_naive_bayes"], config.NLP_NB_MODEL_PATH)

    for key, title in [("sgd_classifier", "SGD Classifier"),
                       ("complement_naive_bayes", "Complement Naive Bayes")]:
        m = metrics[key]
        plot_confusion_matrix(m["confusion_matrix"], m["labels"], title,
                              config.REPORTS_DIR / f"cm_nlp_{key}.png")
        print(f"{title}: accuracy={m['accuracy']} f1_macro={m['f1_macro']} FPR={m['false_positive_rate']}")

    classified = classify_requests(models["sgd_classifier"], df[config.TEXT_COL])
    save_events(http_events(df, classified), config.HTTP_EVENTS_PATH)
    metrics["data_quality"] = quality
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["network", "nlp"])
    args = parser.parse_args()

    for d in (config.MODELS_DIR, config.OUTPUTS_DIR, config.REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    if config.METRICS_PATH.exists():
        all_metrics = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))

    if args.only in (None, "network"):
        if config.NETWORK_DATA_PATH.exists():
            all_metrics["network"] = run_network()
        else:
            print(f"Нет {config.NETWORK_DATA_PATH} — запустите scripts/generate_network_data.py")
    if args.only in (None, "nlp"):
        if config.TEXT_DATA_PATH.exists():
            all_metrics["nlp"] = run_nlp()
        else:
            print(f"Нет {config.TEXT_DATA_PATH} — подготовьте датасет: scripts/prepare_text_dataset.py")

    config.METRICS_PATH.write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("Метрики сохранены в", config.METRICS_PATH)


if __name__ == "__main__":
    main()
