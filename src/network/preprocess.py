"""
Модуль 1 (Студент 1). Загрузка и предобработка сетевого трафика.

Ожидаемый формат — подмножество столбцов CIC-IDS2017 (MachineLearningCVE /
GeneratedLabelledFlows). Подходит и оригинальный файл CIC-IDS2017, и
синтетический учебный CSV из scripts/generate_sample_data.py.

Важно про утечку данных:
* здесь выполняются только операции, НЕ обучаемые на данных: приведение
  типов, замена inf → NaN, удаление полных дубликатов;
* заполнение пропусков (медианой) и масштабирование выполняются внутри
  sklearn Pipeline и обучаются только на train-выборке.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.common.io_utils import data_quality_report, read_csv_safely, require_columns


def load_network_csv(source, filename: str | None = None, require_label: bool = False) -> pd.DataFrame:
    """Загружает CSV и проверяет наличие признаков (и метки, если нужна)."""
    df = read_csv_safely(source, filename=filename)
    required = list(config.NETWORK_FEATURES)
    if require_label:
        required.append(config.NETWORK_LABEL_COL)
    require_columns(df, required)
    return df


def clean_network_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Приводит признаки к числам, заменяет ±inf на NaN, удаляет дубликаты.

    Возвращает очищенный DataFrame и отчёт о качестве ДО очистки.
    """
    report = data_quality_report(df)
    out = df.copy()

    for col in config.NETWORK_FEATURES:
        # нечисловые значения (например, 'Infinity' строкой) → NaN
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out[config.NETWORK_FEATURES] = out[config.NETWORK_FEATURES].replace([np.inf, -np.inf], np.nan)

    # строки, где пропущены ВСЕ признаки, бесполезны
    out = out.dropna(subset=config.NETWORK_FEATURES, how="all")

    if config.NETWORK_LABEL_COL in out.columns:
        out[config.NETWORK_LABEL_COL] = out[config.NETWORK_LABEL_COL].astype(str).str.strip()
        # в CIC-IDS2017 встречаются нестандартные символы в названиях атак
        out[config.NETWORK_LABEL_COL] = out[config.NETWORK_LABEL_COL].str.replace("�", "-", regex=False)

    before = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = int(before - len(out))
    report["rows_after_cleaning"] = int(len(out))
    return out, report


def to_binary_attack(labels: pd.Series) -> np.ndarray:
    """BENIGN → 0, любая атака → 1 (для оценки Isolation Forest)."""
    return (labels.astype(str) != config.NETWORK_BENIGN_LABEL).to_numpy().astype(int)


def parse_timestamps(df: pd.DataFrame) -> pd.Series:
    """Возвращает время в ISO-формате или пустые строки, если столбца нет."""
    if config.NETWORK_TIME_COL not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    ts = pd.to_datetime(df[config.NETWORK_TIME_COL], errors="coerce", format="mixed")
    return ts.dt.strftime("%Y-%m-%dT%H:%M:%S").fillna("")
