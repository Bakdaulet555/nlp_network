"""
Модуль 2 (Студент 2). Предобработка текстовых HTTP-запросов.

Атакующие часто маскируют полезную нагрузку кодированием:
  %27%20OR%201%3D1        → ' or 1=1
  &lt;script&gt;           → <script>
  %253Cscript%253E        → (двойное URL-кодирование) <script>
Поэтому перед векторизацией запрос декодируется и нормализуется.

Специальные символы (' " < > ; = -- /* ( )) НЕ удаляются: именно они
несут основной сигнал атаки. Удалять их, как в «обычном» NLP, нельзя.

Функция normalize_request не обучается на данных, поэтому её можно
безопасно применять до разбиения выборки — утечки нет.
"""
from __future__ import annotations

import html
import re
from urllib.parse import unquote_plus

import pandas as pd

import config
from src.common.io_utils import read_csv_safely, require_columns

_WS_RE = re.compile(r"\s+")
_DIGITS_RE = re.compile(r"\d+")
_LABEL_ALIASES = {
    "normal": "Normal", "benign": "Normal", "0": "Normal",
    "sql injection": "SQL Injection", "sqli": "SQL Injection", "sql": "SQL Injection",
    "xss": "XSS", "cross-site scripting": "XSS",
}


def normalize_request(text) -> str:
    """Декодирует и нормализует один запрос.

    Шаги: обрезка длины → многократное URL-декодирование (до 3 раз, пока
    строка меняется) → HTML-unescape → нижний регистр → числа в «0» →
    схлопывание пробелов.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)[: config.MAX_REQUEST_LENGTH]
    for _ in range(3):
        decoded = unquote_plus(s)
        if decoded == s:
            break
        s = decoded
    s = html.unescape(s)
    s = s.replace("\x00", " ").lower()
    s = _DIGITS_RE.sub("0", s)       # 1=1 и 2=2 становятся одинаковым шаблоном 0=0
    s = _WS_RE.sub(" ", s).strip()
    return s


def normalize_label(value) -> str:
    """Приводит метку к одному из config.TEXT_CLASSES."""
    key = str(value).strip().lower()
    return _LABEL_ALIASES.get(key, str(value).strip())


def load_text_csv(source, filename: str | None = None, require_label: bool = False) -> pd.DataFrame:
    """Загружает CSV с запросами, проверяет структуру и метки."""
    df = read_csv_safely(source, filename=filename)
    required = [config.TEXT_COL] + ([config.TEXT_LABEL_COL] if require_label else [])
    require_columns(df, required)
    df[config.TEXT_COL] = df[config.TEXT_COL].fillna("").astype(str)
    if config.TEXT_LABEL_COL in df.columns:
        df[config.TEXT_LABEL_COL] = df[config.TEXT_LABEL_COL].map(normalize_label)
        unknown = set(df[config.TEXT_LABEL_COL].unique()) - set(config.TEXT_CLASSES)
        if unknown and require_label:
            raise ValueError(f"Неизвестные метки классов: {sorted(unknown)}")
    return df


def clean_text_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Удаляет пустые запросы и дубликаты ПОСЛЕ нормализации.

    Дубликаты по нормализованному тексту удаляются до разбиения: иначе
    один и тот же запрос (например, отличающийся только числом) попадёт
    и в train, и в test, и метрики будут завышены.
    """
    out = df.copy()
    out["_norm"] = out[config.TEXT_COL].map(normalize_request)
    report = {"rows": int(len(out)), "empty_removed": int((out["_norm"] == "").sum())}
    out = out[out["_norm"] != ""]

    # Один и тот же запрос может быть записан по-разному (регистр, URL-кодирование,
    # числа), но после нормализации стать одинаковым. Если у таких вариантов разные
    # метки, нельзя произвольно оставлять первую строку: это шум в разметке и утечка
    # между train/test. Удаляем всю конфликтующую группу до дедупликации.
    report["conflicts_removed"] = 0
    if config.TEXT_LABEL_COL in out.columns:
        label_counts = out.groupby("_norm")[config.TEXT_LABEL_COL].nunique(dropna=False)
        conflicting = set(label_counts[label_counts > 1].index)
        if conflicting:
            mask = out["_norm"].isin(conflicting)
            report["conflicts_removed"] = int(mask.sum())
            out = out[~mask]

    before = len(out)
    out = out.drop_duplicates(subset="_norm").reset_index(drop=True)
    report["duplicates_removed"] = int(before - len(out))
    report["rows_after_cleaning"] = int(len(out))
    if config.TEXT_LABEL_COL in out.columns:
        report["class_distribution"] = {
            k: int(v) for k, v in out[config.TEXT_LABEL_COL].value_counts().items()
        }
    return out.drop(columns="_norm"), report
