"""
Общие функции ввода-вывода: безопасная загрузка CSV, проверка структуры,
безопасная выгрузка и единая схема событий.

Принципы безопасности:
* читаем только CSV (никакого pickle/eval/exec для пользовательских файлов);
* ограничиваем размер файла и число строк;
* проверяем обязательные столбцы до анализа;
* при выгрузке экранируем значения, начинающиеся с = + - @ (CSV/Formula
  injection при открытии в Excel).
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import config


class DataValidationError(ValueError):
    """Ошибка структуры или качества входных данных."""


# --------------------------------------------------------------------------
# Загрузка
# --------------------------------------------------------------------------
def read_csv_safely(
    source,
    max_mb: float = config.MAX_UPLOAD_MB,
    max_rows: int = config.MAX_UPLOAD_ROWS,
    filename: str | None = None,
) -> pd.DataFrame:
    """Читает CSV из пути, bytes или file-like объекта с ограничениями.

    source   — путь (str/Path), bytes или объект с методом read() (Streamlit
               UploadedFile).
    filename — имя файла для проверки расширения (для загруженных файлов).
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        name = filename or path.name
        size_mb = path.stat().st_size / 1024 / 1024
        raw = None
    else:
        raw = source if isinstance(source, bytes) else source.read()
        name = filename or getattr(source, "name", "upload.csv")
        size_mb = len(raw) / 1024 / 1024

    if Path(name).suffix.lower() not in config.ALLOWED_EXTENSIONS:
        raise DataValidationError(f"Допустимы только CSV-файлы, получен: {name}")
    if size_mb > max_mb:
        raise DataValidationError(
            f"Файл {size_mb:.1f} МБ превышает лимит {max_mb} МБ"
        )

    try:
        if raw is None:
            df = pd.read_csv(path, nrows=max_rows + 1, low_memory=False,
                             encoding_errors="replace")
        else:
            df = pd.read_csv(io.BytesIO(raw), nrows=max_rows + 1,
                             low_memory=False, encoding_errors="replace")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DataValidationError(f"Не удалось разобрать CSV: {exc}") from exc

    if len(df) > max_rows:
        raise DataValidationError(f"Слишком много строк (> {max_rows})")
    if df.empty:
        raise DataValidationError("Файл не содержит данных")

    # CIC-IDS2017: у имён столбцов бывают ведущие пробелы
    df.columns = [str(c).strip() for c in df.columns]
    return df


def require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Проверяет наличие обязательных столбцов."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(
            "В файле нет обязательных столбцов: " + ", ".join(missing)
        )


def data_quality_report(df: pd.DataFrame) -> dict:
    """Краткий отчёт о качестве данных: пропуски, дубликаты, бесконечности."""
    numeric = df.select_dtypes(include=[np.number])
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicates": int(df.duplicated().sum()),
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
        "infinite_values": int(np.isinf(numeric.to_numpy()).sum()) if not numeric.empty else 0,
    }


# --------------------------------------------------------------------------
# Выгрузка
# --------------------------------------------------------------------------
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def sanitize_for_csv_export(df: pd.DataFrame) -> pd.DataFrame:
    """Защита от CSV/Formula injection: текстовые ячейки, начинающиеся
    с символов формулы, предваряются апострофом. Числа не трогаем."""
    out = df.copy()
    for col in out.columns:
        # object (pandas < 3) или строковый dtype (pandas ≥ 3)
        if out[col].dtype == object or pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].map(
                lambda v: "'" + v if isinstance(v, str) and v.startswith(_FORMULA_PREFIXES) else v
            )
    return out


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return sanitize_for_csv_export(df).to_csv(index=False).encode("utf-8-sig")


# --------------------------------------------------------------------------
# Единая схема событий (контракт между модулями)
# --------------------------------------------------------------------------
def make_events_frame(**columns) -> pd.DataFrame:
    """Собирает DataFrame событий строго в порядке config.EVENT_COLUMNS.

    Каждый модуль обязан передать все столбцы; отсутствующий столбец —
    ошибка интеграции, и её лучше поймать сразу.
    """
    missing = [c for c in config.EVENT_COLUMNS if c not in columns]
    if missing:
        raise ValueError(f"Не заполнены поля события: {missing}")
    df = pd.DataFrame({c: columns[c] for c in config.EVENT_COLUMNS})
    df["score"] = df["score"].astype(float).round(4)
    df["is_alert"] = df["is_alert"].astype(bool)
    df["needs_review"] = df["needs_review"].astype(bool)
    return df


def validate_events(df: pd.DataFrame) -> None:
    """Проверка, что файл событий соответствует контракту."""
    require_columns(df, config.EVENT_COLUMNS)
    bad_modules = set(df["source_module"].unique()) - {"network", "http"}
    if bad_modules:
        raise DataValidationError(f"Неизвестные source_module: {bad_modules}")
    if not df["score"].between(0, 1).all():
        raise DataValidationError("score должен быть в диапазоне [0, 1]")


def save_events(df: pd.DataFrame, path: Path) -> None:
    validate_events(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_events(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=config.EVENT_COLUMNS)
    df = pd.read_csv(path, dtype={"ground_truth": str, "timestamp": str})
    df["is_alert"] = df["is_alert"].astype(bool)
    df["needs_review"] = df["needs_review"].astype(bool)
    return df
