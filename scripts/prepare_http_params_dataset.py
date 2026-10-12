"""Готовит единый сбалансированный NLP-набор без внешнего датасета.

Используются Morzeux/HttpParamsDataset и XSS-строки fuzzdb. Формируется по
900 строк каждого класса. Train/test разделяются по группам похожих шаблонов,
чтобы близкие варианты одной строки не попадали в обе части.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.nlp.preprocess import clean_text_data, normalize_request  # noqa: E402

LABEL_MAP = {"norm": "Normal", "sqli": "SQL Injection", "xss": "XSS"}


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    clean, _ = clean_text_data(frame[[config.TEXT_COL, config.TEXT_LABEL_COL]])
    return clean


def _load_http_params(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    required = {"payload", "attack_type"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"В HttpParamsDataset отсутствуют столбцы: {sorted(missing)}")
    raw = raw[raw["attack_type"].isin(LABEL_MAP)].copy()
    frame = pd.DataFrame({
        config.TEXT_COL: raw["payload"].fillna("").astype(str),
        config.TEXT_LABEL_COL: raw["attack_type"].map(LABEL_MAP),
    })
    clean = _clean_frame(frame)
    clean["source"] = "HttpParamsDataset"
    return clean


def _load_fuzzdb_xss(directory: Path) -> pd.DataFrame:
    payloads: list[str] = []
    for name in ["xss-other.txt", "xss-rsnake.txt", "XSSPolyglot.txt"]:
        path = directory / name
        if path.exists():
            payloads.extend(
                line.strip()
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )
    contextual = directory / "JHADDIX_XSS_WITH_CONTEXT.doc.txt"
    if contextual.exists():
        for line in contextual.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("Exploit String:"):
                payloads.append(line.split(":", 1)[1].strip())
    frame = pd.DataFrame({config.TEXT_COL: payloads, config.TEXT_LABEL_COL: "XSS"})
    clean = _clean_frame(frame)
    clean["source"] = "fuzzdb"
    return clean


def _template_group(text: str) -> str:
    """Грубый отпечаток шаблона для защиты от похожих train/test строк."""
    value = normalize_request(text)
    value = re.sub(r"0x[a-f0-9]+", "0x0", value)
    value = re.sub(r"(['\"])[^'\"]{1,80}\1", r"\1value\1", value)
    value = re.sub(r"\b[a-z_]{10,}\b", "word", value)
    return value[:48]


def _sample_each(df: pd.DataFrame, count: int) -> pd.DataFrame:
    parts = []
    for label in config.TEXT_CLASSES:
        group = df[df[config.TEXT_LABEL_COL] == label]
        if len(group) < count:
            raise ValueError(f"Для класса {label} доступно {len(group)}, требуется {count}")
        parts.append(group.sample(count, random_state=config.RANDOM_STATE))
    return pd.concat(parts, ignore_index=True)


def prepare(http_params: Path, fuzzdb_xss: Path, per_class: int) -> pd.DataFrame:
    base = _load_http_params(http_params)
    fuzz_xss = _load_fuzzdb_xss(fuzzdb_xss)
    pool = pd.concat([base, fuzz_xss], ignore_index=True)
    pool["_norm"] = pool[config.TEXT_COL].map(normalize_request)
    pool = pool.drop_duplicates(subset="_norm").drop(columns="_norm")
    result = _sample_each(pool, per_class)

    groups = (
        result[config.TEXT_LABEL_COL].astype(str)
        + "|"
        + result[config.TEXT_COL].map(_template_group)
    )
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )
    train_idx, test_idx = next(
        splitter.split(result[config.TEXT_COL], result[config.TEXT_LABEL_COL], groups)
    )
    result[config.TEXT_SPLIT_COL] = "train"
    result.loc[test_idx, config.TEXT_SPLIT_COL] = "test"
    return result.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--http-params", type=Path,
                        default=Path("raw/http_params_dataset/payload_full.csv"))
    parser.add_argument("--fuzzdb-xss", type=Path, default=Path("raw/fuzzdb/attack/xss"))
    parser.add_argument("--out", type=Path, default=config.TEXT_DATA_PATH)
    parser.add_argument("--per-class", type=int, default=900)
    args = parser.parse_args()

    df = prepare(args.http_params, args.fuzzdb_xss, args.per_class)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Записано {len(df)} строк в {args.out}")
    print(pd.crosstab(df[config.TEXT_LABEL_COL], df[config.TEXT_SPLIT_COL]).to_string())
    print("Источники:")
    print(df["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
