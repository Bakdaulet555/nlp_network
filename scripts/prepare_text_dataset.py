"""
Подготовка текстового датасета HTTP-запросов из открытых источников.

Многие открытые наборы публикуются как ДВА бинарных файла:
  * набор по SQL Injection: текст запроса + метка 1 (атака) / 0 (норма);
  * набор по XSS:           текст + метка 1 (атака) / 0 (норма).
Скрипт объединяет их в один трёхклассовый CSV проекта:

    request,label            label ∈ {Normal, SQL Injection, XSS}

Правила разметки:
  * метка 1 из SQLi-файла → "SQL Injection";
  * метка 1 из XSS-файла  → "XSS";
  * метка 0 из любого файла → "Normal";
  * пустые строки и точные дубликаты удаляются;
  * если один и тот же текст встречается с разными метками — строка
    исключается (противоречивая разметка).

Пример (имена столбцов проверьте в скачанных файлах!):
  python scripts/prepare_text_dataset.py \
      --sqli raw/sqli.csv --sqli-text-col Query --sqli-label-col Label \
      --xss  raw/xss.csv  --xss-text-col Sentence --xss-label-col Label \
      --max-per-class 8000 --add-timestamps

Скрипт только читает CSV локально и ничего не отправляет в сеть.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.common.io_utils import read_csv_safely, require_columns  # noqa: E402

ATTACK_VALUES = {"1", "1.0", "true", "attack", "malicious"}


def _load_binary(path: str, text_col: str, label_col: str, attack_class: str) -> pd.DataFrame:
    df = read_csv_safely(path, max_mb=500, max_rows=2_000_000)
    require_columns(df, [text_col, label_col])
    out = pd.DataFrame({
        config.TEXT_COL: df[text_col].astype(str).str.strip(),
        config.TEXT_LABEL_COL: np.where(
            df[label_col].astype(str).str.strip().str.lower().isin(ATTACK_VALUES),
            attack_class, "Normal"),
    })
    return out[out[config.TEXT_COL] != ""]


def build_dataset(args) -> pd.DataFrame:
    parts = []
    if args.sqli:
        parts.append(_load_binary(args.sqli, args.sqli_text_col, args.sqli_label_col, "SQL Injection"))
    if args.xss:
        parts.append(_load_binary(args.xss, args.xss_text_col, args.xss_label_col, "XSS"))
    if args.normal:  # необязательный файл только с нормальными запросами
        n = read_csv_safely(args.normal, max_mb=200, max_rows=2_000_000)
        require_columns(n, [args.normal_text_col])
        parts.append(pd.DataFrame({config.TEXT_COL: n[args.normal_text_col].astype(str),
                                   config.TEXT_LABEL_COL: "Normal"}))
    if not parts:
        raise SystemExit("Укажите хотя бы один из --sqli / --xss / --normal")

    df = pd.concat(parts, ignore_index=True).drop_duplicates()
    conflicts = df.groupby(config.TEXT_COL)[config.TEXT_LABEL_COL].nunique()
    df = df[~df[config.TEXT_COL].isin(conflicts[conflicts > 1].index)]

    if args.max_per_class:
        df = (df.groupby(config.TEXT_LABEL_COL, group_keys=False)
                .apply(lambda g: g.sample(min(len(g), args.max_per_class), random_state=config.RANDOM_STATE)))
    df = df.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)

    if args.add_timestamps:  # условное время — только для демонстрации фильтров панели
        rng = np.random.default_rng(config.RANDOM_STATE)
        ts = pd.Timestamp("2026-09-01 08:00") + pd.to_timedelta(
            np.sort(rng.uniform(0, 8 * 3600, len(df))), unit="s")
        df[config.TEXT_TIME_COL] = ts.strftime("%Y-%m-%d %H:%M:%S")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sqli"); p.add_argument("--sqli-text-col", default="Query")
    p.add_argument("--sqli-label-col", default="Label")
    p.add_argument("--xss"); p.add_argument("--xss-text-col", default="Sentence")
    p.add_argument("--xss-label-col", default="Label")
    p.add_argument("--normal"); p.add_argument("--normal-text-col", default="request")
    p.add_argument("--max-per-class", type=int, default=0)
    p.add_argument("--add-timestamps", action="store_true")
    p.add_argument("--out", default=str(config.TEXT_DATA_PATH))
    args = p.parse_args()

    df = build_dataset(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Записано {len(df)} строк в {args.out}")
    print(df[config.TEXT_LABEL_COL].value_counts().to_string())


if __name__ == "__main__":
    main()
