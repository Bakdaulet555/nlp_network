"""
Генератор учебного (синтетического) сетевого датасета.

Создаёт в data/:
  network_traffic.csv      — обучающий датасет в формате подмножества столбцов
                             CIC-IDS2017 (с пропусками, inf и дубликатами);
  network_traffic_new.csv  — «новые» данные для загрузки в панель.

Используется, пока студенты не скачали CIC-IDS2017 (файлы большие), и для
быстрых тестов. Данные полностью синтетические.

Запуск:  python scripts/generate_network_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

COMMON_PORTS = [80, 443, 443, 443, 53, 53, 22, 8080, 25, 123, 3389, 993]
MAX_CLASS_RATIO = 1.20


def _limit_class_imbalance(df: pd.DataFrame, label_col: str, seed: int) -> pd.DataFrame:
    """Ограничивает каждый класс величиной 120% от самого малого класса."""
    counts = df[label_col].value_counts()
    limit = int(np.floor(counts.min() * MAX_CLASS_RATIO))
    parts = [
        group.sample(min(len(group), limit), random_state=seed)
        for _, group in df.groupby(label_col, sort=False)
    ]
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed)


def _flows(rng: np.random.Generator, n: int, kind: str) -> pd.DataFrame:
    """Генерирует n потоков заданного типа с правдоподобными распределениями."""
    if kind == "BENIGN":
        port = rng.choice(COMMON_PORTS, n)
        duration = rng.lognormal(12, 2.2, n)                 # микросекунды
        fwd = rng.poisson(9, n) + 1
        bwd = rng.poisson(8, n)
        fwd_mean = rng.lognormal(4.2, 1.0, n)
        bwd_mean = rng.lognormal(5.5, 1.3, n)
        syn = rng.binomial(1, 0.05, n)
        # 4% «необычной нормы» (резервные копии, обновления) —
        # источник ложных срабатываний для детектора аномалий
        heavy = rng.random(n) < 0.04
        fwd = np.where(heavy, fwd * 40, fwd)
        bwd = np.where(heavy, bwd * 60, bwd)
        duration = np.where(heavy, duration * 30, duration)
    elif kind == "DoS Hulk":
        port = np.full(n, 80)
        duration = rng.lognormal(10.5, 1.5, n)
        fwd = rng.poisson(6, n) + 2
        bwd = rng.poisson(1, n)
        fwd_mean = rng.normal(55, 25, n).clip(0)
        bwd_mean = rng.normal(1500, 900, n).clip(0)
        syn = rng.binomial(1, 0.2, n)
    elif kind == "DDoS":
        port = np.full(n, 80)
        duration = rng.lognormal(9.5, 1.8, n)
        fwd = rng.poisson(3, n) + 1
        bwd = rng.poisson(4, n)
        fwd_mean = rng.normal(20, 10, n).clip(0)
        bwd_mean = rng.normal(1800, 900, n).clip(0)
        syn = rng.binomial(1, 0.1, n)
    elif kind == "PortScan":
        port = rng.integers(1, 65535, n)
        duration = rng.lognormal(4, 1.5, n)
        fwd = rng.integers(1, 3, n)
        bwd = rng.integers(0, 2, n)
        fwd_mean = rng.normal(2, 3, n).clip(0)
        bwd_mean = rng.normal(3, 4, n).clip(0)
        syn = np.ones(n, dtype=int)
    elif kind == "SSH-Patator":
        port = np.full(n, 22)
        duration = rng.lognormal(14, 0.6, n)
        fwd = rng.poisson(22, n) + 5
        bwd = rng.poisson(33, n) + 5
        fwd_mean = rng.normal(90, 20, n).clip(0)
        bwd_mean = rng.normal(100, 25, n).clip(0)
        syn = rng.binomial(1, 0.3, n)
    else:
        raise ValueError(kind)

    fwd_len = fwd * fwd_mean
    bwd_len = bwd * bwd_mean
    total_pkts = fwd + bwd
    seconds = duration / 1e6
    with np.errstate(divide="ignore", invalid="ignore"):
        bytes_s = (fwd_len + bwd_len) / seconds
        pkts_s = total_pkts / seconds
        iat = duration / np.maximum(total_pkts - 1, 1)
        avg = (fwd_len + bwd_len) / np.maximum(total_pkts, 1)

    return pd.DataFrame({
        "Destination Port": port,
        "Flow Duration": duration.round(0),
        "Total Fwd Packets": fwd,
        "Total Backward Packets": bwd,
        "Total Length of Fwd Packets": fwd_len.round(1),
        "Total Length of Bwd Packets": bwd_len.round(1),
        "Fwd Packet Length Mean": fwd_mean.round(2),
        "Bwd Packet Length Mean": bwd_mean.round(2),
        "Flow Bytes/s": bytes_s.round(2),
        "Flow Packets/s": pkts_s.round(2),
        "Flow IAT Mean": iat.round(2),
        "SYN Flag Count": syn,
        "ACK Flag Count": rng.binomial(1, 0.6, n),
        "Average Packet Size": avg.round(2),
        "Label": kind,
    })


def make_network(n_total: int, seed: int, start: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Учебный набор сбалансирован: каждый класс получает одинаковую долю.
    shares = {"BENIGN": 0.20, "DoS Hulk": 0.20, "PortScan": 0.20,
              "DDoS": 0.20, "SSH-Patator": 0.20}
    parts = [_flows(rng, int(n_total * s), k) for k, s in shares.items()]
    df = pd.concat(parts, ignore_index=True)
    df = _limit_class_imbalance(df, "Label", seed)

    # «Грязь», как в реальном CIC-IDS2017: inf, пропуски, дубликаты
    idx = rng.choice(len(df), size=int(0.01 * len(df)), replace=False)
    df.iloc[idx, df.columns.get_loc("Flow Bytes/s")] = np.inf
    idx = rng.choice(len(df), size=int(0.01 * len(df)), replace=False)
    df.iloc[idx, df.columns.get_loc("Flow IAT Mean")] = np.nan
    ts = pd.Timestamp(start) + pd.to_timedelta(
        np.sort(rng.uniform(0, 8 * 3600, len(df))), unit="s")
    df.insert(0, "Timestamp", ts.strftime("%Y-%m-%d %H:%M:%S"))
    # 2% полных дубликатов (вместе со временем) — их должна удалить очистка
    df = pd.concat([df, df.sample(frac=0.02, random_state=seed)], ignore_index=True)
    # Повторно применяем лимит после добавления дубликатов, чтобы даже сырой CSV
    # буквально соблюдал максимальное отношение классов 1.20.
    df = _limit_class_imbalance(df, "Label", seed)
    return df.sort_values("Timestamp").reset_index(drop=True)


def main() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    # 4 000 исходных строк = по 800 строк каждого класса до добавления грязи.
    make_network(4_000, seed=1, start="2026-09-01 08:00").to_csv(config.NETWORK_DATA_PATH, index=False)
    make_network(1_000, seed=7, start="2026-09-02 08:00").to_csv(
        config.DATA_DIR / "network_traffic_new.csv", index=False)
    print("Сетевые данные записаны в", config.DATA_DIR)
    print("Максимальный размер класса ограничен 120% от минимального.")


if __name__ == "__main__":
    main()
