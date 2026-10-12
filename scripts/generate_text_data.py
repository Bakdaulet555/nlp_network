"""Создаёт безопасный демонстрационный датасет для NLP-модуля.

Набор нужен для локального запуска проекта без скачивания внешних данных.
Строки являются текстовыми примерами и никогда не исполняются. Для итогового
исследования их следует заменить открытыми датасетами через
``prepare_text_dataset.py``.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def _normal_requests() -> list[str]:
    topics = [
        "python book", "select size guide", "javascript tutorial", "drop shipping",
        "union jack flag", "script writing", "delete account help", "insert coin arcade",
        "update windows drivers", "html table example", "alert bracelet", "tables and chairs",
    ]
    qualifiers = ["beginner", "review", "online", "local", "sale", "manual", "course", "shop"]
    rows = [f"/search?q={quote_plus(topic + ' ' + qualifier)}" for topic, qualifier in itertools.product(topics, qualifiers)]

    names = ["o'reilly", "o'connor", "d'angelo", "smith", "lee", "garcia"]
    tabs = ["settings", "orders", "profile", "security", "history", "favorites"]
    rows += [f"/profile?user={quote_plus(name)}&tab={tab}" for name, tab in itertools.product(names, tabs)]

    phrases = [
        "great product -- would buy again", "size equals perfect fit", "price less than budget",
        "rock and roll records", "one equals one math identity", "best laptop review",
    ]
    authors = ["alice", "boris", "carol", "daniyar", "elena", "farid"]
    rows += [f"comment={quote_plus(phrase)}&author={author}" for phrase, author in itertools.product(phrases, authors)]
    return rows


def _sqli_requests() -> list[str]:
    prefixes = ["/item?id=", "/login?user=", "/api/search?q=", "username=", "filter="]
    payloads = [
        "' OR 'a'='a' --", "' UNION SELECT username,password FROM users --",
        "1; DROP TABLE audit_log; --", "admin'/*", "' OR EXISTS(SELECT name FROM users) --",
        "' AND SLEEP(5) --", "' UNION ALL SELECT email,token FROM accounts --",
        "') OR ('x'='x", "' HAVING COUNT(*) > 0 --", "' ORDER BY 1 --",
    ]
    suffixes = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    rows = []
    for prefix, payload, suffix in itertools.product(prefixes, payloads, suffixes):
        value = payload if len(rows) % 2 == 0 else quote_plus(payload)
        rows.append(f"{prefix}{value}&ref={suffix}")
    return rows


def _xss_requests() -> list[str]:
    prefixes = ["/search?q=", "/feedback?text=", "comment=", "name=", "/redirect?next="]
    payloads = [
        "<script>alert('x')</script>", "<img src=x onerror=alert('x')>",
        "<svg onload=alert('x')>", "javascript:alert('x')", "<body onload=alert('x')>",
        "<iframe srcdoc='<script>alert(1)</script>'>", "<a href=javascript:alert(1)>open</a>",
        "<input autofocus onfocus=alert(1)>", "<details open ontoggle=alert(1)>",
        "</textarea><script>alert(1)</script>",
    ]
    suffixes = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    rows = []
    for prefix, payload, suffix in itertools.product(prefixes, payloads, suffixes):
        value = payload if len(rows) % 2 == 0 else quote_plus(payload)
        rows.append(f"{prefix}{value}&ref={suffix}")
    return rows


def make_text_dataset() -> pd.DataFrame:
    frames = []
    for label, requests in [
        ("Normal", _normal_requests()),
        ("SQL Injection", _sqli_requests()),
        ("XSS", _xss_requests()),
    ]:
        frames.append(pd.DataFrame({config.TEXT_COL: requests, config.TEXT_LABEL_COL: label}))
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=config.TEXT_COL)
    df = df.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)
    start = pd.Timestamp("2026-09-01 08:00:00")
    df[config.TEXT_TIME_COL] = (start + pd.to_timedelta(df.index * 20, unit="s")).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    return df


def main() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = make_text_dataset()
    df.to_csv(config.TEXT_DATA_PATH, index=False)
    print(f"Записано {len(df)} строк в {config.TEXT_DATA_PATH}")
    print(df[config.TEXT_LABEL_COL].value_counts().to_string())


if __name__ == "__main__":
    main()
