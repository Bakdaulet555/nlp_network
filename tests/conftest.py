"""Общие фикстуры. Тесты не зависят от больших датасетов и сети."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="session")
def network_df():
    from generate_network_data import make_network
    return make_network(3000, seed=3, start="2026-09-01 08:00")


@pytest.fixture(scope="session")
def placeholder_text_df():
    """Нейтральный датасет-«заглушка» для проверки механики NLP-пайплайна.

    Классы различаются искусственными маркерами, а не реальными строками
    атак: тест проверяет код (векторизацию, обучение, формат вывода),
    а не качество детектора. Качество проверяется на открытом датасете.
    """
    rng = np.random.default_rng(0)
    words = ["catalog", "item", "page", "search", "view", "order", "user", "blog"]
    markers = {"Normal": "", "SQL Injection": "zzqmarker_s", "XSS": "zzqmarker_x"}
    rows = []
    for label, marker in markers.items():
        for i in range(120):
            body = " ".join(rng.choice(words, 4))
            rows.append({"request": f"/{body.replace(' ', '/')}?id={i} {marker}".strip(),
                         "label": label})
    return pd.DataFrame(rows)
