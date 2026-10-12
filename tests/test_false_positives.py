"""
Проверка ложных срабатываний обученной NLP-модели на «трудных» нормальных
запросах (data/fp_benign_requests.csv): слова select/union/script/delete,
апострофы, знаки < > = --.

Тест запускается только после обучения на реальном датасете
(models/nlp_sgd_classifier.joblib), иначе пропускается.
Порог FPR ≤ 25% — учебный; фактическое значение заносится в отчёт.
"""
import joblib
import pandas as pd
import pytest

import config
from src.nlp.models import classify_requests

FP_FILE = config.DATA_DIR / "fp_benign_requests.csv"


@pytest.mark.skipif(not config.NLP_MODEL_PATH.exists(), reason="NLP-модель ещё не обучена")
def test_hard_benign_false_positive_rate():
    model = joblib.load(config.NLP_MODEL_PATH)
    df = pd.read_csv(FP_FILE)
    res = classify_requests(model, df["request"])
    fpr = (res["prediction"] != "Normal").mean()
    print("\nЛожные срабатывания:\n", res.loc[res["prediction"] != "Normal",
                                            ["request", "prediction", "confidence"]].to_string())
    print(f"FPR на трудных нормальных запросах: {fpr:.2%}")
    assert fpr <= 0.25
