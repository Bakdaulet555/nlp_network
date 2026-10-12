"""
Модуль 1 (Студент 1). Модели для сетевого трафика.

* Isolation Forest — обнаружение аномалий без учителя. Обучается только на
  нормальном (BENIGN) трафике из train-выборки: модель учит «как выглядит
  норма» и помечает всё, что от неё сильно отличается.
* Random Forest — классификация с учителем по меткам Label
  (BENIGN, DoS, PortScan, ...). Работает лучше на известных типах атак,
  но не видит новые типы, которых не было в обучении.

Аномалия ≠ атака: необычный, но легитимный трафик (резервное копирование,
обновления, сканирование администратором) тоже будет помечен.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from src.common.io_utils import make_events_frame
from src.common.metrics import evaluate_classifier, false_positive_rate
from src.network.preprocess import parse_timestamps, to_binary_attack

FEATURES = config.NETWORK_FEATURES


# --------------------------------------------------------------------------
# Построение пайплайнов
# --------------------------------------------------------------------------
def build_isoforest_pipeline(contamination: float = config.ISOFOREST_CONTAMINATION) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),  # для деревьев не обязателен, но не вредит
        ("model", IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def build_rf_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",  # компенсация дисбаланса классов
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


# --------------------------------------------------------------------------
# Обучение
# --------------------------------------------------------------------------
def tune_isoforest_contamination(X_train: pd.DataFrame, y_train_bin: np.ndarray,
                                 grid=(0.01, 0.02, 0.03, 0.05, 0.075, 0.10,
                                       0.125, 0.15),
                                 max_validation_fpr: float = 0.15) -> float:
    """Подбирает contamination на ВАЛИДАЦИОННОЙ части train (тест не трогаем).

    Обучаем на нормальных записях из fit-части, оцениваем F1 по атакам на
    val-части. Кандидаты с FPR выше допустимого лимита отбрасываются.
    Запас 0.15 на validation помогает удержать итоговый test FPR ниже 0.20.
    Если меток нет — возвращаем значение из config.
    """
    if y_train_bin.sum() == 0 or y_train_bin.sum() == len(y_train_bin):
        return config.ISOFOREST_CONTAMINATION
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train_bin, test_size=0.25, stratify=y_train_bin,
        random_state=config.RANDOM_STATE,
    )
    best_c, best_f1 = config.ISOFOREST_CONTAMINATION, -1.0
    for c in grid:
        pipe = build_isoforest_pipeline(c).fit(X_fit[y_fit == 0])
        pred = (pipe.predict(X_val) == -1).astype(int)
        fpr = false_positive_rate(y_val, pred)
        f1 = f1_score(y_val, pred, zero_division=0)
        if fpr <= max_validation_fpr and f1 > best_f1:
            best_c, best_f1 = c, f1
    return best_c


def train_network_models(df: pd.DataFrame) -> tuple[dict, dict]:
    """Полный цикл обучения сетевого модуля.

    1. Разбиение train/test ДО любых обучаемых преобразований.
    2. Isolation Forest: подбор contamination на валидации → обучение на BENIGN.
    3. Random Forest: GridSearchCV только на train.
    4. Оценка на test.
    Возвращает (models, metrics).
    """
    label_col = config.NETWORK_LABEL_COL
    data = df.copy()

    # Очень редкие классы (< 10 примеров) нельзя надёжно стратифицировать
    # и оценить — объединяем их в «Other attack»
    counts = data[label_col].value_counts()
    rare = counts[counts < 10].index
    if len(rare):
        data.loc[data[label_col].isin(rare), label_col] = "Other attack"

    X = data[FEATURES]
    y = data[label_col].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    y_train_bin = to_binary_attack(y_train)
    y_test_bin = to_binary_attack(y_test)

    # ---- Isolation Forest ------------------------------------------------
    contamination = tune_isoforest_contamination(X_train, y_train_bin)
    iso = build_isoforest_pipeline(contamination).fit(X_train[y_train_bin == 0])
    iso_pred = (iso.predict(X_test) == -1).astype(int)
    iso_metrics = evaluate_classifier(
        np.where(y_test_bin == 1, "Attack", "Normal"),
        np.where(iso_pred == 1, "Attack", "Normal"),
        labels=["Normal", "Attack"],
    )
    iso_metrics["false_positive_rate"] = round(false_positive_rate(y_test_bin, iso_pred), 4)
    iso_metrics["contamination"] = contamination
    # какие типы атак Isolation Forest замечает лучше/хуже
    iso_metrics["detection_rate_by_label"] = {
        lbl: round(float(iso_pred[(y_test == lbl).to_numpy()].mean()), 4)
        for lbl in sorted(y_test.unique())
    }

    # ---- Random Forest ---------------------------------------------------
    grid = GridSearchCV(
        build_rf_pipeline(),
        param_grid={"model__max_depth": [None, 20], "model__min_samples_leaf": [1, 3]},
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE),
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    rf = grid.best_estimator_
    rf_labels = sorted(y.unique(), key=lambda s: (s != config.NETWORK_BENIGN_LABEL, s))
    rf_metrics = evaluate_classifier(y_test, rf.predict(X_test), labels=rf_labels)
    rf_metrics["best_params"] = grid.best_params_
    rf_metrics["cv_f1_macro"] = round(float(grid.best_score_), 4)
    rf_metrics["false_positive_rate"] = round(
        false_positive_rate(y_test_bin, to_binary_attack(pd.Series(rf.predict(X_test)))), 4
    )
    importances = rf.named_steps["model"].feature_importances_
    rf_metrics["feature_importance"] = {
        f: round(float(v), 4)
        for f, v in sorted(zip(FEATURES, importances), key=lambda t: -t[1])
    }

    models = {"isoforest": iso, "random_forest": rf}
    metrics = {
        "dataset": {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "class_distribution": {k: int(v) for k, v in y.value_counts().items()},
        },
        "isolation_forest": iso_metrics,
        "random_forest": rf_metrics,
    }
    return models, metrics


# --------------------------------------------------------------------------
# Анализ новых данных
# --------------------------------------------------------------------------
def analyze_network(df: pd.DataFrame, iso: Pipeline, rf: Pipeline | None = None) -> pd.DataFrame:
    """Применяет обученные модели к очищенным данным.

    Возвращает исходные признаки + столбцы:
      if_anomaly (bool), if_score (0..1, выше — аномальнее),
      rf_prediction, rf_confidence (если RF передан).
    """
    X = df[FEATURES]
    out = df.copy()
    out["if_anomaly"] = iso.predict(X) == -1
    # score_samples возвращает -s, где s ∈ (0, 1] — оценка аномальности
    # из статьи Liu et al. Поэтому -score_samples уже лежит в (0, 1].
    out["if_score"] = np.clip(-iso.score_samples(X), 0, 1)
    if rf is not None:
        proba = rf.predict_proba(X)
        classes = rf.classes_
        out["rf_prediction"] = classes[proba.argmax(axis=1)]
        out["rf_confidence"] = proba.max(axis=1)
    return out


def network_events(analyzed: pd.DataFrame) -> pd.DataFrame:
    """Переводит результат анализа в единую схему событий для панели.

    prediction = Anomaly/Normal (Isolation Forest).
    needs_review = модели расходятся во мнении (IF vs RF) — такие записи
    аналитику стоит посмотреть вручную в первую очередь.
    """
    n = len(analyzed)
    has_rf = "rf_prediction" in analyzed.columns
    if has_rf:
        rf_attack = analyzed["rf_prediction"] != config.NETWORK_BENIGN_LABEL
        needs_review = analyzed["if_anomaly"] != rf_attack
        rf_text = ("RF: " + analyzed["rf_prediction"].astype(str)
                   + " (" + analyzed["rf_confidence"].round(2).astype(str) + ")")
    else:
        needs_review = pd.Series(False, index=analyzed.index)
        rf_text = pd.Series("RF: —", index=analyzed.index)

    details = (
        "port " + analyzed["Destination Port"].fillna(-1).astype(int).astype(str)
        + " | fwd_pkts " + analyzed["Total Fwd Packets"].fillna(0).astype(int).astype(str)
        + " | " + rf_text
    )
    truth = (analyzed[config.NETWORK_LABEL_COL].astype(str)
             if config.NETWORK_LABEL_COL in analyzed.columns else [""] * n)

    return make_events_frame(
        event_id=[f"net-{i:06d}" for i in range(1, n + 1)],
        timestamp=parse_timestamps(analyzed).to_numpy(),
        source_module=["network"] * n,
        model_name=["IsolationForest"] * n,
        prediction=np.where(analyzed["if_anomaly"], "Anomaly", "Normal"),
        score=analyzed["if_score"].to_numpy(),
        is_alert=analyzed["if_anomaly"].to_numpy(),
        needs_review=needs_review.to_numpy(),
        ground_truth=np.asarray(truth),
        details=details.to_numpy(),
        status=[config.PREDICTION_STATUS] * n,
    )
