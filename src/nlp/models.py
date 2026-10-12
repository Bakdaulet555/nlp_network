"""
Модуль 2 (Студент 2). Классификация HTTP-запросов: Normal / SQL Injection / XSS.

Признаки: TF-IDF по символьным n-граммам (2–5 символов, analyzer="char_wb").
Почему символы, а не слова: сигнатуры атак — это фрагменты вроде "' or",
"--", "<scr", "onerror=", "union sel", которые плохо выделяются словарной
токенизацией, зато устойчиво ловятся символьными n-граммами, в том числе
при изменённом регистре и вставке пробелов.

Модели:
* SGDClassifier с log-loss — основная: линейная вероятностная модель,
  быстро обучающаяся на больших разреженных матрицах;
* Complement Naive Bayes — baseline, устойчивый к дисбалансу классов.

Векторизатор находится ВНУТРИ Pipeline, поэтому словарь и IDF обучаются
только на train (и на train-фолдах внутри GridSearchCV).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

import config
from src.common.io_utils import make_events_frame
from src.common.metrics import evaluate_classifier, false_positive_rate
from src.nlp.preprocess import normalize_request

CLASSES = config.TEXT_CLASSES


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=normalize_request,  # декодирование + нормализация
        lowercase=False,                 # уже сделано в normalize_request
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        max_features=50_000,
        sublinear_tf=True,
    )


def build_sgd_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", build_vectorizer()),
        ("model", SGDClassifier(
            loss="log_loss",
            max_iter=2000,
            tol=1e-4,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        )),
    ])


def build_complement_nb_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", build_vectorizer()),
        ("model", ComplementNB()),
    ])


def train_nlp_models(df: pd.DataFrame) -> tuple[dict, dict]:
    """Обучение и сравнение SGD Classifier и Complement Naive Bayes.

    Гиперпараметры подбираются кросс-валидацией на train; test используется
    один раз — для итоговой оценки.
    """
    missing = {config.TEXT_COL, config.TEXT_LABEL_COL} - set(df.columns)
    if missing:
        raise ValueError(f"Для обучения отсутствуют столбцы: {sorted(missing)}")

    X = df[config.TEXT_COL].astype(str)
    y = df[config.TEXT_LABEL_COL].astype(str)
    unknown = sorted(set(y.unique()) - set(CLASSES))
    if unknown:
        raise ValueError(f"Неизвестные метки классов: {unknown}")
    counts = y.value_counts()
    absent = [label for label in CLASSES if label not in counts]
    if absent:
        raise ValueError(f"В датасете отсутствуют обязательные классы: {absent}")
    too_small = {label: int(counts[label]) for label in CLASSES if counts[label] < 4}
    if too_small:
        raise ValueError(
            "Для разбиения и 3-fold CV нужно не менее 4 строк каждого класса; "
            f"недостаточно данных: {too_small}"
        )
    evaluation = "stratified_random_split"
    if config.TEXT_SPLIT_COL in df.columns:
        split = df[config.TEXT_SPLIT_COL].astype(str).str.strip().str.lower()
        unknown_splits = sorted(set(split.unique()) - {"train", "test"})
        if unknown_splits:
            raise ValueError(f"Неизвестные значения split: {unknown_splits}")
        train_mask = split == "train"
        test_mask = split == "test"
        if not train_mask.any() or not test_mask.any():
            raise ValueError("Столбец split должен содержать строки train и test")
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        evaluation = "predefined_group_holdout"
        for label in CLASSES:
            if label not in set(y_train) or label not in set(y_test):
                raise ValueError(f"Класс {label!r} должен присутствовать и в train, и в test")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
        )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)

    candidates = {
        "sgd_classifier": (
            build_sgd_pipeline(),
            {"model__alpha": [1e-5, 1e-4, 1e-3]},
        ),
        "complement_naive_bayes": (
            build_complement_nb_pipeline(),
            {"model__alpha": [0.01, 0.1, 1.0]},
        ),
    }
    models, metrics = {}, {}
    for name, (pipe, grid) in candidates.items():
        search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        y_pred = best.predict(X_test)
        m = evaluate_classifier(y_test, y_pred, labels=CLASSES)
        m["best_params"] = search.best_params_
        m["cv_f1_macro"] = round(float(search.best_score_), 4)
        # FPR: доля Normal-запросов, ошибочно помеченных как атака
        m["false_positive_rate"] = round(
            false_positive_rate(y_test != "Normal", y_pred != "Normal"), 4
        )
        models[name], metrics[name] = best, m

    metrics["dataset"] = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "evaluation": evaluation,
        "class_distribution": {k: int(v) for k, v in y.value_counts().items()},
        "train_distribution": {k: int(v) for k, v in y_train.value_counts().items()},
        "test_distribution": {k: int(v) for k, v in y_test.value_counts().items()},
    }
    return models, metrics


def classify_requests(model: Pipeline, requests) -> pd.DataFrame:
    """Классифицирует список запросов.

    Возвращает: request, prediction, confidence, prob_<класс>, needs_review.
    confidence — вероятность выбранного класса, а НЕ вероятность того,
    что это реальная атака в вашей системе.
    """
    requests = pd.Series(list(requests), dtype=str)
    if requests.empty:
        columns = ["request", "prediction", "confidence"]
        columns += [f"prob_{c}" for c in model.classes_]
        columns += ["needs_review"]
        return pd.DataFrame(columns=columns)
    proba = model.predict_proba(requests)
    classes = list(model.classes_)
    out = pd.DataFrame({"request": requests})
    out["prediction"] = np.array(classes)[proba.argmax(axis=1)]
    out["confidence"] = proba.max(axis=1).round(4)
    for i, c in enumerate(classes):
        out[f"prob_{c}"] = proba[:, i].round(4)
    out["needs_review"] = out["confidence"] < config.NLP_REVIEW_THRESHOLD
    return out


def explain_request(model: Pipeline, request: str, top_k: int = 8) -> pd.DataFrame:
    """Какие n-граммы сильнее всего повлияли на прогноз линейной модели.

    Вклад n-граммы = TF-IDF вес в запросе × коэффициент модели для класса.
    """
    clf = model.named_steps["model"]
    if not hasattr(clf, "coef_"):
        return pd.DataFrame(columns=["ngram", "contribution"])
    vec = model.named_steps["tfidf"]
    x = vec.transform([request])
    cls_idx = int(np.argmax(clf.predict_proba(x)[0]))
    coef = clf.coef_[cls_idx]
    idx = x.nonzero()[1]
    contrib = x.toarray()[0, idx] * coef[idx]
    names = vec.get_feature_names_out()[idx]
    df = pd.DataFrame({"ngram": names, "contribution": contrib.round(4)})
    return df.sort_values("contribution", ascending=False).head(top_k).reset_index(drop=True)


def http_events(
    source_df: pd.DataFrame,
    classified: pd.DataFrame,
    model_name: str = "SGDClassifier",
) -> pd.DataFrame:
    """Перевод результатов в единую схему событий."""
    n = len(classified)
    if len(source_df) != n:
        raise ValueError(
            "Число исходных запросов и результатов классификации не совпадает: "
            f"{len(source_df)} != {n}"
        )
    ts = (pd.to_datetime(source_df[config.TEXT_TIME_COL], errors="coerce", format="mixed")
          .dt.strftime("%Y-%m-%dT%H:%M:%S").fillna("").to_numpy()
          if config.TEXT_TIME_COL in source_df.columns else [""] * n)
    truth = (source_df[config.TEXT_LABEL_COL].astype(str).to_numpy()
             if config.TEXT_LABEL_COL in source_df.columns else [""] * n)
    return make_events_frame(
        event_id=[f"http-{i:06d}" for i in range(1, n + 1)],
        timestamp=ts,
        source_module=["http"] * n,
        model_name=[model_name] * n,
        prediction=classified["prediction"].to_numpy(),
        score=classified["confidence"].to_numpy(),
        is_alert=(classified["prediction"] != "Normal").to_numpy(),
        needs_review=classified["needs_review"].to_numpy(),
        ground_truth=truth,
        details=classified["request"].str.slice(0, 160).to_numpy(),
        status=[config.PREDICTION_STATUS] * n,
    )
