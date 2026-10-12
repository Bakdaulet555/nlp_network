import numpy as np
import pandas as pd

import config
from src.common.io_utils import validate_events
from src.nlp.models import classify_requests, explain_request, http_events, train_nlp_models
from src.nlp.preprocess import clean_text_data, normalize_label, normalize_request


def test_normalize_decodes_url_and_html():
    assert normalize_request("Hello%20World") == "hello world"
    assert normalize_request("a%2520b") == "a b"            # двойное кодирование
    assert normalize_request("&lt;b&gt;bold&lt;/b&gt;") == "<b>bold</b>"
    assert normalize_request("id=12345") == "id=0"
    assert normalize_request(None) == ""
    assert len(normalize_request("x" * 10_000)) == config.MAX_REQUEST_LENGTH


def test_special_characters_preserved():
    s = normalize_request("q=it's;a<b>(c)--d/*e*/")
    for ch in "';<>()-/*":
        assert ch in s


def test_label_aliases():
    assert normalize_label("sqli") == "SQL Injection"
    assert normalize_label("benign") == "Normal"
    assert normalize_label(" XSS ") == "XSS"


def test_dedup_after_normalization():
    df = pd.DataFrame({"request": ["/page?id=1", "/page?id=2", "/page%3Fid%3D3", ""],
                       "label": ["Normal"] * 4})
    clean, rep = clean_text_data(df)
    assert len(clean) == 1 and rep["duplicates_removed"] == 2 and rep["empty_removed"] == 1


def test_conflicting_labels_after_normalization_are_removed():
    df = pd.DataFrame({
        "request": ["/item?id=1", "/ITEM?id=2", "/safe"],
        "label": ["Normal", "SQL Injection", "Normal"],
    })
    clean, rep = clean_text_data(df)
    assert clean["request"].tolist() == ["/safe"]
    assert rep["conflicts_removed"] == 2


def test_pipeline_mechanics(placeholder_text_df):
    clean, _ = clean_text_data(placeholder_text_df)
    models, metrics = train_nlp_models(clean)
    for name in ("sgd_classifier", "complement_naive_bayes"):
        m = metrics[name]
        assert m["labels"] == config.TEXT_CLASSES
        assert np.array(m["confusion_matrix"]).shape == (3, 3)
        assert m["f1_macro"] > 0.9   # маркеры тривиально разделимы

    main_model = models["sgd_classifier"]
    res = classify_requests(main_model, ["/catalog/item?id=5", "/catalog zzqmarker_x"])
    prob_cols = [f"prob_{c}" for c in main_model.classes_]
    assert np.allclose(res[prob_cols].sum(axis=1), 1, atol=1e-3)
    assert res.loc[1, "prediction"] == "XSS"
    assert not explain_request(main_model, "/catalog zzqmarker_x").empty

    ev = http_events(pd.DataFrame({"request": res["request"]}), res)
    validate_events(ev)
    assert ev["is_alert"].tolist() == (res["prediction"] != "Normal").tolist()


def test_review_flag_threshold(placeholder_text_df):
    clean, _ = clean_text_data(placeholder_text_df)
    models, _ = train_nlp_models(clean)
    res = classify_requests(models["sgd_classifier"], ["completely unrelated text ###"])
    assert res.loc[0, "needs_review"] == (res.loc[0, "confidence"] < config.NLP_REVIEW_THRESHOLD)


def test_empty_classification_batch(placeholder_text_df):
    clean, _ = clean_text_data(placeholder_text_df)
    models, _ = train_nlp_models(clean)
    res = classify_requests(models["sgd_classifier"], [])
    assert res.empty
    assert {"request", "prediction", "confidence", "needs_review"} <= set(res.columns)
