import numpy as np

import config
from src.common.io_utils import validate_events
from src.network.models import analyze_network, network_events, train_network_models
from src.network.preprocess import clean_network_data


def test_cleaning_removes_inf_and_duplicates(network_df):
    clean, report = clean_network_data(network_df)
    assert report["duplicates"] > 0 and report["duplicates_removed"] == report["duplicates"]
    assert not np.isinf(clean[config.NETWORK_FEATURES].to_numpy()).any()
    assert not clean.duplicated().any()


def test_training_and_events(network_df):
    clean, _ = clean_network_data(network_df)
    models, metrics = train_network_models(clean)
    rf = metrics["random_forest"]
    # RF на учебных данных должен заметно превосходить случайное угадывание
    assert rf["f1_macro"] > 0.7
    # Isolation Forest: FPR на норме ограничен, PortScan хорошо виден как аномалия
    iso = metrics["isolation_forest"]
    assert iso["false_positive_rate"] < 0.3
    assert iso["detection_rate_by_label"]["PortScan"] > 0.8

    analyzed = analyze_network(clean.head(200), models["isoforest"], models["random_forest"])
    assert analyzed["if_score"].between(0, 1).all()
    ev = network_events(analyzed)
    validate_events(ev)
    assert len(ev) == 200 and set(ev["prediction"]) <= {"Anomaly", "Normal"}
    assert ev["timestamp"].str.len().gt(0).all()


def test_works_without_label_and_timestamp(network_df):
    clean, _ = clean_network_data(network_df)
    models, _ = train_network_models(clean)
    unlabeled = clean.drop(columns=["Label", "Timestamp"]).head(50)
    ev = network_events(analyze_network(unlabeled, models["isoforest"]))
    validate_events(ev)
    assert (ev["ground_truth"] == "").all()
