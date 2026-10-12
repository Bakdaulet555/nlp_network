"""Интеграция: события обоих модулей объединяются в одну таблицу панели."""
import pandas as pd

import config
from src.common.io_utils import load_events, save_events, validate_events
from src.network.models import analyze_network, network_events, train_network_models
from src.network.preprocess import clean_network_data
from src.nlp.models import classify_requests, http_events, train_nlp_models
from src.nlp.preprocess import clean_text_data


def test_events_from_both_modules_merge(tmp_path, network_df, placeholder_text_df):
    clean_net, _ = clean_network_data(network_df)
    net_models, _ = train_network_models(clean_net)
    net_ev = network_events(analyze_network(clean_net.head(100), net_models["isoforest"],
                                            net_models["random_forest"]))

    clean_txt, _ = clean_text_data(placeholder_text_df)
    nlp_models, _ = train_nlp_models(clean_txt)
    res = classify_requests(nlp_models["sgd_classifier"], clean_txt["request"].head(30))
    http_ev = http_events(clean_txt.head(30), res)

    # сохранение/загрузка через файлы — так модули передают результаты панели
    save_events(net_ev, tmp_path / "n.csv"); save_events(http_ev, tmp_path / "h.csv")
    merged = pd.concat([load_events(tmp_path / "n.csv"), load_events(tmp_path / "h.csv")],
                       ignore_index=True)
    validate_events(merged)
    assert list(merged.columns) == config.EVENT_COLUMNS
    assert merged["event_id"].is_unique
    assert set(merged["source_module"]) == {"network", "http"}
    assert (merged["status"] == config.PREDICTION_STATUS).all()
