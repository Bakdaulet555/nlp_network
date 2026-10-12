import pandas as pd
import pytest

import config
from src.common.io_utils import (
    DataValidationError, make_events_frame, read_csv_safely, require_columns,
    sanitize_for_csv_export, validate_events,
)


def test_rejects_non_csv_extension():
    with pytest.raises(DataValidationError):
        read_csv_safely(b"a,b\n1,2\n", filename="data.xlsx")


def test_rejects_too_large_file():
    big = b"a\n" + b"1\n" * 300_000
    with pytest.raises(DataValidationError):
        read_csv_safely(big, filename="x.csv", max_mb=0.1)


def test_rejects_too_many_rows():
    with pytest.raises(DataValidationError):
        read_csv_safely(b"a\n" + b"1\n" * 50, filename="x.csv", max_rows=10)


def test_strips_column_names():
    df = read_csv_safely(b" Flow Duration, Label\n1,BENIGN\n", filename="x.csv")
    assert list(df.columns) == ["Flow Duration", "Label"]


def test_missing_columns_reported():
    with pytest.raises(DataValidationError, match="request"):
        require_columns(pd.DataFrame({"x": [1]}), ["request"])


def test_csv_formula_injection_escaped():
    df = pd.DataFrame({"t": ["=1+1", "@cmd", "normal", "-x"], "n": [-1, 2, 3, 4]})
    out = sanitize_for_csv_export(df)
    assert out["t"].tolist() == ["'=1+1", "'@cmd", "normal", "'-x"]
    assert out["n"].tolist() == [-1, 2, 3, 4]  # числа не изменяются


def _event(**over):
    base = dict(event_id=["e1"], timestamp=[""], source_module=["http"], model_name=["m"],
                prediction=["Normal"], score=[0.9], is_alert=[False], needs_review=[False],
                ground_truth=[""], details=["d"], status=[config.PREDICTION_STATUS])
    base.update(over)
    return base


def test_event_contract_complete():
    df = make_events_frame(**_event())
    assert list(df.columns) == config.EVENT_COLUMNS
    validate_events(df)


def test_event_contract_missing_field():
    data = _event(); data.pop("status")
    with pytest.raises(ValueError):
        make_events_frame(**data)


def test_event_contract_bad_score():
    with pytest.raises(DataValidationError):
        validate_events(make_events_frame(**_event(score=[1.5])))
