import json
import numpy as np
import pytest
from unittest.mock import MagicMock
from serving.inference import input_fn, predict_fn, output_fn


@pytest.fixture
def sample_payload():
    return json.dumps({
        "recency_days": 10,
        "orders_30d": 3,
        "avg_order_value": 150.0,
        "email_opens_30d": 5,
        "age": 35,
    })


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.2, 0.8]])
    return model


def test_input_fn_returns_dataframe(sample_payload):
    import pandas as pd
    result = input_fn(sample_payload, "application/json")
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["recency_days", "orders_30d", "avg_order_value", "email_opens_30d", "age"]


def test_input_fn_unsupported_content_type_raises(sample_payload):
    with pytest.raises(ValueError):
        input_fn(sample_payload, "text/csv")


def test_predict_fn_returns_scores(sample_payload, mock_model):
    input_data = input_fn(sample_payload, "application/json")
    scores = predict_fn(input_data, mock_model)
    assert len(scores) == 1
    assert 0.0 <= scores[0] <= 1.0


def test_output_fn_returns_json(mock_model, sample_payload):
    input_data = input_fn(sample_payload, "application/json")
    scores = predict_fn(input_data, mock_model)
    body, content_type = output_fn(scores)
    assert content_type == "application/json"
    result = json.loads(body)
    assert "propensity_score" in result[0]


def test_output_fn_score_is_rounded(mock_model, sample_payload):
    input_data = input_fn(sample_payload, "application/json")
    scores = predict_fn(input_data, mock_model)
    body, _ = output_fn(scores)
    result = json.loads(body)
    score = result[0]["propensity_score"]
    assert score == round(score, 4)
