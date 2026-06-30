import pandas as pd
import pytest
from datetime import datetime
from features.feature_engineering import compute_features


@pytest.fixture
def sample_customers():
    return pd.DataFrame({
        "customer_id": [1, 2, 3],
        "email_opens_30d": [5, 0, 10],
        "age": [30, 45, 25],
        "account_status": ["active", "active", "inactive"],
    })


@pytest.fixture
def sample_orders():
    return pd.DataFrame({
        "customer_id": [1, 1, 2, 3],
        "order_date": ["2026-05-01", "2026-05-20", "2026-04-15", "2026-05-25"],
        "order_value": [100.0, 200.0, 50.0, 75.0],
        "order_status": ["completed", "completed", "completed", "completed"],
    })


def test_feature_output_columns(sample_customers, sample_orders):
    snapshot_date = datetime(2026, 6, 1)
    result = compute_features(sample_customers, sample_orders, snapshot_date)
    expected_cols = {"customer_id", "recency_days", "orders_30d", "avg_order_value",
                     "email_opens_30d", "age", "account_status", "will_buy_7d", "snapshot_date"}
    assert expected_cols.issubset(set(result.columns))


def test_all_customers_in_output(sample_customers, sample_orders):
    snapshot_date = datetime(2026, 6, 1)
    result = compute_features(sample_customers, sample_orders, snapshot_date)
    assert len(result) == len(sample_customers)


def test_will_buy_7d_is_binary(sample_customers, sample_orders):
    snapshot_date = datetime(2026, 6, 1)
    result = compute_features(sample_customers, sample_orders, snapshot_date)
    assert set(result["will_buy_7d"].unique()).issubset({0, 1})


def test_no_negative_recency(sample_customers, sample_orders):
    snapshot_date = datetime(2026, 6, 1)
    result = compute_features(sample_customers, sample_orders, snapshot_date)
    assert (result["recency_days"] >= 0).all()


def test_orders_30d_non_negative(sample_customers, sample_orders):
    snapshot_date = datetime(2026, 6, 1)
    result = compute_features(sample_customers, sample_orders, snapshot_date)
    assert (result["orders_30d"] >= 0).all()
