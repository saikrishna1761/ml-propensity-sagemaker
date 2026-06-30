import pandas as pd
import pytest
from pandera.errors import SchemaError
from data.validate import customers_schema, orders_schema


@pytest.fixture
def valid_customers():
    return pd.DataFrame({
        "customer_id": list(range(9000)),
        "email_opens_30d": [5] * 9000,
        "age": [30] * 9000,
        "account_status": ["active"] * 9000,
    })


@pytest.fixture
def valid_orders():
    return pd.DataFrame({
        "order_id": list(range(1000)),
        "customer_id": list(range(1000)),
        "order_date": pd.to_datetime(["2026-05-01"] * 1000),
        "order_value": [100.0] * 1000,
        "order_status": ["completed"] * 1000,
    })


def test_valid_customers_passes(valid_customers):
    customers_schema.validate(valid_customers)


def test_valid_orders_passes(valid_orders):
    orders_schema.validate(valid_orders)


def test_customers_below_minimum_rows_fails(valid_customers):
    small_df = valid_customers.head(100)
    with pytest.raises(SchemaError):
        customers_schema.validate(small_df)


def test_customers_null_customer_id_fails(valid_customers):
    valid_customers.loc[0, "customer_id"] = None
    with pytest.raises(SchemaError):
        customers_schema.validate(valid_customers)


def test_customers_invalid_account_status_fails(valid_customers):
    valid_customers.loc[0, "account_status"] = "suspended"
    with pytest.raises(SchemaError):
        customers_schema.validate(valid_customers)


def test_customers_negative_age_fails(valid_customers):
    valid_customers.loc[0, "age"] = -1
    with pytest.raises(SchemaError):
        customers_schema.validate(valid_customers)


def test_orders_invalid_status_fails(valid_orders):
    valid_orders.loc[0, "order_status"] = "unknown"
    with pytest.raises(SchemaError):
        orders_schema.validate(valid_orders)


def test_orders_negative_value_fails(valid_orders):
    valid_orders.loc[0, "order_value"] = -50.0
    with pytest.raises(SchemaError):
        orders_schema.validate(valid_orders)
