import boto3
import yaml
import io
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check

CONFIG_PATH = "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def read_parquet_from_s3(bucket, key, region):
    s3 = boto3.client("s3", region_name=region)
    response = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(response["Body"].read()))


customers_schema = DataFrameSchema(
    columns={
        "customer_id": Column(int, nullable=False, unique=True),
        "email_opens_30d": Column(int, nullable=False, checks=Check.greater_than_or_equal_to(0)),
        "age": Column(int, nullable=False, checks=[Check.greater_than(0), Check.less_than(120)]),
        "account_status": Column(str, nullable=False, checks=Check.isin(["active", "inactive"])),
    },
    checks=Check(lambda df: len(df) >= 9000, error="customers table has fewer than 9000 rows"),
)

orders_schema = DataFrameSchema(
    columns={
        "order_id": Column(int, nullable=False, unique=True),
        "customer_id": Column(int, nullable=False),
        "order_date": Column("datetime64[ns]", nullable=False),
        "order_value": Column(float, nullable=False, checks=Check.greater_than(0)),
        "order_status": Column(str, nullable=False, checks=Check.isin(["completed", "pending", "cancelled", "returned"])),
    },
    checks=Check(lambda df: len(df) >= 1000, error="orders table has fewer than 1000 rows"),
)


def main():
    config = load_config()
    region = config["aws"]["region"]
    bucket = config["aws"]["s3_bucket"]
    raw_prefix = config["s3"]["raw_prefix"]
    partition = f"snapshot_date={config['snapshot_date']}"

    print("Reading customers from S3...")
    customers_df = read_parquet_from_s3(
        bucket,
        f"{raw_prefix}/customers/{partition}/customers.parquet",
        region,
    )

    print("Reading orders from S3...")
    orders_df = read_parquet_from_s3(
        bucket,
        f"{raw_prefix}/orders/{partition}/orders.parquet",
        region,
    )

    print(f"Validating customers ({len(customers_df)} rows)...")
    customers_schema.validate(customers_df)
    print("  customers: OK")

    print(f"Validating orders ({len(orders_df)} rows)...")
    orders_schema.validate(orders_df)
    print("  orders: OK")

    print("Validation passed.")


if __name__ == "__main__":
    main()
