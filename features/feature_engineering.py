import boto3
import pandas as pd
import yaml
import io
from datetime import datetime, timedelta

CONFIG_PATH = "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def read_parquet_from_s3(bucket, key, region):
    s3 = boto3.client("s3", region_name=region)
    response = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(response["Body"].read()))


def write_parquet_to_s3(df, bucket, key, region):
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, engine="pyarrow", compression="snappy")
    buffer.seek(0)
    s3 = boto3.client("s3", region_name=region)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    print(f"  Written: s3://{bucket}/{key} ({len(df)} rows)")


def compute_features(customers_df, orders_df, snapshot_date):
    # temporal split: features use data before cutoff, label uses cutoff → snapshot_date
    cutoff_date = snapshot_date - timedelta(days=7)
    label_window_end = snapshot_date

    completed = orders_df[orders_df["order_status"] == "completed"].copy()
    completed["order_date"] = pd.to_datetime(completed["order_date"])

    # orders used for feature computation (before cutoff)
    hist = completed[completed["order_date"] < cutoff_date]

    # recency: days since last completed order before cutoff
    last_order = (
        hist.groupby("customer_id")["order_date"]
        .max()
        .rename("last_order_date")
        .reset_index()
    )
    last_order["recency_days"] = (cutoff_date - last_order["last_order_date"]).dt.days

    # frequency: completed orders in 30 days before cutoff
    freq_window = cutoff_date - timedelta(days=30)
    freq = (
        hist[hist["order_date"] >= freq_window]
        .groupby("customer_id")
        .size()
        .rename("orders_30d")
        .reset_index()
    )

    # monetary: average order value of all completed orders before cutoff
    monetary = (
        hist.groupby("customer_id")["order_value"]
        .mean()
        .rename("avg_order_value")
        .reset_index()
    )

    # will_buy_7d label: completed order in [cutoff_date, snapshot_date)
    future = completed[
        (completed["order_date"] >= cutoff_date) &
        (completed["order_date"] < label_window_end)
    ]
    buyers = set(future["customer_id"].unique())

    # assemble feature table
    features = customers_df[["customer_id", "email_opens_30d", "age", "account_status"]].copy()
    features = features.merge(last_order[["customer_id", "recency_days"]], on="customer_id", how="left")
    features = features.merge(freq, on="customer_id", how="left")
    features = features.merge(monetary, on="customer_id", how="left")

    features["recency_days"] = features["recency_days"].fillna(365)
    features["orders_30d"] = features["orders_30d"].fillna(0).astype(int)
    features["avg_order_value"] = features["avg_order_value"].fillna(0.0)
    features["will_buy_7d"] = features["customer_id"].apply(lambda x: 1 if x in buyers else 0)
    features["snapshot_date"] = snapshot_date.strftime("%Y-%m-%d")

    return features


def main():
    config = load_config()
    region = config["aws"]["region"]
    bucket = config["aws"]["s3_bucket"]
    raw_prefix = config["s3"]["raw_prefix"]
    features_prefix = config["s3"]["features_prefix"]
    snapshot_date = datetime.strptime(config["snapshot_date"], "%Y-%m-%d")
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

    print(f"Customers: {len(customers_df)} | Orders: {len(orders_df)}")

    print("Computing features...")
    features_df = compute_features(customers_df, orders_df, snapshot_date)

    label_rate = features_df["will_buy_7d"].mean()
    print(f"Label distribution: {features_df['will_buy_7d'].value_counts().to_dict()}")
    print(f"Positive rate: {label_rate:.2%}")

    print("Writing features to S3...")
    write_parquet_to_s3(
        features_df,
        bucket,
        f"{features_prefix}/{partition}/features.parquet",
        region,
    )

    print("Feature engineering complete.")


if __name__ == "__main__":
    main()
