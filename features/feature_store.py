import boto3
import yaml
import time
import pandas as pd
import io

CONFIG_PATH = "config.yaml"
FEATURE_GROUP_NAME = "propensity-features"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def create_feature_group(region, role, bucket):
    """Run this once locally to create the feature group. Not called by Airflow."""
    sm_client = boto3.client("sagemaker", region_name=region)

    try:
        sm_client.describe_feature_group(FeatureGroupName=FEATURE_GROUP_NAME)
        print(f"Feature group '{FEATURE_GROUP_NAME}' already exists.")
        return
    except sm_client.exceptions.ResourceNotFound:
        pass

    print(f"Creating feature group '{FEATURE_GROUP_NAME}'...")
    sm_client.create_feature_group(
        FeatureGroupName=FEATURE_GROUP_NAME,
        RecordIdentifierFeatureName="customer_id",
        EventTimeFeatureName="event_time",
        FeatureDefinitions=[
            {"FeatureName": "customer_id", "FeatureType": "Integral"},
            {"FeatureName": "recency_days", "FeatureType": "Fractional"},
            {"FeatureName": "orders_30d", "FeatureType": "Integral"},
            {"FeatureName": "avg_order_value", "FeatureType": "Fractional"},
            {"FeatureName": "email_opens_30d", "FeatureType": "Integral"},
            {"FeatureName": "age", "FeatureType": "Integral"},
            {"FeatureName": "account_status", "FeatureType": "String"},
            {"FeatureName": "will_buy_7d", "FeatureType": "Integral"},
            {"FeatureName": "snapshot_date", "FeatureType": "String"},
            {"FeatureName": "event_time", "FeatureType": "Fractional"},
        ],
        OnlineStoreConfig={"EnableOnlineStore": True},
        OfflineStoreConfig={
            "S3StorageConfig": {"S3Uri": f"s3://{bucket}/feature-store/"}
        },
        RoleArn=role,
    )

    print("Waiting for feature group to be created...")
    while True:
        status = sm_client.describe_feature_group(
            FeatureGroupName=FEATURE_GROUP_NAME
        )["FeatureGroupStatus"]
        print(f"  Status: {status}")
        if status == "Created":
            break
        if status == "CreateFailed":
            raise RuntimeError("Feature group creation failed")
        time.sleep(5)

    print("Feature group created.")


def ingest_features(features_df, region):
    """Pure boto3 ingestion — no sagemaker SDK, safe for Docker/Airflow."""
    fs_client = boto3.client("sagemaker-featurestore-runtime", region_name=region)

    df = features_df.copy()
    event_time = str(float(int(time.time())))
    df["customer_id"] = df["customer_id"].astype(int)
    df["recency_days"] = df["recency_days"].astype(float)
    df["orders_30d"] = df["orders_30d"].astype(int)
    df["avg_order_value"] = df["avg_order_value"].astype(float)
    df["email_opens_30d"] = df["email_opens_30d"].astype(int)
    df["age"] = df["age"].astype(int)
    df["will_buy_7d"] = df["will_buy_7d"].astype(int)

    print(f"Ingesting {len(df)} records into Feature Store...")
    for i, row in df.iterrows():
        record = [
            {"FeatureName": "customer_id", "ValueAsString": str(int(row["customer_id"]))},
            {"FeatureName": "recency_days", "ValueAsString": str(float(row["recency_days"]))},
            {"FeatureName": "orders_30d", "ValueAsString": str(int(row["orders_30d"]))},
            {"FeatureName": "avg_order_value", "ValueAsString": str(float(row["avg_order_value"]))},
            {"FeatureName": "email_opens_30d", "ValueAsString": str(int(row["email_opens_30d"]))},
            {"FeatureName": "age", "ValueAsString": str(int(row["age"]))},
            {"FeatureName": "account_status", "ValueAsString": str(row["account_status"])},
            {"FeatureName": "will_buy_7d", "ValueAsString": str(int(row["will_buy_7d"]))},
            {"FeatureName": "snapshot_date", "ValueAsString": str(row["snapshot_date"])},
            {"FeatureName": "event_time", "ValueAsString": event_time},
        ]
        fs_client.put_record(FeatureGroupName=FEATURE_GROUP_NAME, Record=record)

        if (i + 1) % 10 == 0:
            print(f"  {i + 1} records ingested")

    print("Ingestion complete.")


def read_parquet_from_s3(bucket, key, region):
    s3 = boto3.client("s3", region_name=region)
    response = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(response["Body"].read()))


def main():
    config = load_config()
    region = config["aws"]["region"]
    bucket = config["aws"]["s3_bucket"]
    features_prefix = config["s3"]["features_prefix"]
    partition = f"snapshot_date={config['snapshot_date']}"

    print("Reading features from S3...")
    features_df = read_parquet_from_s3(
        bucket,
        f"{features_prefix}/{partition}/features.parquet",
        region,
    )
    features_df = features_df.head(100)
    print(f"  {len(features_df)} rows loaded")

    ingest_features(features_df, region)


if __name__ == "__main__":
    main()
