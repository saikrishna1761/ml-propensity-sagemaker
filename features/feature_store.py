import boto3
import yaml
import time
import pandas as pd
import io
from sagemaker.session import Session
from sagemaker.feature_store.feature_group import FeatureGroup, FeatureDefinition, FeatureTypeEnum

CONFIG_PATH = "config.yaml"
FEATURE_GROUP_NAME = "propensity-features"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_or_create_feature_group(sagemaker_session, role, bucket):
    boto_session = sagemaker_session.boto_session
    sm_client = boto_session.client("sagemaker")

    try:
        sm_client.describe_feature_group(FeatureGroupName=FEATURE_GROUP_NAME)
        print(f"Feature group '{FEATURE_GROUP_NAME}' already exists.")
        return FeatureGroup(name=FEATURE_GROUP_NAME, sagemaker_session=sagemaker_session)
    except sm_client.exceptions.ResourceNotFound:
        pass

    print(f"Creating feature group '{FEATURE_GROUP_NAME}'...")

    feature_group = FeatureGroup(
        name=FEATURE_GROUP_NAME,
        sagemaker_session=sagemaker_session,
    )

    feature_group.feature_definitions = [
        FeatureDefinition(feature_name="customer_id", feature_type=FeatureTypeEnum.INTEGRAL),
        FeatureDefinition(feature_name="recency_days", feature_type=FeatureTypeEnum.FRACTIONAL),
        FeatureDefinition(feature_name="orders_30d", feature_type=FeatureTypeEnum.INTEGRAL),
        FeatureDefinition(feature_name="avg_order_value", feature_type=FeatureTypeEnum.FRACTIONAL),
        FeatureDefinition(feature_name="email_opens_30d", feature_type=FeatureTypeEnum.INTEGRAL),
        FeatureDefinition(feature_name="age", feature_type=FeatureTypeEnum.INTEGRAL),
        FeatureDefinition(feature_name="account_status", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="will_buy_7d", feature_type=FeatureTypeEnum.INTEGRAL),
        FeatureDefinition(feature_name="snapshot_date", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="event_time", feature_type=FeatureTypeEnum.FRACTIONAL),
    ]

    feature_group.create(
        s3_uri=f"s3://{bucket}/feature-store/",
        record_identifier_name="customer_id",
        event_time_feature_name="event_time",
        role_arn=role,
        enable_online_store=True,
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
    return feature_group


def ingest_features(feature_group, features_df):
    df = features_df.copy()
    df["event_time"] = float(int(time.time()))
    df["customer_id"] = df["customer_id"].astype(int)
    df["recency_days"] = df["recency_days"].astype(float)
    df["orders_30d"] = df["orders_30d"].astype(int)
    df["avg_order_value"] = df["avg_order_value"].astype(float)
    df["email_opens_30d"] = df["email_opens_30d"].astype(int)
    df["age"] = df["age"].astype(int)
    df["will_buy_7d"] = df["will_buy_7d"].astype(int)

    print(f"Ingesting {len(df)} records into Feature Store...")
    feature_group.ingest(data_frame=df, max_workers=1, wait=True)
    print("Ingestion complete.")


def read_parquet_from_s3(bucket, key, region):
    s3 = boto3.client("s3", region_name=region)
    response = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(response["Body"].read()))


def main():
    config = load_config()
    region = config["aws"]["region"]
    bucket = config["aws"]["s3_bucket"]
    role = config["aws"]["sagemaker_role_arn"]
    features_prefix = config["s3"]["features_prefix"]
    partition = f"snapshot_date={config['snapshot_date']}"

    boto_session = boto3.Session(region_name=region)
    sagemaker_session = Session(boto_session=boto_session)

    feature_group = get_or_create_feature_group(sagemaker_session, role, bucket)

    print("Reading features from S3...")
    features_df = read_parquet_from_s3(
        bucket,
        f"{features_prefix}/{partition}/features.parquet",
        region,
    )
    print(f"  {len(features_df)} rows loaded")

    ingest_features(feature_group, features_df)


if __name__ == "__main__":
    main()
