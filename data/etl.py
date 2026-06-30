import boto3
import json
import psycopg2
import pandas as pd
import yaml
import io

CONFIG_PATH = "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_db_credentials(secret_name, region):
    client = boto3.client("secretsmanager", region_name=region)
    secret = client.get_secret_value(SecretId=secret_name)
    return json.loads(secret["SecretString"])


def get_connection(creds):
    return psycopg2.connect(
        host=creds["host"],
        port=creds["port"],
        dbname=creds["dbname"],
        user=creds["username"],
        password=creds["password"],
    )


def extract_table(conn, query):
    return pd.read_sql(query, conn)


def write_parquet_to_s3(df, bucket, key, region):
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, engine="pyarrow", compression="snappy")
    buffer.seek(0)
    s3 = boto3.client("s3", region_name=region)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    print(f"  Written: s3://{bucket}/{key} ({len(df)} rows)")


def main():
    config = load_config()
    region = config["aws"]["region"]
    bucket = config["aws"]["s3_bucket"]
    raw_prefix = config["s3"]["raw_prefix"]
    snapshot_date = config["snapshot_date"]
    partition = f"snapshot_date={snapshot_date}"

    print("Fetching credentials from Secrets Manager...")
    creds = get_db_credentials(config["postgres"]["secret_name"], region)

    print("Connecting to Postgres...")
    conn = get_connection(creds)

    print("Extracting customers...")
    customers_df = extract_table(conn, "SELECT * FROM customers")

    print("Extracting orders...")
    orders_df = extract_table(conn, "SELECT * FROM orders")

    conn.close()

    print(f"Customers: {len(customers_df)} rows")
    print(f"Orders:    {len(orders_df)} rows")

    print("Writing to S3...")
    write_parquet_to_s3(
        customers_df,
        bucket,
        f"{raw_prefix}/customers/{partition}/customers.parquet",
        region,
    )
    write_parquet_to_s3(
        orders_df,
        bucket,
        f"{raw_prefix}/orders/{partition}/orders.parquet",
        region,
    )

    print("ETL complete.")


if __name__ == "__main__":
    main()
