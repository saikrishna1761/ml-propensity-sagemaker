from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "ml-team",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
    "email": ["ml-alerts@company.com"],
}

with DAG(
    dag_id="propensity_etl_pipeline",
    description="Nightly ETL: raw Postgres → S3 features for purchase propensity model",
    schedule_interval="0 2 * * *",
    start_date=datetime(2026, 6, 1),
    catchup=False,
    default_args=default_args,
    tags=["ml", "etl", "propensity"],
) as dag:

    extract_load = BashOperator(
        task_id="extract_load_to_s3",
        bash_command="cd /opt/airflow && python data/etl.py",
    )

    validate_raw = BashOperator(
        task_id="validate_raw_data",
        bash_command="cd /opt/airflow && python data/validate.py",
    )

    feature_engineering = BashOperator(
        task_id="feature_engineering",
        bash_command="cd /opt/airflow && python features/feature_engineering.py",
    )

    feature_store_ingest = BashOperator(
        task_id="feature_store_ingest",
        bash_command="cd /opt/airflow && python features/feature_store.py",
    )

    extract_load >> validate_raw >> feature_engineering >> feature_store_ingest
