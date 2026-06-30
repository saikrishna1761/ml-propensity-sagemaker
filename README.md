# Production Purchase Propensity ML Pipeline on AWS SageMaker

An end-to-end enterprise ML pipeline that predicts customer purchase propensity using AWS SageMaker, Apache Airflow, and production-grade MLOps practices.

## Architecture

```
Postgres (raw data)
    ↓  ETL (data/etl.py)
S3 raw layer (parquet, partitioned by date)
    ↓  Validation (data/validate.py)
S3 features layer (RFM features)
    ↓  Training (training/submit.py)
S3 models layer (XGBoost artifact)
    ↓  Deployment (serving/deploy.py)
SageMaker Endpoint (real-time inference)
```

## Stack

| Layer | Technology |
|---|---|
| Orchestration | Apache Airflow (Docker Compose) |
| Training | AWS SageMaker XGBoost |
| Feature Engineering | Pandas, PyArrow |
| Data Validation | Pandera |
| Serving | SageMaker Real-time Endpoint |
| Storage | S3 (medallion architecture) |
| Secrets | AWS Secrets Manager |
| Infrastructure | Terraform |

## Project Structure

```
├── data/
│   ├── etl.py                  # Extract from Postgres, load to S3
│   ├── seed.py                 # Generate synthetic training data
│   └── validate.py             # Pandera schema validation
├── features/
│   └── feature_engineering.py  # RFM feature computation
├── training/
│   ├── train.py                # Runs inside SageMaker container
│   └── submit.py               # Submits training job from local
├── serving/
│   ├── inference.py            # SageMaker serving contract
│   ├── deploy.py               # Deploys endpoint
│   └── test_inference.py       # Endpoint tests
├── dags/
│   └── propensity_etl_dag.py   # Airflow DAG (nightly at 02:00 UTC)
├── infra/
│   └── main.tf                 # Terraform infrastructure
├── docker-compose.yaml         # Airflow local setup
├── config.yaml.example         # Config template (copy to config.yaml)
└── requirements.txt
```

## Setup

### 1. Clone and configure
```bash
git clone <repo-url>
cd ml-propensity-sagemaker
cp config.yaml.example config.yaml
# Edit config.yaml with your AWS details
```

### 2. Install dependencies
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Start Airflow
```bash
docker compose up -d
```
Open http://localhost:8080 (airflow / airflow)

### 4. Run pipeline manually
```bash
python data/seed.py        # seed Postgres with synthetic data
python data/etl.py         # extract to S3
python data/validate.py    # validate raw data
python features/feature_engineering.py  # compute features
python training/submit.py  # launch SageMaker training job
python serving/deploy.py   # deploy endpoint
```

## Features

- **Medallion architecture** — raw → features → models, each layer immutable in S3
- **Temporal split** — prevents data leakage in label generation
- **RFM features** — Recency, Frequency, Monetary + email engagement
- **Data validation** — Pandera schema checks before feature engineering
- **Automated orchestration** — Airflow DAG runs nightly at 02:00 UTC
- **Retry logic** — 2 retries with 5-minute delay on failure
- **Email alerts** — on task failure
