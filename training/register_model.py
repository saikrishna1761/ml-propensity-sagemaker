import boto3
import yaml
import json
import subprocess
import tarfile
import tempfile
import os
from sagemaker import image_uris

CONFIG_PATH = "config.yaml"
MODEL_PACKAGE_GROUP = "propensity-model"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def get_latest_training_job(sm_client):
    response = sm_client.list_training_jobs(
        SortBy="CreationTime",
        SortOrder="Descending",
        StatusEquals="Completed",
        MaxResults=1,
    )
    return response["TrainingJobSummaries"][0]["TrainingJobName"]


def get_metrics_from_job(sm_client, job_name, bucket, region):
    job = sm_client.describe_training_job(TrainingJobName=job_name)
    model_s3_uri = job["ModelArtifacts"]["S3ModelArtifacts"]

    s3 = boto3.client("s3", region_name=region)
    bucket_name = model_s3_uri.split("/")[2]
    key_prefix = "/".join(model_s3_uri.split("/")[3:])

    with tempfile.TemporaryDirectory() as tmpdir:
        local_tar = os.path.join(tmpdir, "model.tar.gz")
        s3.download_file(bucket_name, key_prefix, local_tar)

        with tarfile.open(local_tar, "r:gz") as tar:
            tar.extractall(tmpdir)

        metrics_path = os.path.join(tmpdir, "metrics.json")
        with open(metrics_path) as f:
            metrics = json.load(f)

    return metrics, model_s3_uri


def ensure_model_package_group(sm_client):
    try:
        sm_client.create_model_package_group(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP,
            ModelPackageGroupDescription="Purchase propensity XGBoost model versions",
        )
        print(f"Created model package group: {MODEL_PACKAGE_GROUP}")
    except Exception as e:
        if "already exists" in str(e):
            print(f"Model package group already exists: {MODEL_PACKAGE_GROUP}")
        else:
            raise


def register_model(sm_client, job_name, model_s3_uri, metrics, config):
    region = config["aws"]["region"]
    snapshot_date = config["snapshot_date"]
    git_sha = get_git_sha()

    shap_s3_path = f"s3://{config['aws']['s3_bucket']}/models/{job_name}/output/shap_summary.png"

    xgboost_image = image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.7-1",
    )
    print(f"Using image: {xgboost_image}")

    response = sm_client.create_model_package(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP,
        ModelPackageDescription=(
            f"XGBoost propensity model | "
            f"AUC={metrics['auc_roc']} | "
            f"dataset={snapshot_date} | "
            f"git={git_sha}"
        ),
        InferenceSpecification={
            "Containers": [
                {
                    "Image": xgboost_image,
                    "ModelDataUrl": model_s3_uri,
                    "Framework": "XGBOOST",
                    "FrameworkVersion": "1.7-1",
                }
            ],
            "SupportedContentTypes": ["text/csv", "application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
        },
        ModelApprovalStatus="PendingManualApproval",
        CustomerMetadataProperties={
            "auc_roc": str(metrics["auc_roc"]),
            "average_precision": str(metrics["average_precision"]),
            "recall": str(metrics["recall"]),
            "precision": str(metrics["precision"]),
            "f1": str(metrics["f1"]),
            "training_job_name": job_name,
            "dataset_version": snapshot_date,
            "git_sha": git_sha,
            "shap_report_s3": shap_s3_path,
        },
    )

    model_package_arn = response["ModelPackageArn"]
    print("\nModel registered successfully.")
    print(f"  ARN:    {model_package_arn}")
    print("  Status: PendingManualApproval")
    print(f"  AUC:    {metrics['auc_roc']}")
    print(f"  Git:    {git_sha}")
    print(f"  Data:   {snapshot_date}")
    print("\nApprove with:")
    print(f"  aws sagemaker update-model-package --model-package-arn {model_package_arn} --model-approval-status Approved --region {region}")
    return model_package_arn


def main():
    config = load_config()
    region = config["aws"]["region"]
    bucket = config["aws"]["s3_bucket"]

    sm_client = boto3.client("sagemaker", region_name=region)

    job_name = get_latest_training_job(sm_client)
    print(f"Latest training job: {job_name}")

    metrics, model_s3_uri = get_metrics_from_job(sm_client, job_name, bucket, region)
    print(f"Metrics: {metrics}")

    ensure_model_package_group(sm_client)

    register_model(sm_client, job_name, model_s3_uri, metrics, config)


if __name__ == "__main__":
    main()
