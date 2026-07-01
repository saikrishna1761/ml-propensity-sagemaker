import boto3
import yaml
import sagemaker
from sagemaker.xgboost.estimator import XGBoost

CONFIG_PATH = "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    region = config["aws"]["region"]
    bucket = config["aws"]["s3_bucket"]
    role = config["aws"]["sagemaker_role_arn"]
    instance_type = config["training"]["instance_type"]
    snapshot_date = config["snapshot_date"]

    boto_session = boto3.Session(region_name=region)

    if instance_type == "local":
        from sagemaker.local import LocalSession
        sess = LocalSession(boto_session=boto_session)
        sess.config = {"local": {"local_code": True}}
        train_uri = f"s3://{bucket}/features/snapshot_date={snapshot_date}"
    else:
        sess = sagemaker.Session(boto_session=boto_session)
        train_uri = f"s3://{bucket}/features/snapshot_date={snapshot_date}"

    print(f"Instance type : {instance_type}")
    print(f"Training data : {train_uri}")
    print(f"Role          : {role}")

    estimator = XGBoost(
        entry_point="train.py",
        source_dir="training",
        role=role,
        instance_type=instance_type,
        instance_count=1,
        framework_version="1.7-1",
        py_version="py3",
        sagemaker_session=sess,
        hyperparameters={
            "n-estimators": 200,
            "max-depth": 5,
            "learning-rate": 0.1,
            "subsample": 0.8,
            "colsample-bytree": 0.8,
            "scale-pos-weight": 2.5,
        },
        environment={
            "SNAPSHOT_DATE": snapshot_date,
        },
        output_path=f"s3://{bucket}/models/",
    )

    print("Submitting training job...")
    estimator.fit({"train": train_uri})

    print("\nTraining complete.")
    print(f"Model artifact: {estimator.model_data}")


if __name__ == "__main__":
    main()
