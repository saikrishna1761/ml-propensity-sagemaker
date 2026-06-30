import boto3
import yaml
import sagemaker
from sagemaker.xgboost.model import XGBoostModel

CONFIG_PATH = "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_latest_model_artifact(bucket, models_prefix, region):
    s3 = boto3.client("s3", region_name=region)
    response = s3.list_objects_v2(Bucket=bucket, Prefix=models_prefix + "/")
    objects = [o for o in response.get("Contents", []) if o["Key"].endswith("model.tar.gz")]
    if not objects:
        raise FileNotFoundError(f"No model.tar.gz found in s3://{bucket}/{models_prefix}/")
    latest = sorted(objects, key=lambda o: o["LastModified"], reverse=True)[0]
    return f"s3://{bucket}/{latest['Key']}"


def main():
    config = load_config()
    region = config["aws"]["region"]
    bucket = config["aws"]["s3_bucket"]
    role = config["aws"]["sagemaker_role_arn"]
    models_prefix = config["s3"]["models_prefix"]
    instance_type = config["serving"]["instance_type"]
    endpoint_name = config["serving"]["endpoint_name"]

    boto_session = boto3.Session(region_name=region)

    if instance_type == "local":
        from sagemaker.local import LocalSession
        sess = LocalSession(boto_session=boto_session)
        sess.config = {"local": {"local_code": True}}
    else:
        sess = sagemaker.Session(boto_session=boto_session)

    print(f"Looking for latest model artifact in s3://{bucket}/{models_prefix}/...")
    model_artifact = get_latest_model_artifact(bucket, models_prefix, region)
    print(f"Model artifact: {model_artifact}")

    model = XGBoostModel(
        model_data=model_artifact,
        role=role,
        framework_version="1.7-1",
        py_version="py3",
        sagemaker_session=sess,
        entry_point="inference.py",
        source_dir="serving",
    )

    print(f"Deploying endpoint: {endpoint_name} (instance: {instance_type})...")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
    )

    print(f"Endpoint deployed: {endpoint_name}")
    return predictor


if __name__ == "__main__":
    main()
