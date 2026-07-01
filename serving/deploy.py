import boto3
import yaml
import sagemaker
from sagemaker.xgboost.model import XGBoostModel

CONFIG_PATH = "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


MODEL_PACKAGE_GROUP = "propensity-model"


def get_approved_model_from_registry(sm_client):
    response = sm_client.list_model_packages(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    packages = response.get("ModelPackageSummaryList", [])
    if not packages:
        raise RuntimeError(
            f"No Approved model found in registry group '{MODEL_PACKAGE_GROUP}'. "
            "Run register_model.py and approve it first."
        )
    arn = packages[0]["ModelPackageArn"]
    version = packages[0]["ModelPackageVersion"]

    detail = sm_client.describe_model_package(ModelPackageName=arn)
    model_data_url = detail["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
    metadata = detail.get("CustomerMetadataProperties", {})

    print(f"Found Approved model: version={version}")
    print(f"  ARN:     {arn}")
    print(f"  AUC:     {metadata.get('auc_roc', 'N/A')}")
    print(f"  Git SHA: {metadata.get('git_sha', 'N/A')}")
    print(f"  Dataset: {metadata.get('dataset_version', 'N/A')}")
    print(f"  Model:   {model_data_url}")

    return model_data_url


def main():
    config = load_config()
    region = config["aws"]["region"]
    role = config["aws"]["sagemaker_role_arn"]
    instance_type = config["serving"]["instance_type"]
    endpoint_name = config["serving"]["endpoint_name"]

    boto_session = boto3.Session(region_name=region)
    sm_client = boto3.client("sagemaker", region_name=region)

    if instance_type == "local":
        from sagemaker.local import LocalSession
        sess = LocalSession(boto_session=boto_session)
        sess.config = {"local": {"local_code": True}}
    else:
        sess = sagemaker.Session(boto_session=boto_session)

    print(f"Fetching latest Approved model from registry: {MODEL_PACKAGE_GROUP}...")
    model_artifact = get_approved_model_from_registry(sm_client)

    # Delete existing endpoint config if present so we can redeploy cleanly
    try:
        sm_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        print(f"Deleted existing endpoint config: {endpoint_name}")
    except sm_client.exceptions.ClientError:
        pass

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
