import boto3
import json
import yaml
import urllib.request

CONFIG_PATH = "config.yaml"
LOCAL_ENDPOINT_URL = "http://localhost:8080/invocations"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def invoke_endpoint(endpoint_name, payload, region, local=False):
    if local:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            LOCAL_ENDPOINT_URL,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    else:
        client = boto3.client("sagemaker-runtime", region_name=region)
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload),
        )
        return json.loads(response["Body"].read())


def main():
    config = load_config()
    region = config["aws"]["region"]
    endpoint_name = config["serving"]["endpoint_name"]
    is_local = config["serving"]["instance_type"] == "local"

    # sample customers with different profiles
    test_cases = [
        {
            "name": "High-value active customer",
            "features": {
                "recency_days": 3,
                "orders_30d": 5,
                "avg_order_value": 250.0,
                "email_opens_30d": 12,
                "age": 35,
            },
        },
        {
            "name": "Dormant customer",
            "features": {
                "recency_days": 180,
                "orders_30d": 0,
                "avg_order_value": 50.0,
                "email_opens_30d": 0,
                "age": 55,
            },
        },
        {
            "name": "New customer, moderate activity",
            "features": {
                "recency_days": 15,
                "orders_30d": 2,
                "avg_order_value": 120.0,
                "email_opens_30d": 4,
                "age": 28,
            },
        },
    ]

    print(f"Calling endpoint: {endpoint_name}\n")
    print(f"{'Customer':<35} {'Propensity Score':>16}")
    print("-" * 53)

    for case in test_cases:
        result = invoke_endpoint(endpoint_name, case["features"], region, local=is_local)
        score = result[0]["propensity_score"]
        print(f"{case['name']:<35} {score:>16.4f}")

    print("\nDone. Scores range from 0.0 (unlikely to buy) to 1.0 (very likely to buy).")


if __name__ == "__main__":
    main()
