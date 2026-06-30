import os
import json
import joblib
import pandas as pd

FEATURE_COLS = ["recency_days", "orders_30d", "avg_order_value", "email_opens_30d", "age"]


def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.joblib")
    return joblib.load(model_path)


def input_fn(request_body, content_type="application/json"):
    if content_type == "application/json":
        data = json.loads(request_body)
        if isinstance(data, dict):
            data = [data]
        return pd.DataFrame(data)[FEATURE_COLS]
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    scores = model.predict_proba(input_data)[:, 1]
    return scores


def output_fn(prediction, accept="application/json"):  # noqa: ARG001
    result = [{"propensity_score": round(float(score), 4)} for score in prediction]
    return json.dumps(result), "application/json"
