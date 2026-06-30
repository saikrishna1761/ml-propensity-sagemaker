import argparse
import os
import json
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib

FEATURE_COLS = ["recency_days", "orders_30d", "avg_order_value", "email_opens_30d", "age"]
LABEL_COL = "will_buy_7d"


def parse_args():
    parser = argparse.ArgumentParser()

    # hyperparameters
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--scale-pos-weight", type=float, default=2.5)

    # SageMaker injects these as environment variables
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))

    return parser.parse_args()


def load_data(data_dir):
    files = [f for f in os.listdir(data_dir) if f.endswith(".parquet")]
    dfs = [pd.read_parquet(os.path.join(data_dir, f)) for f in files]
    return pd.concat(dfs, ignore_index=True)


def train(args):
    print(f"Loading data from {args.train}...")
    df = load_data(args.train)
    print(f"Data shape: {df.shape}")
    print(f"Label distribution:\n{df[LABEL_COL].value_counts().to_dict()}")

    X = df[FEATURE_COLS]
    y = df[LABEL_COL]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")

    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        scale_pos_weight=args.scale_pos_weight,
        eval_metric="auc",
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    val_preds = model.predict_proba(X_val)[:, 1]
    auc_roc = roc_auc_score(y_val, val_preds)
    avg_precision = average_precision_score(y_val, val_preds)

    print(f"Validation AUC-ROC:          {auc_roc:.4f}")
    print(f"Validation Average Precision: {avg_precision:.4f}")

    metrics = {
        "auc_roc": round(auc_roc, 4),
        "average_precision": round(avg_precision, 4),
    }

    os.makedirs(args.model_dir, exist_ok=True)

    with open(os.path.join(args.model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
