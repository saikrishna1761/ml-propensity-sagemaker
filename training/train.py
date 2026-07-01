import argparse
import os
import json
import subprocess
import pandas as pd
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, f1_score, precision_score
import joblib
import mlflow
import mlflow.xgboost

FEATURE_COLS = ["recency_days", "orders_30d", "avg_order_value", "email_opens_30d", "age"]
LABEL_COL = "will_buy_7d"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--scale-pos-weight", type=float, default=2.5)

    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))

    return parser.parse_args()


def load_data(data_dir):
    files = [f for f in os.listdir(data_dir) if f.endswith(".parquet")]
    dfs = [pd.read_parquet(os.path.join(data_dir, f)) for f in files]
    return pd.concat(dfs, ignore_index=True)


def get_git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def setup_mlflow_tracking():
    """Connect MLflow to SageMaker Experiments when running inside a training job."""
    training_job_name = os.environ.get("TRAINING_JOB_NAME", "")
    if training_job_name:
        try:
            from sagemaker.experiments.run import Run
            from sagemaker.session import Session
            import boto3
            sagemaker_session = Session(boto_session=boto3.Session())
            run = Run(
                experiment_name="propensity-model",
                run_name=training_job_name,
                sagemaker_session=sagemaker_session,
            )
            print(f"Connected to SageMaker Experiments: propensity-model / {training_job_name}")
            return run
        except Exception as e:
            print(f"SageMaker Experiments setup failed: {e}")
    return None


def train(args):
    sm_run = setup_mlflow_tracking()

    mlflow.set_experiment("propensity-model")

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

    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "scale_pos_weight": args.scale_pos_weight,
    }

    tags = {
        "git_sha": get_git_sha(),
        "dataset_version": os.environ.get("SNAPSHOT_DATE", "unknown"),
        "training_job_name": os.environ.get("TRAINING_JOB_NAME", "local"),
    }

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.set_tags(tags)

        model = xgb.XGBClassifier(
            **params,
            eval_metric="auc",
            random_state=42,
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

        val_preds_proba = model.predict_proba(X_val)[:, 1]
        val_preds = model.predict(X_val)

        auc_roc = roc_auc_score(y_val, val_preds_proba)
        avg_precision = average_precision_score(y_val, val_preds_proba)
        recall = recall_score(y_val, val_preds)
        precision = precision_score(y_val, val_preds)
        f1 = f1_score(y_val, val_preds)

        print(f"Validation AUC-ROC:          {auc_roc:.4f}")
        print(f"Validation Average Precision: {avg_precision:.4f}")
        print(f"Validation Recall:            {recall:.4f}")
        print(f"Validation Precision:         {precision:.4f}")
        print(f"Validation F1:               {f1:.4f}")

        metrics = {
            "auc_roc": round(auc_roc, 4),
            "average_precision": round(avg_precision, 4),
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
        }

        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, name="model")

        print("Computing SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val)

        shap.summary_plot(shap_values, X_val, show=False)
        shap_plot_path = os.path.join(args.model_dir, "shap_summary.png")
        os.makedirs(args.model_dir, exist_ok=True)
        plt.savefig(shap_plot_path, bbox_inches="tight", dpi=150)
        plt.close()
        mlflow.log_artifact(shap_plot_path, artifact_path="shap")
        print(f"SHAP summary saved to {shap_plot_path}")

        feature_importance = dict(zip(FEATURE_COLS, abs(shap_values).mean(axis=0).tolist()))
        print("Feature importances (mean |SHAP|):")
        for feat, imp in sorted(feature_importance.items(), key=lambda x: -x[1]):
            print(f"  {feat}: {imp:.4f}")
        mlflow.log_metrics({f"shap_{k}": round(v, 4) for k, v in feature_importance.items()})

        # Also log to SageMaker Experiments if running inside a training job
        if sm_run:
            for k, v in params.items():
                sm_run.log_parameter(k, v)
            for k, v in metrics.items():
                sm_run.log_metric(k, v)
            for k, v in tags.items():
                sm_run.log_parameter(k, v)

        with open(os.path.join(args.model_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        model_path = os.path.join(args.model_dir, "model.joblib")
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
