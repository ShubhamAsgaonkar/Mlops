from kfp import dsl, compiler
from kfp.dsl import Dataset, Model, Input, Output
from kfp import kubernetes


# ── Step 1: Load CSV from MinIO ───────────────────────────────────────────────
@dsl.component(
    base_image="python:3.10",
    packages_to_install=["minio", "pandas"]
)
def load_data_from_minio(output_dataset: Output[Dataset]):
    """
    Downloads k8s_app_logs_1000.csv from MinIO bucket 'devops-mlops-data'
    into a KFP-managed artifact path so the next step can consume it.
    Credentials are read from env vars injected by the Kubernetes secret
    'minio-secret' (keys: accesskey, secretkey).
    """
    import os
    from minio import Minio

    client = Minio(
        "minio-service.kubeflow:9000",
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        secure=False
    )

    client.fget_object(
        "devops-mlops-data",           # bucket name
        "logs/k8s_app_logs_1000.csv",  # object path inside bucket
        output_dataset.path            # KFP manages this path (shared storage)
    )

    import pandas as pd
    df = pd.read_csv(output_dataset.path)
    print(f"Loaded dataset: {df.shape}")
    print(df.head())


# ── Step 2: Preprocess ────────────────────────────────────────────────────────
@dsl.component(
    base_image="python:3.10",
    packages_to_install=["pandas"]
)
def preprocess_op(input_dataset: Input[Dataset], output_dataset: Output[Dataset]):
    """
    Parses latency, response codes, p99 latency, error flags, and service
    category from the raw log message column using regex, then writes
    the feature matrix as a CSV for the training step.
    """
    import re
    import pandas as pd

    def extract_latency(msg):
        m = re.search(r'(\d+)ms', str(msg))
        return float(m.group(1)) if m else 0.0

    def extract_response_code(msg):
        for pattern in [
            r'(?:status=|response=)(\d{3})',
            r'Completed\s+(\d{3})\s+',
            r'\b([45]\d{2})\b',
        ]:
            m = re.search(pattern, str(msg))
            if m:
                return int(m.group(1))
        return 200

    def extract_p99_latency(msg):
        m = re.search(r'p99=(\d+)ms', str(msg))
        return float(m.group(1)) if m else 0.0

    df = pd.read_csv(input_dataset.path)
    # guard against 'imestamp' header typo
    df.columns = [c.lstrip('i') if c.startswith('imestamp') else c for c in df.columns]

    df['latency']       = df['message'].apply(extract_latency)
    df['response_code'] = df['message'].apply(extract_response_code)
    df['p99_latency']   = df['message'].apply(extract_p99_latency)

    anomaly_kw = r'(?i)(error|crashloopbackoff|oomkilled|imagepullbackoff|timeout|refused|deadlock|503|502|segmentation fault|latency spike)'
    df['is_error'] = (
        df['log_level'].isin(['ERROR', 'WARN']) |
        df['message'].str.contains(anomaly_kw, regex=True, na=False)
    ).astype(int)

    service_map = {'nginx': 0, 'backend': 1, 'auth': 2, 'payments': 3, 'orders': 4}
    df['service'] = df['pod_name'].apply(
        lambda x: next((v for k, v in service_map.items() if k in str(x)), -1)
    )

    features = df[['latency', 'response_code', 'p99_latency', 'is_error', 'service']].fillna(0)
    features.to_csv(output_dataset.path, index=False)
    print(f"Preprocessed features saved  shape={features.shape}")


# ── Step 3: Train model ───────────────────────────────────────────────────────
@dsl.component(
    base_image="python:3.10",
    packages_to_install=["pandas", "scikit-learn", "joblib", "minio"]
)
def train_op(input_dataset: Input[Dataset], k8s_anomaly_detector: Output[Model]):
    """
    Trains an IsolationForest on the feature matrix.
    contamination=0.25 matches the dataset's known ~25 % anomaly rate.
    """
    import pandas as pd
    from sklearn.ensemble import IsolationForest
    import joblib

    df = pd.read_csv(input_dataset.path)

    model = IsolationForest(contamination=0.25, random_state=42, n_estimators=100)
    model.fit(df)

    preds = model.predict(df)
    n_anom = (preds == -1).sum()
    print(f"Anomalies detected: {n_anom}/{len(preds)} ({n_anom / len(preds) * 100:.1f}%)")

    joblib.dump(model, k8s_anomaly_detector.path)
    print(f"Model saved to KFP artifact path: {k8s_anomaly_detector.path}")

    # ── Also upload explicitly as .pkl to devops-mlops-data bucket ──
    import os
    from minio import Minio

    minio_client = Minio(
        "minio-service.kubeflow:9000",
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        secure=False
    )
    minio_client.fput_object(
        "devops-mlops-data",
        "models/k8s_anomaly_detector.pkl",
        k8s_anomaly_detector.path
    )
    print("Model uploaded to MinIO: devops-mlops-data/models/k8s_anomaly_detector.pkl")


# ── Pipeline definition ───────────────────────────────────────────────────────
@dsl.pipeline(name="k8s-logs-anomaly-pipeline")
def logs_pipeline():
    # Step 1 — fetch CSV from MinIO
    load_task = load_data_from_minio()

    # Inject MinIO credentials from the Kubernetes secret 'minio-secret'
    # (kubectl create secret generic minio-secret -n kubeflow
    #      --from-literal=accesskey=minio --from-literal=secretkey=minio123)
    kubernetes.use_secret_as_env(
        task=load_task,
        secret_name="minio-secret",
        secret_key_to_env={
            "accesskey": "MINIO_ACCESS_KEY",
            "secretkey": "MINIO_SECRET_KEY",
        },
    )

    # Step 2 — preprocess (receives KFP Dataset artifact from step 1)
    preprocess_task = preprocess_op(
        input_dataset=load_task.outputs["output_dataset"]
    )

    # Step 3 — train (receives KFP Dataset artifact from step 2)
    train_task = train_op(
        input_dataset=preprocess_task.outputs["output_dataset"]
    )

    # Inject MinIO credentials into train pod (needed for model upload)
    kubernetes.use_secret_as_env(
        task=train_task,
        secret_name="minio-secret",
        secret_key_to_env={
            "accesskey": "MINIO_ACCESS_KEY",
            "secretkey": "MINIO_SECRET_KEY",
        },
    )


# ── Compile ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=logs_pipeline,
        package_path="logs_pipeline.yaml",
    )
    print("Pipeline compiled → logs_pipeline.yaml")