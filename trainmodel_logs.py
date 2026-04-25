import re
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalise column names (guard against the 'imestamp' typo variant)
    df.columns = [c.strip().lstrip('i') if c.startswith('imestamp') else c.strip()
                  for c in df.columns]
    return df


def extract_latency(msg: str) -> float:
    """Return the first latency value (ms) found in the message, else 0."""
    match = re.search(r'(\d+)ms', str(msg))
    return float(match.group(1)) if match else 0.0


def extract_response_code(msg: str) -> int:
    """Return the first HTTP-like status code (1xx-5xx) found, else 200."""
    # Prefer codes that appear after 'status=' or 'response=' or 'Completed NNN'
    patterns = [
        r'(?:status=|response=)(\d{3})',          # status=200, response=502
        r'Completed\s+(\d{3})\s+',                # Completed 200 GET …
        r'\b([45]\d{2})\b',                        # any 4xx/5xx standalone
    ]
    for pattern in patterns:
        match = re.search(pattern, str(msg))
        if match:
            return int(match.group(1))
    return 200


def extract_p99_latency(msg: str) -> float:
    """Return p99 latency from latency-spike messages, else 0."""
    match = re.search(r'p99=(\d+)ms', str(msg))
    return float(match.group(1)) if match else 0.0


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- feature engineering from message column ---
    df['latency']       = df['message'].apply(extract_latency)
    df['response_code'] = df['message'].apply(extract_response_code)
    df['p99_latency']   = df['message'].apply(extract_p99_latency)

    # binary error flag: ERROR / WARN log levels, or message mentions known anomalies
    anomaly_keywords = r'(?i)(error|crashloopbackoff|oomkilled|imagepullbackoff|timeout|refused|deadlock|503|502|segmentation fault|latency spike)'
    df['is_error'] = (
        df['log_level'].isin(['ERROR', 'WARN']) |
        df['message'].str.contains(anomaly_keywords, regex=True, na=False)
    ).astype(int)

    # encode pod service category (nginx, backend, auth, payments, orders)
    service_map = {
        'nginx': 0, 'backend': 1, 'auth': 2, 'payments': 3, 'orders': 4
    }
    df['service'] = df['pod_name'].apply(
        lambda x: next((v for k, v in service_map.items() if k in str(x)), -1)
    )

    feature_cols = ['latency', 'response_code', 'p99_latency', 'is_error', 'service']
    return df[feature_cols].fillna(0)


def train_model(df: pd.DataFrame):
    # Dataset has ~25 % anomalies per the Kaggle description
    model = IsolationForest(contamination=0.25, random_state=42, n_estimators=100)
    model.fit(df)
    return model


def save_model(model, path: str = "model.pkl"):
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def main():
    print("Loading data …")
    df = load_data("k8s_app_logs_1000.csv")
    print(f"  Raw shape : {df.shape}")

    print("Preprocessing …")
    features = preprocess(df)
    print(f"  Feature shape : {features.shape}")
    print(features.describe().round(2))

    print("Training IsolationForest …")
    model = train_model(features)

    # Quick sanity check
    preds = model.predict(features)          # -1 = anomaly, 1 = normal
    n_anomalies = (preds == -1).sum()
    print(f"  Detected anomalies : {n_anomalies} / {len(preds)} "
          f"({n_anomalies / len(preds) * 100:.1f} %)")

    save_model(model)
    print("Done.")


if __name__ == "__main__":
    main()