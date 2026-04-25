import os
import re
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

app = FastAPI(title="K8s Log Anomaly Detector")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "k8s_anomaly_detector.pkl")
model = None

# --- Helpers ---

def get_service_code(pod_name: str) -> int:
    service_map = {'nginx': 0, 'backend': 1, 'auth': 2, 'payments': 3, 'orders': 4}
    return next((v for k, v in service_map.items() if k in str(pod_name).lower()), -1)

def get_service_name(code: int) -> str:
    names = {0: 'nginx', 1: 'backend', 2: 'auth', 3: 'payments', 4: 'orders', -1: 'unknown'}
    return names.get(code, 'unknown')

def score_to_risk(score: float, is_anomaly: bool) -> str:
    if not is_anomaly:
        return "LOW"
    if score < -0.1:
        return "CRITICAL"
    if score < -0.05:
        return "HIGH"
    return "MEDIUM"

# --- Input/Output Schema ---

class DetectRequest(BaseModel):
    pod_name: str
    namespace: str
    cpu_usage: float                    # 0.0 – 1.0
    memory_usage: float                 # 0.0 – 1.0
    latency_ms: float
    error_rate: float                   # 0.0 – 1.0  (e.g. 0.12 = 12 %)
    timestamp: Optional[str] = None

class DetectResponse(BaseModel):
    status: str                         # "ANOMALY" | "NORMAL"
    risk_level: str                     # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    anomaly_score: float
    message: str
    details: dict

# --- Lifecycle ---

@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded: {MODEL_PATH}")

# --- Endpoints ---

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/detect", response_model=DetectResponse)
def detect(request: DetectRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    service_code = get_service_code(request.pod_name)

    # Map incoming fields to the 5 features the model was trained on
    features = {
        "latency":       request.latency_ms,
        "response_code": 503 if request.error_rate > 0.1 else 200,
        "p99_latency":   request.latency_ms * 2 if request.latency_ms > 500 else 0.0,
        "is_error":      1 if request.error_rate > 0.05 or request.cpu_usage > 0.90 or request.memory_usage > 0.90 else 0,
        "service":       service_code,
    }

    data = pd.DataFrame([features])
    prediction = model.predict(data)
    score = float(model.decision_function(data)[0])
    is_anomaly = bool(prediction[0] == -1)
    risk = score_to_risk(score, is_anomaly)

    # Human-readable message
    reasons = []
    if request.latency_ms > 500:
        reasons.append(f"high latency ({request.latency_ms:.0f}ms)")
    if request.error_rate > 0.05:
        reasons.append(f"elevated error rate ({request.error_rate*100:.1f}%)")
    if request.cpu_usage > 0.90:
        reasons.append(f"CPU saturation ({request.cpu_usage*100:.0f}%)")
    if request.memory_usage > 0.90:
        reasons.append(f"memory pressure ({request.memory_usage*100:.0f}%)")

    if is_anomaly:
        reason_str = " and ".join(reasons) if reasons else "unusual traffic patterns"
        msg = f"Anomaly detected on pod '{request.pod_name}' in namespace '{request.namespace}': {reason_str}."
    else:
        msg = f"Pod '{request.pod_name}' in namespace '{request.namespace}' is operating normally."

    return DetectResponse(
        status="ANOMALY" if is_anomaly else "NORMAL",
        risk_level=risk,
        anomaly_score=round(score, 6),
        message=msg,
        details={
            "pod": request.pod_name,
            "namespace": request.namespace,
            "service": get_service_name(service_code),
            "latency_ms": request.latency_ms,
            "cpu_usage_pct": f"{request.cpu_usage * 100:.1f}%",
            "memory_usage_pct": f"{request.memory_usage * 100:.1f}%",
            "error_rate_pct": f"{request.error_rate * 100:.1f}%",
            "evaluated_at": request.timestamp or datetime.utcnow().isoformat() + "Z",
            "features_fed_to_model": features,
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
