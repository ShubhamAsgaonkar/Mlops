import os
import re
import joblib
import pandas as pd
import json
import requests
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

class PredictRawRequest(BaseModel):
    content: str

class PredictRawResponse(BaseModel):
    pod_name: Optional[str] = None
    anomaly: bool = True
    score: float = 0.91
    message: str
    reasoning: str
    solution: str

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

@app.post("/predict_raw", response_model=PredictRawResponse)
def predict_raw(request: PredictRawRequest):
    vllm_url = os.getenv("VLLM_URL", "http://llm-service:80/v1/chat/completions")
    
    prompt = f"""
    You are a Kubernetes expert and MLOps log analyzer.
    Analyze the following raw log or error message from a user:
    
    "{request.content}"
    
    Extract the following information and return ONLY a valid JSON object with these exact keys:
    - "pod_name": (string) The name of the pod mentioned, or "unknown" if not found.
    - "anomaly": (boolean) Always true for errors/issues.
    - "score": (float) A simulated anomaly score between 0.8 and 1.0 for issues.
    - "message": (string) A short summary of what the error is.
    - "reasoning": (string) Detailed technical reasoning explaining why this error occurred based on the log.
    - "solution": (string) A clear, actionable step-by-step solution to fix the issue.
    
    Do not wrap the response in markdown blocks like ```json. Return just the raw JSON.
    """
    
    try:
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that outputs only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 512
        }
        
        response = requests.post(vllm_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        res_json = response.json()
        
        text = res_json['choices'][0]['message']['content'].strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
            
        result = json.loads(text.strip())
        return PredictRawResponse(
            pod_name=result.get("pod_name", "unknown"),
            anomaly=result.get("anomaly", True),
            score=result.get("score", 0.91),
            message=result.get("message", "Error analyzed"),
            reasoning=result.get("reasoning", "No reasoning provided."),
            solution=result.get("solution", "No solution provided.")
        )
    except Exception as e:
        return PredictRawResponse(
            pod_name="unknown",
            anomaly=True,
            score=0.91,
            message="Error analyzing log with LLM",
            reasoning=f"An exception occurred during LLM processing: {str(e)}",
            solution="Check the LLM API connectivity or the parsing logic."
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
