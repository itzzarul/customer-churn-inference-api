from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "tenure_months": 6,
    "support_tickets": 3,
    "monthly_spend_inr": 799,
    "last_login_days": 14,
    "plan_type": "Basic",
}

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["predict"] == "/predict"

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True

def test_predict_returns_probability_and_warning():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("churned", "not_churned")
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["model_status"] == "experimental_not_validated"
    assert "12 records" in body["warning"]

def test_invalid_plan_type_is_rejected():
    payload = {**VALID_PAYLOAD, "plan_type": "Enterprise"}
    assert client.post("/predict", json=payload).status_code == 422

def test_negative_tenure_is_rejected():
    payload = {**VALID_PAYLOAD, "tenure_months": -1}
    assert client.post("/predict", json=payload).status_code == 422

def test_extra_fields_are_rejected():
    payload = {**VALID_PAYLOAD, "customer_id": "C001"}
    assert client.post("/predict", json=payload).status_code == 422
