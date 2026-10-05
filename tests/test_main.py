import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.database import Base, engine

@pytest.fixture(autouse=True)
def setup_database():
    """Ensure database tables are created synchronously before running tests."""
    # Use sync_engine if it's an AsyncEngine, otherwise use engine directly
    sync_eng = getattr(engine, "sync_engine", engine)
    Base.metadata.create_all(bind=sync_eng)
    yield

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint returns 200 OK and correct status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"

def test_predict_transaction_endpoint():
    """Test the real-time ML prediction endpoint with a valid transaction payload."""
    payload = {
        "step": 150,
        "type": "TRANSFER",
        "amount": 250000.0,
        "oldbalanceOrg": 250000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 250000.0
    }
    
    response = client.post("/predictions/score", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "transaction_id" in data
    assert "fraud_probability" in data
    assert "risk_category" in data
    assert "model_version" in data
    assert "top_risk_factors" in data
    
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["risk_category"] in ["LOW", "MEDIUM", "HIGH"]
    assert isinstance(data["top_risk_factors"], list)