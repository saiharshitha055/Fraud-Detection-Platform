import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.database import Base, engine, get_db
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

# Use a separate in-memory SQLite database for testing
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False, future=True)
TestingSessionLocal = async_sessionmaker(
    test_engine, class_=AsyncSession, expire_on_commit=False
)

async def override_get_db():
    async with TestingSessionLocal() as session:
        yield session

# Override the app's database dependency with the test database
app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(autouse=True, scope="function")
async def setup_test_db():
    """Create database tables before each test and drop them after."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

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
    # Verify core enterprise response schema keys
    assert "transaction_id" in data
    assert "fraud_probability" in data
    assert "risk_category" in data
    assert "model_version" in data
    assert "top_risk_factors" in data
    
    # Verify value types and ranges
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["risk_category"] in ["LOW", "MEDIUM", "HIGH"]
    assert isinstance(data["top_risk_factors"], list)