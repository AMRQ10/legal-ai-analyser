import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database import Base, get_db
from main import app
from unittest.mock import MagicMock, patch

@pytest.fixture(autouse=True)
def mock_embedder():
    with patch("rag.embedder.SentenceTransformer") as mock:
        mock_instance = MagicMock()
        mock_instance.encode.return_value = [[0.1, 0.2, 0.3] * 128]
        mock.return_value = mock_instance
        yield mock

@pytest.fixture(autouse=True)
def mock_redis():
    with patch("cache.redis_cache.redis.from_url") as mock:
        mock_client = MagicMock()
        mock_client.get.return_value = None
        mock_client.ping.return_value = True
        mock.return_value = mock_client
        yield mock

# Use a separate test database - never test against production data
TEST_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(scope="function")
def client():
    """
    Creates fresh database tables before each test,
    drops them after. Every test starts with a clean slate.
    """
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as test_client:
        yield test_client
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def test_user_data():
    return {
        "email": "testuser@example.com",
        "password": "SecurePass123",
        "full_name": "Test User"
    }

@pytest.fixture
def authenticated_client(client, test_user_data):
    """
    Registers a user, logs them in, and returns a client
    with the auth header already set. Use this for any test
    that needs a logged-in user.
    """
    client.post("/auth/register", json=test_user_data)
    response = client.post("/auth/login", json={
        "email": test_user_data["email"],
        "password": test_user_data["password"]
    })
    token = response.json()["access_token"]
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client