from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_health_check():
    """
    Test the health check endpoint returns the correct status and service name.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "pdf-rag"
    }
