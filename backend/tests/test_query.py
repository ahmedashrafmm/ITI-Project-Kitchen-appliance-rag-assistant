import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_happy_path(client):
    response = client.post("/query", json={"question": "How do I clean the coffee maker?"})
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert "sources" in body
    assert isinstance(body["sources"], list)


def test_query_invalid_input(client):
    response = client.post("/query", json={})
    assert response.status_code == 422


def test_query_empty_question_rejected(client):
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422
