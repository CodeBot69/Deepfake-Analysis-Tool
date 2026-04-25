"""
API endpoint tests using FastAPI TestClient.
"""
import pytest
import numpy as np
import cv2
import io
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from app import app
    return TestClient(app)


@pytest.fixture
def test_image_bytes():
    """Create a test image as bytes."""
    img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", img)
    return encoded.tobytes()


class TestHealthEndpoint:
    def test_health_check(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "ensemble_weights" in data


class TestAnalyzeFrameEndpoint:
    def test_analyze_single_frame(self, client, test_image_bytes):
        response = client.post(
            "/api/analyze/frame",
            files={"file": ("test.jpg", io.BytesIO(test_image_bytes), "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "faces_detected" in data
        assert "overall" in data

    def test_invalid_file_format(self, client):
        response = client.post(
            "/api/analyze/frame",
            files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
        )
        assert response.status_code == 400


class TestAnalyzeEndpoint:
    def test_analyze_image_upload(self, client, test_image_bytes):
        response = client.post(
            "/api/analyze",
            files={"file": ("test.jpg", io.BytesIO(test_image_bytes), "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "queued"

    def test_unsupported_extension(self, client):
        response = client.post(
            "/api/analyze",
            files={"file": ("test.xyz", io.BytesIO(b"data"), "application/octet-stream")},
        )
        assert response.status_code == 400


class TestResultsEndpoint:
    def test_nonexistent_job(self, client):
        response = client.get("/api/results/nonexistent-job-id")
        assert response.status_code == 404

    def test_job_lifecycle(self, client, test_image_bytes):
        # Submit analysis
        resp = client.post(
            "/api/analyze",
            files={"file": ("test.jpg", io.BytesIO(test_image_bytes), "image/jpeg")},
        )
        job_id = resp.json()["job_id"]

        # Check results (may still be processing)
        resp = client.get(f"/api/results/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == job_id
        assert data["status"] in ("queued", "processing", "completed", "failed")
