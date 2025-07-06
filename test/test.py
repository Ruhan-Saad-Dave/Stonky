import pytest
from httpx import AsyncClient
from main import app
from unittest.mock import patch

@pytest.mark.asyncio
async def test_predict_endpoint():
    with patch('src.api.stonky_instance.predict') as mock_predict:
        mock_predict.return_value = [100.0, 101.0, 102.0]
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/api/predict", json={
                "ticker": "GOOG",
                "days": 3
            })
        assert response.status_code == 200
        assert response.json() == {"ticker": "GOOG", "days": 3, "predictions": [100.0, 101.0, 102.0]}
        mock_predict.assert_called_once_with("GOOG", 3)

@pytest.mark.asyncio
async def test_evaluate_endpoint():
    with patch('src.api.stonky_instance.evaluate') as mock_evaluate:
        mock_evaluate.return_value = (0.05, 0.15, 0.95)
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/api/evaluate", json={
                "ticker": "GOOG"
            })
        assert response.status_code == 200
        assert response.json() == {"ticker": "GOOG", "mse": 0.05, "mae": 0.15, "r2": 0.95}
        mock_evaluate.assert_called_once_with("GOOG")

@pytest.mark.asyncio
async def test_refresh_endpoint():
    with patch('src.api.stonky_instance.refresh_stonky') as mock_refresh:
        mock_refresh.return_value = None
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/api/refresh", json={
                "ticker": "GOOG"
            })
        assert response.status_code == 200
        assert response.json() == {"message": "Model for GOOG refreshed successfully"}
        mock_refresh.assert_called_once_with("GOOG")

@pytest.mark.asyncio
async def test_gradio_app_loads():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/")
    assert response.status_code == 200
    assert "Gradio App" in response.text # Check for a common Gradio string
