"""
This module defines the API endpoints for the Stonky application.
"""
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .model import Stonky

logger = logging.getLogger(__name__)
router = APIRouter()
stonky = Stonky()

class PredictionRequest(BaseModel):
    """Request model for stock prediction."""
    ticker: str
    days: int

class EvaluationRequest(BaseModel):
    """Request model for model evaluation."""
    ticker: str

@router.post("/predict")
async def predict_stock(request: PredictionRequest):
    """Predicts the stock price for a given ticker for a number of days."""
    logger.info("Post request for /predict has been made.")
    try:
        predictions = await run_in_threadpool(stonky.predict, request.ticker, request.days)
        logger.info("Prediction on %s is successful.", request.ticker)
        return {"ticker": request.ticker, "days": request.days, "predictions": predictions}
    except Exception as e:
        logger.warning("Failed to make prediction on the stock. %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

@router.post("/refresh")
async def refresh_model(request: EvaluationRequest):
    """Refreshes the model for a given ticker."""
    logger.info("Post request for /refresh has been made.")
    try:
        await run_in_threadpool(stonky.refresh_stonky, request.ticker)
        logger.info("Model for %s has been refreshed.", request.ticker)
        return {"message": f"Model for {request.ticker} refreshed successfully"}
    except Exception as e:
        logger.warning("Unable to refresh model: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

@router.post("/evaluate")
async def evaluate_model(request: EvaluationRequest):
    """Evaluates the model for a given ticker."""
    logger.info("Post request for /evaluate has been made")
    try:
        mse, mae, r2 = await run_in_threadpool(stonky.evaluate, request.ticker)
        logger.info("Successfully obtained performance of the model.")
        return {"ticker": request.ticker, "mse": mse, "mae": mae, "r2": r2}
    except Exception as e:
        logger.warning("Failed to evaluate the model performance.")
        raise HTTPException(status_code=500, detail=str(e)) from e
