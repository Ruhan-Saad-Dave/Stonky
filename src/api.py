from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from typing import List

from model import Stonky

router = APIRouter()
stonky = Stonky()

class PredictionRequest(BaseModel):
    ticker: str
    days: int

class EvaluationRequest(BaseModel):
    ticker: str

@router.post("/predict")
async def predict_stock(request: PredictionRequest):
    try:
        predictions = await run_in_threadpool(stonky.predict, request.ticker, request.days)
        return {"ticker": request.ticker, "days": request.days, "predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/refresh")
async def refresh_model(request: EvaluationRequest):
    try:
        await run_in_threadpool(stonky.refresh_stonky, request.ticker)
        return {"message": f"Model for {request.ticker} refreshed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate")
async def evaluate_model(request: EvaluationRequest):
    try:
        mse, mae, r2 = await run_in_threadpool(stonky.evaluate, request.ticker)
        return {"ticker": request.ticker, "mse": mse, "mae": mae, "r2": r2}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
