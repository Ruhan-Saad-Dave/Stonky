from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import gradio as gr

from src.api import router as api_router
from src.app import create_gradio_app

app = FastAPI(
    title="Stonky Stock Prediction API",
    description="API for stock price prediction and model evaluation.",
    version="1.0.0",
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include API router
app.include_router(api_router, prefix="/api")

# Mount Gradio app
gradio_app = create_gradio_app()
app = gr.mount_gradio_app(app, gradio_app, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload = True)