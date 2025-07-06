# Developer Documentation

This document provides a technical overview of the Stock Prediction System. It is intended for developers who want to understand, maintain, or contribute to the project.

## 1. Architectural Overview

The system is built on a robust and scalable architecture, now featuring a web-based interface and a RESTful API:

1.  **Web Interface (Gradio - `src/app.py`):**
    -   Provides a user-friendly graphical interface for interacting with the stock prediction system.
    -   Features a "Prediction" tab for forecasting future stock prices, a "Performance" tab for evaluating model accuracy, and a "Documentation" tab with useful links and a ticker symbol cheatsheet.

2.  **RESTful API (FastAPI - `src/api.py`):**
    -   Offers programmatic access to the system's functionalities, including prediction, model evaluation, and model refreshing.
    -   Designed for asynchronous access and uses `run_in_threadpool` to handle CPU-bound tasks, ensuring responsiveness and supporting multiple concurrent users.

3.  **Core Prediction Logic (`src/model.py`):**
    -   Encapsulates all the functionality for loading data, managing models, making predictions, and evaluating performance.
    -   Loads pre-trained models and scalers from the `models/` directory for quick responses.
    -   If no model exists for a given stock, it performs initial training and saves the artifacts.

4.  **Background Refresh Process (`src/refresh.py`):**
    -   A long-running background service (though currently triggered manually via API) that acts as a model guardian.
    -   Its responsibility is to ensure the predictive models remain accurate and up-to-date by periodically evaluating performance and triggering retraining.

This architecture ensures a responsive user experience while allowing for powerful backend operations.

## 2. Codebase Guide

The core logic is organized within the `src/` directory, with `main.py` serving as the application entry point:

-   **`main.py`**: The primary entry point for the application. It initializes the FastAPI application, mounts the Gradio web interface at the root path (`/`), and includes the API endpoints under `/api`. It uses `uvicorn` to serve the application.

-   **`app.py`**: Defines the Gradio web interface, including the layout of the three tabs ("Prediction", "Performance", and "Documentation") and their respective input/output components.

-   **`api.py`**: Implements the RESTful API endpoints (`/api/predict`, `/api/refresh`, `/api/evaluate`) using FastAPI. It handles incoming requests, calls the appropriate `Stonky` class methods, and returns JSON responses.

-   **`model.py`**: This is the heart of the prediction system. It contains the `Stonky` class, which encapsulates all the functionality for loading data, managing models, making predictions, and evaluating performance.

-   **`refresh.py`**: This script implements the automated model maintenance logic. It is designed to run in a continuous loop, checking both performance-based and time-based triggers to decide when to call the `refresh_stonky` method from the `Stonky` class.

-   **`config.py`**: A centralized configuration file for global parameters like `FEATURES`, `LOOKBACK`, `EPOCHS`, and `BATCH_SIZE`. This makes it easy to tune the model without modifying the core logic.

-   **`download.py`**: A utility script responsible for fetching historical stock data from sources like Yahoo Finance.

-   **`data/ticker_cheatsheet.json`**: A JSON file containing a mapping of company names to their ticker symbols, used in the Gradio documentation tab.

## 3. Core Logic: The `Stonky` Class

The `Stonky` class orchestrates all the key operations:

-   `load_data()`: Fetches historical data for a given stock, caching it locally in the `data/` directory.
-   `load_scaler()`: Loads a pre-fitted `MinMaxScaler` for a stock. If one doesn't exist, it creates, fits, and saves a new one.
-   `load_stonky()`: Loads a pre-trained Keras model. If one doesn't exist, it triggers the full training pipeline.
-   `evaluate()`: Calculates and returns the key performance metrics (MSE, MAE, R²) for a given model on the test set.
-   `predict()`: Generates future price predictions for a specified number of days.
-   `refresh_stonky()`: A dedicated method to retrain a model from scratch using the latest data. This is called by the `refresh.py` service or via the API.

## 4. The Model Refresh Strategy

The intelligence of the system lies in its refresh strategy, which is designed to be both proactive and reactive.

-   **Dynamic MAE Threshold:** The Mean Absolute Error (MAE) benchmark is not a fixed value. It is dynamically calculated as **5% of the stock's last closing price**. This makes the performance check meaningful across stocks with vastly different price ranges.
-   **R-squared Threshold:** The R² check ensures the model's explanatory power doesn't degrade. A value below **0.70** indicates that the model is no longer capturing the majority of the price variance effectively.
-   **Scheduled Refresh:** A time-based trigger retrains the model every **30 days**, ensuring that even a well-performing model is kept current with the latest market trends.

## 5. Deployment with Docker

The application can be easily deployed using Docker. A `Dockerfile` is provided in the project root that sets up the environment and runs the application.

To build the Docker image:
```bash
docker build -t stonky-app .
```

To run the Docker container:
```bash
docker run -p 8000:8000 stonky-app
```

This will make the Gradio interface accessible at `http://localhost:8000` and the API endpoints at `http://localhost:8000/api/`.

## 6. How to Contribute

-   **Code Style:** Please adhere to the existing code style, including the use of type hints and comprehensive docstrings.
-   **Configuration:** Add any new model or system parameters to `src/config.py`.
-   **Testing:** Before submitting changes, ensure that both the prediction and refresh scripts run without errors.
