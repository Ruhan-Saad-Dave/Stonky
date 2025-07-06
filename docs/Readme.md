# Developer Documentation

This document provides a technical overview of the Stock Smith prediction system. It is intended for developers who want to understand, maintain, or contribute to the project.

## 1. Architectural Overview

The system is built on a decoupled, two-process architecture designed for robustness and scalability:

1.  **Interactive Prediction Process (`src/model.py`):**
    - This is the user-facing application.
    - It handles on-demand prediction requests.
    - It's designed to be lightweight and responsive. When a user requests a prediction for a stock, it loads the pre-trained model and scaler from the `models/` directory to provide a quick response.
    - If no model exists for a given stock, it will perform an initial training and save the artifacts before making a prediction.

2.  **Background Refresh Process (`src/refresh.py`):**
    - This is a long-running background service that acts as a model guardian.
    - Its sole responsibility is to ensure the predictive models remain accurate and up-to-date.
    - It operates on a schedule, periodically evaluating model performance and triggering retraining based on a set of predefined rules. This intensive process is decoupled from the user-facing application to avoid impacting prediction latency.

This separation ensures that the user experience is never compromised by the resource-intensive task of model training.

## 2. Codebase Guide

The core logic is organized within the `src/` directory:

-   **`model.py`**: This is the heart of the user-facing application. It contains the `Stonky` class, which encapsulates all the functionality for loading data, managing models, making predictions, and evaluating performance. The `if __name__ == "__main__":` block serves as the entry point for interactive use.

-   **`refresh.py`**: This script implements the automated model maintenance logic. It runs in a continuous loop, checking both performance-based and time-based triggers to decide when to call the `refresh_stonky` method from the `Stonky` class.

-   **`config.py`**: A centralized configuration file for global parameters like `FEATURES`, `LOOKBACK`, `EPOCHS`, and `BATCH_SIZE`. This makes it easy to tune the model without modifying the core logic.

-   **`download.py`**: A utility script (currently integrated into `model.py` but can be expanded) responsible for fetching historical stock data from sources like Yahoo Finance.

## 3. Core Logic: The `Stonky` Class

The `Stonky` class orchestrates all the key operations:

-   `load_data()`: Fetches historical data for a given stock, caching it locally in the `data/` directory.
-   `load_scaler()`: Loads a pre-fitted `MinMaxScaler` for a stock. If one doesn't exist, it creates, fits, and saves a new one.
-   `load_stonky()`: Loads a pre-trained Keras model. If one doesn't exist, it triggers the full training pipeline.
-   `evaluate()`: Calculates and returns the key performance metrics (MSE, MAE, R²) for a given model on the test set.
-   `predict()`: Generates future price predictions for a specified number of days.
-   `refresh_stonky()`: A dedicated method to retrain a model from scratch using the latest data. This is called by the `refresh.py` service.

## 4. The Model Refresh Strategy

The intelligence of the system lies in its refresh strategy, which is designed to be both proactive and reactive.

-   **Dynamic MAE Threshold:** The Mean Absolute Error (MAE) benchmark is not a fixed value. It is dynamically calculated as **5% of the stock's last closing price**. This makes the performance check meaningful across stocks with vastly different price ranges.
-   **R-squared Threshold:** The R² check ensures the model's explanatory power doesn't degrade. A value below **0.70** indicates that the model is no longer capturing the majority of the price variance effectively.
-   **Scheduled Refresh:** A time-based trigger retrains the model every **30 days**, ensuring that even a well-performing model is kept current with the latest market trends.

## 5. How to Contribute

-   **Code Style:** Please adhere to the existing code style, including the use of type hints and comprehensive docstrings.
-   **Configuration:** Add any new model or system parameters to `src/config.py`.
-   **Testing:** Before submitting changes, ensure that both the prediction and refresh scripts run without errors.
