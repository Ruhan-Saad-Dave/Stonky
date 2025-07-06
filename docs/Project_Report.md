# Stock Prediction System: Phase 1 Report

## 1. Introduction

This report details the first phase of the stock prediction project. The primary objective of this phase was to develop a functional proof-of-concept for predicting future stock prices using a Long Short-Term Memory (LSTM) neural network. The project successfully implements a multivariate time series model using Google (GOOG) stock data.

## 2. Methodology

The project followed a structured approach, encompassing data acquisition, preprocessing, model development, and evaluation.

### 2.1. Data

- **Source:** The project utilizes historical stock data for Google (GOOG), obtained from `yfinance` and stored in `data/GOOG.csv`.
- **Features:** The model was trained on a multivariate dataset including the following features: `Open`, `High`, `Low`, `Close`, and `Volume`.

### 2.2. Preprocessing

- **Scaling:** All features were normalized using the `MinMaxScaler` from scikit-learn to a range between 0 and 1. This is a crucial step for LSTM models to ensure stable training.
- **Sequencing:** The time series data was transformed into sequences of a fixed length (`LOOKBACK = 60` days) to be used as input for the LSTM model. The target for each sequence was the 'Close' price of the subsequent day.

### 2.3. Model Architecture

- **Framework:** The model was built using TensorFlow and Keras.
- **Layers:** The architecture consists of:
    - An LSTM layer with 64 units and `return_sequences=True`.
    - A Dropout layer with a rate of 0.2 to prevent overfitting.
    - A second LSTM layer with 32 units.
    - Another Dropout layer with a rate of 0.2.
    - A Dense output layer with a single unit to predict the 'Close' price.
- **Compilation:** The model was compiled with the Adam optimizer and Mean Squared Error (MSE) as the loss function.

### 2.4. Training and Evaluation

- **Training:** The model was trained for 10 epochs with a batch size of 32.
- **Evaluation:** The model's performance was evaluated on a held-out test set (20% of the data) using the following metrics:
    - Mean Squared Error (MSE)
    - Mean Absolute Error (MAE)
    - R-squared (R²)

## 3. Results

The model demonstrated its ability to learn the trends in the historical data and make reasonable predictions. The evaluation metrics on the test set provide a quantitative measure of its performance. The saved model (`stock_predictor.h5`) and scaler (`scaler.gz`) allow for reproducible predictions.

## 4. Phase 2: Architectural Refinements and Automation

Phase 2 focused on maturing the project from a single script into a robust, maintainable, and production-ready system. The key improvements were:

### 4.1. Object-Oriented Refactoring
The entire codebase was refactored into a `Stonky` class within `src/model.py`. This encapsulates all the core logic for data handling, training, prediction, and evaluation, making the system more organized, reusable, and easier to debug.

### 4.2. Modular Code Structure
The initial monolithic script was split into a modular structure:
- `src/model.py`: Contains the main `Stonky` class for user interactions and predictions.
- `src/refresh.py`: A new, separate script to handle the automated model retraining logic.
- `src/config.py`: Centralized configuration for model parameters.
- `src/download.py`: Dedicated script for downloading stock data.

This separation of concerns is a critical step towards building a scalable application.

### 4.3. Automated Model Refresh System
A key feature of Phase 2 is the introduction of an automated model refresh service (`src/refresh.py`). This background service ensures the model remains accurate over time by automatically retraining it based on two triggers:
- **Performance-Based Trigger:** The model is retrained if its R² score drops below 0.70 or if its Mean Absolute Error (MAE) exceeds 5% of the stock's last closing price. This dynamic MAE threshold makes the system adaptable to different stocks.
- **Time-Based Trigger:** The model is automatically retrained every 30 days, regardless of performance, to ensure it always incorporates the latest market data.

### 4.4. Code Quality Enhancements
- **Type Hinting:** Type hints were added throughout the codebase, improving readability and allowing for static analysis to catch potential errors early.
- **Docstrings:** Comprehensive docstrings were added to all functions and methods, clarifying their purpose, arguments, and return values.

These changes have transformed the project from a simple proof-of-concept into a well-structured and automated system, laying a solid foundation for future deployment and feature expansion.

## 5. Future Work

The next phase of the project will focus on:

- **Hyperparameter Tuning:** Optimizing the model's architecture and training parameters to improve accuracy.
- **Feature Engineering:** Incorporating additional data sources, such as technical indicators or sentiment analysis, to enhance predictive power.
- **Deployment:** Creating a user-friendly interface (e.g., a web application or API) to interact with the model.
- **Advanced Models:** Exploring more complex architectures, such as attention-based models or transformers, for potentially better performance.
