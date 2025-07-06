# Stock Smith: A Stock Price Prediction System

This project is a stock price prediction system that uses a Long Short-Term Memory (LSTM) neural network to forecast future stock prices based on historical data.

## Project Overview

The system is designed to be a robust and maintainable application. It is architected with a clear separation of concerns, featuring an interactive prediction module and an automated background service for model maintenance.

## Getting Started

### Prerequisites

Ensure you have Python 3.12 or higher installed. The required libraries are listed in the `pyproject.toml` file and can be installed using a package manager `UV`

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Stonky
   ```

2. **Install dependencies:**
   ```bash
   rye sync
   ```
   or
   ```bash
   pip install -r requirements.txt # (You may need to generate this file from pyproject.toml)
   ```

## How to Use

The system operates in two independent modes: **Prediction** and **Refresh**.

### 1. Getting Predictions (Interactive Mode)

To get stock predictions, run the main application script. It will prompt you for a stock ticker and the number of days you want to forecast.

```bash
python src/model.py
```

This script will automatically train a model if one doesn't exist for the requested stock, evaluate its performance, and then print the future predictions.

### 2. Automated Model Refresh (Background Service)

The system includes a background service that continuously monitors the model's performance and retrains it when necessary. This ensures the model stays up-to-date with market changes.

To start the refresh service, run the following command in a separate terminal:

```bash
python src/refresh.py
```

The service will run in the background, checking the model every 24 hours and retraining it based on performance and time-based triggers.

## Project Structure

- `data/`: Contains the raw data (e.g., `GOOG.csv`).
- `docs/`: Project documentation, including the phase reports.
- `models/`: Stores the saved model files (`.h5`) and scalers (`.pkl`).
- `src/`:
  - `model.py`: The main application script for user interaction and predictions.
  - `refresh.py`: The background service for automated model retraining.
  - `config.py`: Centralized configuration for model parameters.
  - `download.py`: Script for downloading stock data.
