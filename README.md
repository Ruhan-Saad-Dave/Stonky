# Stocky: Your Personal Stock Prediction Assistant 📈

![Stonk Guy Meme Placeholder](path/to/your/stonk_guy_meme.png)

Welcome to **Stocky**, your go-to tool for predicting stock prices with the power of AI! Whether you're a seasoned investor or just curious about future market trends, Stocky provides an intuitive way to get insights into stock movements.

## ✨ What is Stocky?

Stocky is a stock price prediction system built using a Long Short-Term Memory (LSTM) neural network. It analyzes historical stock data to forecast future prices, helping you make informed decisions.

## 🚀 Features at a Glance

-   **Interactive Predictions:** Get instant stock price predictions for any ticker symbol and a specified number of days into the future.
-   **Performance Insights:** See how well our models are performing with key metrics like MSE, MAE, and R2 score.
-   **Easy-to-Use Interface:** A friendly web interface powered by Gradio makes predictions and evaluations a breeze.
-   **Robust API:** Integrate Stocky's powerful prediction and evaluation capabilities into your own applications using our FastAPI endpoints.
-   **Automated Model Refresh:** Models can be refreshed to stay accurate with the latest market data.
-   **Dockerized Deployment:** Easily deploy Stocky anywhere with Docker.

## 📖 Dive Deeper

For a comprehensive technical overview, architectural details, codebase guide, and more, please refer to our [Developer Documentation](docs/Readme.md).

## 🏁 Getting Started

### Prerequisites

Ensure you have Python 3.9 or higher installed. We use `uv` as our package manager for fast and reliable dependency management.

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Stonky
    ```

2.  **Install `uv` (if you don't have it):**
    ```bash
    pip install uv
    ```

3.  **Install dependencies using `uv`:**
    ```bash
    uv pip install -r requirements.txt
    ```
    *(You might need to generate `requirements.txt` from `pyproject.toml` if it's not present. A simple `uv pip freeze > requirements.txt` can help.)*

## 🏃 How to Run Stocky

Stocky runs as a web application, accessible through your browser.

1.  **Start the application:**
    ```bash
    uv run main.py
    ```

2.  **Access the interface:**
    Once the application starts, open your web browser and navigate to `http://127.0.0.1:8000` (or the address shown in your terminal).

    -   **Prediction Tab:** Enter a stock ticker (e.g., `GOOG`) and the number of days to predict. You'll see a graph and a table of predictions.
    -   **Performance Tab:** Enter a stock ticker to view model performance metrics and a deviation plot.
    -   **Documentation Tab:** Find useful links and a ticker symbol cheatsheet.

## 🐳 Docker Deployment

For easy deployment, Stocky comes with a `Dockerfile`.

1.  **Build the Docker image:**
    ```bash
    docker build -t stonky-app .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -p 8000:8000 stonky-app
    ```
    This will make the Gradio interface accessible at `http://localhost:8000` and the API endpoints at `http://localhost:8000/api/`.

## 📂 Project Structure

-   `data/`: Stores historical stock data (e.g., `GOOG.csv`) and the `ticker_cheatsheet.json`.
-   `docs/`: Project documentation, including detailed developer guides (`Readme.md`).
-   `models/`: Contains saved model files (`.h5`) and scalers (`.pkl`).
-   `src/`:
    -   `main.py`: The main application entry point, serving both the Gradio UI and FastAPI API.
    -   `app.py`: Defines the Gradio web interface.
    -   `api.py`: Implements the RESTful API endpoints for prediction, evaluation, and refresh.
    -   `model.py`: Core logic for data loading, model management, prediction, and evaluation.
    -   `refresh.py`: Script for automated model maintenance (can be triggered via API).
    -   `config.py`: Centralized configuration for model parameters.
    -   `download.py`: Utility for downloading stock data.
-   `test/`: Contains unit tests for the API endpoints (`test.py`).
-   `Dockerfile`: Defines the Docker image for containerized deployment.

---

*Made with ❤️ by Your Name/Team Name*