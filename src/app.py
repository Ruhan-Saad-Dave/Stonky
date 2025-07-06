
import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from sklearn.model_selection import train_test_split

from model import Stonky

stonky = Stonky()

def predict_price(ticker, days):
    predictions = stonky.predict(ticker, days)
    
    # Create a DataFrame for the predictions
    future_dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, days + 1)]
    prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predictions})
    
    # Get historical data for the plot
    historical_data = stonky.load_data(ticker).tail(30)
    
    # Create the plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_data['Date'], y=historical_data['Close'], mode='lines', name='Historical Prices'))
    fig.add_trace(go.Scatter(x=prediction_df['Date'], y=prediction_df['Predicted Price'], mode='lines', name='Predicted Prices'))
    
    return fig, prediction_df

def evaluate_performance(ticker):
    mse, mae, r2 = stonky.evaluate(ticker)
    
    # Create a DataFrame for the metrics
    metrics_df = pd.DataFrame({'Metric': ['Mean Squared Error', 'Mean Absolute Error', 'R2 Score'], 'Value': [mse, mae, r2]})
    
    # Get historical and predicted data for the plot
    model = stonky.load_stonky(ticker)
    scaler = stonky.load_scaler(ticker)
    df = stonky.load_data(ticker)
    scaled_df = scaler.transform(df)
    sequences = []
    targets =[]
    for i in range(60, len(scaled_df)):
        seq_x = scaled_df[i - 60 : i]
        seq_y = scaled_df[i, 3]
        sequences.append(seq_x)
        targets.append(seq_y)
    X, y = np.array(sequences), np.array(targets)
    _, X_test, _, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)
    preds_scaled = model.predict(X_test)

    # Inverse transform predictions and actuals to their original scale
    dummy_preds = np.zeros((len(preds_scaled), 5))
    dummy_preds[:, 3] = preds_scaled.flatten()
    unscaled_preds = scaler.inverse_transform(dummy_preds)[:, 3]

    dummy_y_test = np.zeros((len(y_test), 5))
    dummy_y_test[:, 3] = y_test.flatten()
    unscaled_y_test = scaler.inverse_transform(dummy_y_test)[:, 3]
    
    # Create the plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(unscaled_y_test)), y=unscaled_y_test, mode='lines', name='Actual Prices'))
    fig.add_trace(go.Scatter(x=np.arange(len(unscaled_preds)), y=unscaled_preds, mode='lines', name='Predicted Prices'))
    
    return fig, metrics_df


def create_gradio_app():
    with gr.Blocks() as demo:
        with gr.Tab("Prediction"):
            with gr.Row():
                with gr.Column():
                    ticker_input = gr.Textbox(label="Stock Ticker")
                    days_input = gr.Number(label="Days to Predict", value=7)
                    predict_button = gr.Button("Predict")
                with gr.Column():
                    prediction_plot = gr.Plot()
                    prediction_table = gr.DataFrame()

        predict_button.click(
            predict_price,
            inputs=[ticker_input, days_input],
            outputs=[prediction_plot, prediction_table]
        )

        with gr.Tab("Performance"):
            with gr.Row():
                with gr.Column():
                    performance_ticker_input = gr.Textbox(label="Stock Ticker")
                    performance_button = gr.Button("Evaluate")
                with gr.Column():
                    performance_plot = gr.Plot()
                    metrics_table = gr.DataFrame()

        performance_button.click(
            evaluate_performance,
            inputs=[performance_ticker_input],
            outputs=[performance_plot, metrics_table]
        )

        with gr.Tab("Documentation"):
            gr.Markdown("""
            # Stock Prediction App

            **GitHub Repository:** [https://github.com/your-repo-link](https://github.com/your-repo-link)

            **Documentation:** [https://your-docs-link.com](https://your-docs-link.com)

            ## Ticker Symbol Cheatsheet
            """)
            
            import json
            with open("data/ticker_cheatsheet.json", "r") as f:
                cheatsheet_data = json.load(f)
            
            ticker_cheatsheet = gr.DataFrame(pd.DataFrame(list(cheatsheet_data.items()), columns=["Company", "Ticker Symbol"]))

    return demo

if __name__ == "__main__":
    app = create_gradio_app()
    app.launch()
