import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# --- Configuration ---
LOOKBACK = 60
FEATURES = ["Open", "High", "Low", "Close", "Volume"]
DAYS_TO_PREDICT = 10
MODEL_PATH = "stock_predictor.h5"
SCALER_PATH = "scaler.gz"
DATA_PATH = "data/GOOG.csv"

# --- Load Model and Scaler ---
print("Loading model and scaler...")
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# --- Load and Prepare Data ---
print("Loading and preparing data...")
df = pd.read_csv(DATA_PATH)
df = df[FEATURES]

# Scale the data using the loaded scaler
scaled_df = scaler.transform(df)

# --- Multi-Day Future Prediction ---
print(f"Predicting stock prices for the next {DAYS_TO_PREDICT} days...")
future_predictions = []
# Use the last sequence from the original scaled data as the starting point
current_sequence = list(scaled_df[-LOOKBACK:])

for _ in range(DAYS_TO_PREDICT):
    # Reshape the sequence for model input
    input_for_prediction = np.array(current_sequence).reshape(1, LOOKBACK, len(FEATURES))
    
    # Predict the next scaled value
    scaled_prediction = model.predict(input_for_prediction)[0, 0]
    
    # Create a new row for the predicted day (using the prediction for all price features)
    new_row = np.zeros(len(FEATURES))
    new_row[FEATURES.index("Open")] = scaled_prediction
    new_row[FEATURES.index("High")] = scaled_prediction
    new_row[FEATURES.index("Low")] = scaled_prediction
    new_row[FEATURES.index("Close")] = scaled_prediction
    # Volume can be left as 0 or estimated differently if needed
    
    # Inverse transform the 'Close' price to its original scale for reporting
    dummy_pred = np.zeros((1, len(FEATURES)))
    dummy_pred[0, FEATURES.index("Close")] = scaled_prediction
    unscaled_future_price = scaler.inverse_transform(dummy_pred)[0, FEATURES.index("Close")]
    future_predictions.append(unscaled_future_price)
    
    # Append the new predicted row to the sequence
    current_sequence.append(new_row)
    # Remove the oldest row to maintain the lookback window
    current_sequence.pop(0)

# Print the final predictions
for i, price in enumerate(future_predictions):
    print(f"Day +{i+1}: Predicted Close Price = ${price:.2f}")
