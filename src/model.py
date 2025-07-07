from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout 
from tensorflow.keras.optimizers import Adam 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, List
import pandas as pd 
import numpy as np 
import joblib 

from src.download import download_data
from src.config import *

class Stonky:
    def load_data(self, stock:str) -> pd.DataFrame:
        """
        Loads the data of the stock, if not present then downloads it.

        Args:
            stock (str): The ticker symbol of the stock in Yahoo Finance format.
        Returns:
            Pandas DataFrame of the related price info of the required stock.
        """
        try:
            df = pd.read_csv(f"data/{stock}.csv")
        except FileNotFoundError:
            df = download_data(stock, PERIOD, INTERVAL)
        df = df[FEATURES]
        return df 
    
    def load_scaler(self, stock:str) -> MinMaxScaler:
        """
        Loads the MinMaxScaler to normilize the data between 0 and 1, if not scaler is found then 1 is created.

        Args:
            stock (str): The ticker symbol of the stock data in Yahoo Finance Format.
        Returns:
            A MinMaxScaler scaled on the corresponding stock data.
        """
        try:
            scaler = joblib.load(f"models/{stock}_scaler.pkl")
        except FileNotFoundError:
            df = self.load_data(stock)
            scaler = MinMaxScaler()
            scaler.fit(df)
            joblib.dump(scaler, f"models/{stock}_scaler.pkl")
        return scaler
    
    def load_stonky(self, stock:str) -> Sequential:
        """
        Loads the prediction model of the stock, if it doesn't exist the 1 is trained.

        Args:
            stock (str): The ticker symbol of the stock data in Yahoo Finance Format.
        Returns:
            The LSTM model trained on the required stock
        """
        try:
            model = load_model(f"models/{stock}_stonky.h5")
        except FileNotFoundError:
            print(f"No model was trained before on the stock: {stock}")
            df = self.load_data(stock)
            scaler = self.load_scaler(stock)
            scaled_df = scaler.transform(df)
            sequences = []
            targets =[]
            for i in range(60, len(scaled_df)):
                seq_x = scaled_df[i - LOOKBACK : i]
                seq_y = scaled_df[i, FEATURES.index("Close")]
                sequences.append(seq_x)
                targets.append(seq_y)
            X, y = np.array(sequences), np.array(targets)
            X_train, _, y_train, _ = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)

            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                Dropout(0.2),
                LSTM(32),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss="mse",
                metrics=["mae"]
            )

            print("Training model...")
            model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, verbose=1)
            print("\nSaving model...")
            model.save(f"models/{stock}_stonky.h5")
        return model
    
    def predict(self, stock:str, days:int = 10) -> List[int]:
        """
        pass
        """
        model = self.load_stonky(stock)
        scaler = self.load_scaler(stock)
        data = self.load_data(stock)
        scaled_df = scaler.transform(data)
        future_predictions = []
        current_sequence = list(scaled_df[-LOOKBACK:])

        for _ in range(days):
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
        return future_predictions
    
    def evaluate(self, stock:str) -> Tuple[int]:
        """
        Evaluates the prediction model of a particular stock.

        Args:
            stock (str): The ticker symbol of the stock data in Yahoo Finance Format.
        Returns:
            Mean Squared Error, Mean Absolute Error and R2 score.
        """
        model = self.load_stonky(stock)
        scaler = self.load_scaler(stock)
        df = self.load_data(stock)
        scaled_df = scaler.transform(df)
        sequences = []
        targets =[]
        for i in range(60, len(scaled_df)):
            seq_x = scaled_df[i - LOOKBACK : i]
            seq_y = scaled_df[i, FEATURES.index("Close")]
            sequences.append(seq_x)
            targets.append(seq_y)
        X, y = np.array(sequences), np.array(targets)
        _, X_test, _, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)
        preds_scaled = model.predict(X_test)

        # Inverse transform predictions and actuals to their original scale
        dummy_preds = np.zeros((len(preds_scaled), len(FEATURES)))
        dummy_preds[:, FEATURES.index("Close")] = preds_scaled.flatten()
        unscaled_preds = scaler.inverse_transform(dummy_preds)[:, FEATURES.index("Close")]

        dummy_y_test = np.zeros((len(y_test), len(FEATURES)))
        dummy_y_test[:, FEATURES.index("Close")] = y_test.flatten()
        unscaled_y_test = scaler.inverse_transform(dummy_y_test)[:, FEATURES.index("Close")]

        # Calculate and print performance metrics
        mse = mean_squared_error(unscaled_y_test, unscaled_preds)
        mae = mean_absolute_error(unscaled_y_test, unscaled_preds)
        r2 = r2_score(unscaled_y_test, unscaled_preds)
        return (mse, mae, r2)
    
    def refresh_stonky(self, stock:str) -> None:
        """
        Retrains the prediction model when it has low performance.

        Args:
            stock (str): The ticker symbol of the stock data in Yahoo Finance Format.
        """
        df = self.load_data(stock)
        scaler = MinMaxScaler()
        scaler.fit(df)
        joblib.dump(scaler, f"models/{stock}_scaler.pkl")
        scaled_df = scaler.transform(df)
        sequences = []
        targets =[]
        for i in range(60, len(scaled_df)):
            seq_x = scaled_df[i - LOOKBACK : i]
            seq_y = scaled_df[i, FEATURES.index("Close")]
            sequences.append(seq_x)
            targets.append(seq_y)
        X, y = np.array(sequences), np.array(targets)
        X_train, _, y_train, _ = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"]
        )

        print("Re-training model...")
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
        print("\nSaving model...")
        model.save(f"models/{stock}_stonky.h5")

if __name__ == "__main__":
    stonky = Stonky()
    stock = input("Enter the Stock ticker symbol in yahoo finance format:")
    days = int(input("Enter the number of days into the future to predict:"))
    mse, mae, r2 = stonky.evaluate(stock)
    future_predictions = stonky.predict(stock, days)

    print(f"Mean Squeared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R2 Score: {r2}\n")
    for i, price in enumerate(future_predictions):
        print(f"Day +{i+1}: Predicted Close Price = ${price:.2f}")

    print()
    print(type(future_predictions))
