"""
This module contains the Stonky class, which is the main class for the stock prediction model.
"""
import logging
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd 
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 

from .config import (
    BATCH_SIZE,
    EPOCHS,
    FEATURES,
    INTERVAL,
    LOOKBACK,
    PERIOD,
)
from .download import download_data

logger = logging.getLogger(__name__)

class Stonky:
    """The main class for the stock prediction model."""
    def load_data(self, stock:str) -> pd.DataFrame:
        """
        Loads the data of the stock, if not present then downloads it.

        Args:
            stock (str): The ticker symbol of the stock in Yahoo Finance format.
        Returns:
            Pandas DataFrame of the related price info of the required stock.
        """
        logger.info(f"Loading stock data for {stock}")
        try:
            df = pd.read_csv(f"data/{stock}.csv")
            logger.info("Successfully loaded the data.")
        except FileNotFoundError:
            logger.warning("Data on the required stock does not exist. "
                           "Attempting to download the data.")
            try:
                df = download_data(stock, PERIOD, INTERVAL)
                logger.info(f"Data for {stock} has been downloaded.")
            except Exception as e:
                logger.exception(f"Failed to download data of {stock}: {e}")
                raise
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
        logger.info(f"Loading scaler for {stock}")
        try:
            scaler = joblib.load(f"models/{stock}_scaler.pkl")
            logger.info("Scaler loaded successfully.")
        except FileNotFoundError:
            logger.warning(f"Scaler for {stock} does not exist. Attempting to creat one.")
            try:
                df = self.load_data(stock)
                scaler = MinMaxScaler()
                scaler.fit(df)
                logger.info("Scaler created successfully.")
                joblib.dump(scaler, f"models/{stock}_scaler.pkl")
                logger.info("Scaler saved successfully.")
            except Exception as e:
                logger.critical(f"Unable to create scaler for {stock}: {e}")
                raise
        return scaler
    
    def load_stonky(self, stock:str) -> Sequential:
        """
        Loads the prediction model of the stock, if it doesn't exist the 1 is trained.

        Args:
            stock (str): The ticker symbol of the stock data in Yahoo Finance Format.
        Returns:
            The LSTM model trained on the required stock
        """
        logger.info(f"Loading prediction model for {stock}")
        try:
            model = load_model(f"models/{stock}_stonky.h5")
            logger.info("Model loaded successfully.")
        except FileNotFoundError:
            logger.warning(f"No model was trained before on the stock: {stock}. "
                           f"Attempting to create one.")
            try:
                df = self.load_data(stock)
                scaler = self.load_scaler(stock)
                scaled_df = scaler.transform(df)
                logger.debug("Scaler and data loaded successfully")
                sequences = []
                targets =[]
                for i in range(60, len(scaled_df)):
                    seq_x = scaled_df[i - LOOKBACK : i]
                    seq_y = scaled_df[i, FEATURES.index("Close")]
                    sequences.append(seq_x)
                    targets.append(seq_y)
                x, y = np.array(sequences), np.array(targets)
                x_train, _, y_train, _ = train_test_split(x, y, shuffle=True, 
                                                              test_size=0.2, random_state=42)
                logger.info(f"Training set for {stock} created successfully. Creating model...")

                model = Sequential([
                    LSTM(64, return_sequences=True, input_shape=(x.shape[1], x.shape[2])),
                    Dropout(0.2),
                    LSTM(32),
                    Dropout(0.2),
                    Dense(1)
                ])
                logger.info("Compiling model.")
                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss="mse",
                    metrics=["mae"]
                )

                logger.info("Training model...")
                model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, verbose=1)
                logger.info("Training successfull. Saving model...")
                model.save(f"models/{stock}_stonky.h5")
                logger.info("Model saved successfully and is ready to perform predictions.")
            except Exception as e:
                logger.critical(f"Unable to create a model: {e}")
                raise
        return model
    
    def predict(self, stock:str, days:int = 10) -> List[float]:
        """
        Makes prediction of a stock the specified number of days into the future.

        Args:
            stock (str): Stock ticker symbol in yahoo finance format.
            days (int): Number of days into the future to predict.
        Returns:
            A list of predicted stock prices for the future.
        """
        logger.info(f"Performing prediction on {stock} for {days} days.")
        try:
            model = self.load_stonky(stock)
            scaler = self.load_scaler(stock)
            data = self.load_data(stock)
            logger.debug("Model, data and scaler loaded successfully.")
            scaled_df = scaler.transform(data)
            future_predictions = []
            current_sequence = list(scaled_df[-LOOKBACK:])

            logger.info("Starting prediction...")
            for _ in range(days):
                # Reshape the sequence for model input
                input_for_prediction = np.array(current_sequence).reshape(1, LOOKBACK, len(FEATURES))
                
                # Predict the next scaled value
                scaled_prediction = model.predict(input_for_prediction, verbose=0)[0, 0]
                
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
            logger.info(f"Stock price has been predicted: {future_predictions}")
        except Exception as e:
            logger.exception(f"Unable to perform prediction on stock {stock}: {e}")
            raise
        return future_predictions
    
    def evaluate(self, stock:str) -> Tuple[float, float, float]:
        """
        Evaluates the prediction model of a particular stock.

        Args:
            stock (str): The ticker symbol of the stock data in Yahoo Finance Format.
        Returns:
            Mean Squared Error, Mean Absolute Error and R2 score.
        """
        logger.info(f"Evaluating performance of model for {stock}")
        try:
            model = self.load_stonky(stock)
            scaler = self.load_scaler(stock)
            df = self.load_data(stock)
            logger.debug("Model, data and scaler has been loaded.")
            scaled_df = scaler.transform(df)
            sequences = []
            targets =[]
            logger.info("Creating testing set.")
            for i in range(60, len(scaled_df)):
                seq_x = scaled_df[i - LOOKBACK : i]
                seq_y = scaled_df[i, FEATURES.index("Close")]
                sequences.append(seq_x)
                targets.append(seq_y)
            x, y = np.array(sequences), np.array(targets)
            _, x_test, _, y_test = train_test_split(x, y, shuffle=True, test_size=0.2, random_state=42)
            logger.info("Testing set has been created. Performing predictions.")
            preds_scaled = model.predict(x_test, verbose=0)

            # Inverse transform predictions and actuals to their original scale
            dummy_preds = np.zeros((len(preds_scaled), len(FEATURES)))
            dummy_preds[:, FEATURES.index("Close")] = preds_scaled.flatten()
            unscaled_preds = scaler.inverse_transform(dummy_preds)[:, FEATURES.index("Close")]

            dummy_y_test = np.zeros((len(y_test), len(FEATURES)))
            dummy_y_test[:, FEATURES.index("Close")] = y_test.flatten()
            unscaled_y_test = scaler.inverse_transform(dummy_y_test)[:, FEATURES.index("Close")]

            # Calculate and print performance metrics
            logger.info("Calculating performance...")
            mse = mean_squared_error(unscaled_y_test, unscaled_preds)
            mae = mean_absolute_error(unscaled_y_test, unscaled_preds)
            r2 = r2_score(unscaled_y_test, unscaled_preds)
            logger.info(f"Successfully evaluated the model. mse: {mse}, mae: {mae}, r2_score: {r2}")
        except Exception as e:
            logger.exception(f"Unable to evaluate the model: {e}")
            raise
        return (mse, mae, r2)
    
    def refresh_stonky(self, stock:str) -> None:
        """
        Retrains the prediction model when it has low performance.

        Args:
            stock (str): The ticker symbol of the stock data in Yahoo Finance Format.
        """
        logger.info(f"Refreshing data, model and scaler of {stock}...")
        try:
            df = self.load_data(stock)
            logger.info("Data has been refreshed.")
            scaler = MinMaxScaler()
            scaler.fit(df)
            joblib.dump(scaler, f"models/{stock}_scaler.pkl")
            logger.info("Scaler has been refreshed.")
            scaled_df = scaler.transform(df)
            sequences = []
            targets =[]
            logger.info("Preparing training set...")
            for i in range(60, len(scaled_df)):
                seq_x = scaled_df[i - LOOKBACK : i]
                seq_y = scaled_df[i, FEATURES.index("Close")]
                sequences.append(seq_x)
                targets.append(seq_y)
            x, y = np.array(sequences), np.array(targets)
            x_train, _, y_train, _ = train_test_split(x, y, shuffle=True, test_size=0.2, random_state=42)
            logger.info("Training set created successfully.")

            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(x.shape[1], x.shape[2])),
                Dropout(0.2),
                LSTM(32),
                Dropout(0.2),
                Dense(1)
            ])
            logger.info("Compiling the model...")
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss="mse",
                metrics=["mae"]
            )

            logger.info("Re-training model...")
            model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, verbose=1)
            logger.info("Saving model...")
            model.save(f"models/{stock}_stonky.h5")
            logger.info("System refreshed successfully.")
        except Exception as e:
            logger.exception(f"Unable to refresh the system: {e}")
            raise

if __name__ == "__main__":
    stonky = Stonky()
    stock_ticker = input("Enter the Stock ticker symbol in yahoo finance format:")
    num_days = int(input("Enter the number of days into the future to predict:"))
    mse, mae, r2 = stonky.evaluate(stock_ticker)
    future_predictions = stonky.predict(stock_ticker, num_days)

    print(f"Mean Squeared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R2 Score: {r2}\n")
    for i, price in enumerate(future_predictions):
        print(f"Day +{i+1}: Predicted Close Price = ${price:.2f}")

    print()
    print(type(future_predictions))

