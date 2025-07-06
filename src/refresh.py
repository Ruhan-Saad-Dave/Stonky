from datetime import datetime
import time
import os

from download import download_data
from model import Stonky
from config import *

def refresh():
    """
    Refreshes a model every month or if the model performance is low (checked everyday).
    """
    if not os.path.exists("data"):
        print(f"The folder 'data' does not exist. File an issue on github.")
        return
    
    stonky = Stonky()
    print("Starting the model refreshing service...")

    while True:
        print(f"\n--- Running scheduled check at {datetime.now()} ---")
        stocks = []
        for file_name in os.listdir("data"):
            file_path = os.path.join("data", file_name)
            if os.path.isfile(file_path):
                stocks.append(file_name[:-4]) # remove the .csv extention
                download_data(stock, PERIOD, INTERVAL) # Refresh data daily
        
        # Check if 1st day of the month
        today = datetime.today()
        if today.day == 1:
            first_day = True
        else:
            first_day = False

        for stock in stocks:
            if not first_day:
                should_refresh = False
                try:
                    latest_data = stonky.load_data(stock)
                    last_close_price = latest_data['Close'].iloc[-1]
                    dynamic_mae_threshold = last_close_price * MAE_PERCENTAGE_THRESHOLD
                    print(f"Last close price: ${last_close_price:.2f}. Dynamic MAE Threshold set to ${dynamic_mae_threshold:.2f}")

                    mse, mae, r2 = stonky.evaluate(stock)
                    print(f"Current Performance - MSE: {mse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
                    
                    if r2 < R2_THRESHOLD or mae > dynamic_mae_threshold:
                        print(f"Performance threshold breached (R²: {r2:.2f} < {R2_THRESHOLD} or MAE: {mae:.2f} > {dynamic_mae_threshold:.2f}).")
                        should_refresh = True
                    else:
                        print("Model performance is within acceptable limits.")
                except Exception as e:
                    print(f"Could not evaluate model. Error: {e}. Triggering refresh as a precaution.")
                    should_refresh = True
            
            if first_day or should_refresh:
                print(f"\n>>> Refresh required. Retraining model for {stock}... <<<")
                try:
                    stonky.refresh_stonky(stock)
                    print("Model refresh completed successfully.")
                except Exception as e:
                    print(f"An error occurred during model refresh: {e}")
            else:
                print("\nNo refresh required. System is healthy.")


        print(f"\nNext check scheduled in {CHECK_INTERVAL_SECONDS / 3600:.1f} hours.")
        time.sleep(CHECK_INTERVAL_SECONDS)
