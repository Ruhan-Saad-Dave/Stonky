import time
from datetime import datetime, timedelta
import os
from model import Stonky

# --- Configuration ---
STOCK_TO_MONITOR = "GOOG"  # The stock ticker to monitor
CHECK_INTERVAL_SECONDS = 86400  # Check every 24 hours

# Performance Benchmarks
R2_THRESHOLD = 0.70
MAE_PERCENTAGE_THRESHOLD = 0.05  # Retrain if MAE is > 5% of the last close price

# Time-based Benchmark
REFRESH_DAYS_THRESHOLD = 30
LAST_REFRESH_DATE_FILE = "models/last_refresh_date.txt"

def get_last_refresh_date():
    """Reads the last refresh date from the tracking file."""
    if not os.path.exists(LAST_REFRESH_DATE_FILE):
        return None
    with open(LAST_REFRESH_DATE_FILE, "r") as f:
        return datetime.fromisoformat(f.read().strip())

def set_last_refresh_date():
    """Writes the current date to the tracking file."""
    os.makedirs("models", exist_ok=True)
    with open(LAST_REFRESH_DATE_FILE, "w") as f:
        f.write(datetime.now().isoformat())
    print("Updated last refresh date.")

def main():
    """Main loop to monitor model performance and trigger retraining."""
    stonky = Stonky()
    print("Starting the model refresh service...")

    while True:
        print(f"\n--- Running scheduled check at {datetime.now()} ---")
        should_refresh = False

        # 1. Performance-based check
        print("Checking model performance...")
        try:
            # Calculate the dynamic MAE threshold
            latest_data = stonky.load_data(STOCK_TO_MONITOR)
            last_close_price = latest_data['Close'].iloc[-1]
            dynamic_mae_threshold = last_close_price * MAE_PERCENTAGE_THRESHOLD
            print(f"Last close price: ${last_close_price:.2f}. Dynamic MAE Threshold set to ${dynamic_mae_threshold:.2f}")

            mse, mae, r2 = stonky.evaluate(STOCK_TO_MONITOR)
            print(f"Current Performance - MSE: {mse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
            
            if r2 < R2_THRESHOLD or mae > dynamic_mae_threshold:
                print(f"Performance threshold breached (R²: {r2:.2f} < {R2_THRESHOLD} or MAE: {mae:.2f} > {dynamic_mae_threshold:.2f}).")
                should_refresh = True
            else:
                print("Model performance is within acceptable limits.")
        except Exception as e:
            print(f"Could not evaluate model. Error: {e}. Triggering refresh as a precaution.")
            should_refresh = True

        # 2. Time-based check
        if not should_refresh:
            print("\nChecking time since last refresh...")
            last_refresh = get_last_refresh_date()
            if last_refresh is None:
                print("No previous refresh date found. Triggering initial refresh.")
                should_refresh = True
            elif (datetime.now() - last_refresh).days > REFRESH_DAYS_THRESHOLD:
                print(f"Time since last refresh exceeds {REFRESH_DAYS_THRESHOLD} days.")
                should_refresh = True
            else:
                print("Not enough time has passed for a scheduled refresh.")

        # 3. Trigger refresh if needed
        if should_refresh:
            print("\n>>> Refresh required. Retraining model... <<<")
            try:
                stonky.refresh_stonky(STOCK_TO_MONITOR)
                set_last_refresh_date()  # Update the date only after a successful refresh
                print("Model refresh completed successfully.")
            except Exception as e:
                print(f"An error occurred during model refresh: {e}")
        else:
            print("\nNo refresh required. System is healthy.")
        
        print(f"\nNext check scheduled in {CHECK_INTERVAL_SECONDS / 3600:.1f} hours.")
        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
