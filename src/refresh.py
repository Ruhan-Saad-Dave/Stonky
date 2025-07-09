"""
This module contains functions for refreshing the stock prediction model.
"""
from datetime import datetime
import logging
import os
import time

from .download import download_data
from .model import Stonky
from .config import (
    CHECK_INTERVAL_SECONDS,
    MAE_PERCENTAGE_THRESHOLD,
    PERIOD,
    INTERVAL,
    R2_THRESHOLD,
)

logger = logging.getLogger(__name__)

def refresh():
    """
    Refreshes a model every month or if the model performance is low (checked everyday).
    """
    try:
        logger.info("Starting refresh module...")
        if not os.path.exists("data"):
            logger.error("The folder 'data' does not exist. File an issue on github.")
            return
        
        stonky = Stonky()
        logger.info("Starting the model refreshing service...")

        while True:
            logger.info("--- Running scheduled check at %s ---", datetime.now())
            stocks = []
            for file_name in os.listdir("data"):
                file_path = os.path.join("data", file_name)
                if os.path.isfile(file_path):
                    stock_name = file_name[:-4]
                    stocks.append(stock_name) # remove the .csv extention
                    download_data(stock_name, PERIOD, INTERVAL) # Refresh data daily
            logger.info("All stock data has been refreshed.")
            
            # Check if 1st day of the month
            today = datetime.today()
            first_day = (today.day == 1)

            for stock in stocks:
                should_refresh = False
                if not first_day:
                    try:
                        latest_data = stonky.load_data(stock)
                        last_close_price = latest_data['Close'].iloc[-1]
                        dynamic_mae_threshold = last_close_price * MAE_PERCENTAGE_THRESHOLD
                        logger.debug("Last close price: $%.2f. "
                                   "Dynamic MAE Threshold set to $%.2f", last_close_price, dynamic_mae_threshold)

                        mse, mae, r2 = stonky.evaluate(stock)
                        logger.debug("Current Performance - MSE: %.2f, MAE: %.2f, R²: %.2f", mse, mae, r2)
                        
                        if r2 < R2_THRESHOLD or mae > dynamic_mae_threshold:
                            logger.info("Performance threshold breached (R²: %.2f < %s "
                                       "or MAE: %.2f > %.2f).", r2, R2_THRESHOLD, mae, dynamic_mae_threshold)
                            should_refresh = True
                        else:
                            logger.info("Model performance is within acceptable limits.")
                    except Exception as e:
                        logger.exception("Could not evaluate model. Error: %s. Triggering refresh as a precaution.", e)
                        should_refresh = True
                
                if first_day or should_refresh:
                    logger.info(">>> Refresh required. Retraining model for %s... <<<", stock)
                    try:
                        stonky.refresh_stonky(stock)
                        logger.info("Model refresh completed successfully.")
                    except Exception as e:
                        logger.exception("An error occurred during model refresh: %s", e)
                else:
                    logger.info("No refresh required. System is healthy.")


            logger.info("Next check scheduled in %.1f hours.", CHECK_INTERVAL_SECONDS / 3600)
            time.sleep(CHECK_INTERVAL_SECONDS)
    except Exception as e:
        logger.critical("Something went wrong in the refresh system: %s", e)
        raise
