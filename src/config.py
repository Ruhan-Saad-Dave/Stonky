"""
This module contains the configuration parameters for the Stonky application.
"""
PERIOD = "1y"
INTERVAL = "1d"
LOOKBACK = 50  #Number of days to look at in each training step
EPOCHS = 10
BATCH_SIZE = 32
FEATURES = ["Open", "High", "Low", "Close", "Volume"]
CHECK_INTERVAL_SECONDS = 86400  # Check every 24 hours
R2_THRESHOLD = 0.70
MAE_PERCENTAGE_THRESHOLD = 0.05  # Retrain if MAE is > 5% of the last close price
REFRESH_DAYS_THRESHOLD = 30
LOG_FILE = "stonky.log"
