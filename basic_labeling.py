import pandas as pd
import numpy as np
import duckdb
import logging
from datetime import timedelta

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

DB_PATH = "database/crypto_data.db"
SOURCE_TABLE = "candles_5m"
TARGET_TABLE = "candles_5m_basic_labels"
ASSETS = ["BTCUSDT", "ES", "M6E"]

# N minutes for future return/direction calculation
WINDOW_DURATION_MINUTES = 30

if WINDOW_DURATION_MINUTES % 5 != 0:
    raise ValueError("WINDOW_DURATION_MINUTES must be a multiple of 5.")
WINDOW_PERIODS = WINDOW_DURATION_MINUTES // 5

def calculate_future_return(df: pd.DataFrame, window_periods: int):
    """
    Calculates the percentage return over the next N periods (window_periods).

    Args:
        df (pd.DataFrame): DataFrame with 'close' column, indexed by 'open_time'.
        window_periods (int): Number of periods (candles) to look forward.

    Returns:
        pd.Series: The future percentage return for each time step.
    """
    logging.info(f"Calculating future return for {window_periods} periods...")
    future_prices = df["close"].shift(-window_periods)
    returns = (future_prices - df["close"]) / df["close"]
    logging.info(f"Finished future return calculation.")
    return returns

def calculate_future_direction(df: pd.DataFrame, window_periods: int):
    """
    Calculates the direction of the return over the next N periods.
    1 for positive return, -1 for negative return, 0 for no change or NaN.

    Args:
        df (pd.DataFrame): DataFrame with 'close' column, indexed by 'open_time'.
        window_periods (int): Number of periods (candles) to look forward.

    Returns:
        pd.Series: The direction of future return (1, -1, or 0).
    """
    logging.info(f"Calculating future direction for {window_periods} periods...")
    future_prices = df["close"].shift(-window_periods)
    price_diff = future_prices - df["close"]
    direction = np.sign(price_diff).fillna(0).astype(int)
    logging.info(f"Finished future direction calculation.")
    return direction

def main():
    try:
        con = duckdb.connect(DB_PATH)
        logging.info(f"Connected to database: {DB_PATH}")

        con.execute(
            f"""
        CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
            open_time TIMESTAMP,
            symbol VARCHAR,
            window_minutes INTEGER,
            future_return DOUBLE,
            future_direction INTEGER,
            PRIMARY KEY (symbol, open_time, window_minutes)
        );
        """
        )
        logging.info(f"Ensured target table '{TARGET_TABLE}' exists.")

        for asset in ASSETS:
            logging.info(f"--- Processing asset: {asset} ---")

            logging.info(f"Fetching close prices for {asset} from '{SOURCE_TABLE}'...")
            df = con.execute(
                f"SELECT open_time, close FROM {SOURCE_TABLE} WHERE symbol = ? ORDER BY open_time",
                [asset],
            ).fetchdf()

            if df.empty:
                logging.warning(f"No data found for asset {asset}. Skipping.")
                continue

            df = df.set_index("open_time")
            df.index = pd.to_datetime(df.index)
            inferred_freq = pd.infer_freq(df.index)
            if inferred_freq:
                df = df.asfreq(inferred_freq)
                logging.info(f"Inferred and set index frequency to: {inferred_freq}")
            else:
                logging.warning(
                    "Could not infer index frequency. Attempting to set to '5min'. Data might be irregular."
                )
                try:
                    df = df.asfreq("5min")
                except ValueError as e:
                    logging.error(
                        f"Failed to set frequency to '5min': {e}. Data might be missing timestamps. Skipping asset {asset}."
                    )
                    continue

            df["close"] = df["close"].astype(float)
            logging.info(f"Loaded {len(df)} data points for {asset}.")

            future_returns = calculate_future_return(df, WINDOW_PERIODS)
            future_directions = calculate_future_direction(df, WINDOW_PERIODS)

            result_df = pd.DataFrame(
                {
                    "symbol": asset,
                    "window_minutes": WINDOW_DURATION_MINUTES,
                    "future_return": future_returns,
                    "future_direction": future_directions,
                }
            ).reset_index()
            
            result_df.rename(columns={'index': 'open_time'}, inplace=True) # ensure open_time is a column

            if result_df.empty:
                logging.warning(
                    f"Resulting labeled series for {asset} is empty. Skipping insertion."
                )
                continue
            
            # Drop rows where labels could not be calculated (typically at the end of the series)
            result_df.dropna(subset=['future_return', 'future_direction'], inplace=True)
            if result_df.empty:
                logging.warning(
                    f"Resulting labeled series for {asset} is empty after dropping NaNs. Skipping insertion."
                )
                continue

            logging.info(
                f"Inserting/updating {len(result_df)} rows into '{TARGET_TABLE}' for {asset} with window={WINDOW_DURATION_MINUTES}min..."
            )
            
            con.register("result_df_temp", result_df)
            con.execute(
                f"""
                INSERT INTO {TARGET_TABLE}
                SELECT open_time, symbol, window_minutes, future_return, future_direction
                FROM result_df_temp
                ON CONFLICT (symbol, open_time, window_minutes) DO UPDATE SET
                    future_return = excluded.future_return,
                    future_direction = excluded.future_direction;
            """
            )
            con.unregister("result_df_temp")
            logging.info(
                f"Successfully inserted/updated basic labels for {asset} with window={WINDOW_DURATION_MINUTES}min."
            )

    except ValueError as ve:
        logging.error(f"Configuration Error: {ve}")
    except Exception as e:
        logging.error(f"An error occurred during the process: {e}", exc_info=True)
    finally:
        if "con" in locals() and con:
            con.close()
            logging.info("Database connection closed.")

if __name__ == "__main__":
    main()
    logging.info("Basic labeling script finished.") 