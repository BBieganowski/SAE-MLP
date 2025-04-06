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
TARGET_TABLE = "candles_5m_labels"
ASSETS = ["ETHUSDT", "BTCUSDT", "LTCUSDT"]



WINDOW_DURATION_MINUTES = 15  
BARRIER_LAMBDA = 0.001  



if WINDOW_DURATION_MINUTES % 5 != 0:
    raise ValueError("WINDOW_DURATION_MINUTES must be a multiple of 5.")
WINDOW_PERIODS = WINDOW_DURATION_MINUTES // 5




def calculate_triple_barrier_labels(
    df: pd.DataFrame, window_periods: int, barrier_lambda: float
):
    """
    Calculates triple barrier labels for a given price series DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with 'high', 'low', 'close' columns, indexed by 'open_time'.
        window_periods (int): Number of periods (candles) to look forward.
        barrier_lambda (float): Percentage difference for upper/lower barriers.

    Returns:
        tuple[pd.Series, pd.Series]: A tuple containing:
            - labels (pd.Series): The calculated label (1, -1, or 0) for each time step.
            - hit_times (pd.Series): The timestamp when a horizontal barrier was hit, pd.NaT otherwise.
    """
    n = len(df)

    
    entry_prices = df["close"]
    upper_barrier = entry_prices * (1 + barrier_lambda)
    lower_barrier = entry_prices * (1 - barrier_lambda)

    
    labels = pd.Series(0, index=df.index, dtype=int)
    hit_times = pd.Series(pd.NaT, index=df.index)

    
    active = pd.Series(True, index=df.index)

    logging.info(
        f"Calculating labels with window_periods={window_periods}, lambda={barrier_lambda}..."
    )

    for k in range(1, window_periods + 1):
        if active.sum() == 0:
            logging.info(f"All labels determined by period {k-1}. Stopping early.")
            break  

        
        
        future_indices = df.index[k:n]
        current_time_k = df.index.shift(-k)  

        
        future_highs_k = df["high"].shift(-k).loc[active.index[active]]
        future_lows_k = df["low"].shift(-k).loc[active.index[active]]

        
        active_upper_barrier = upper_barrier.loc[active.index[active]]
        active_lower_barrier = lower_barrier.loc[active.index[active]]
        active_time_k = current_time_k[active]

        
        
        hit_upper_mask = future_highs_k >= active_upper_barrier
        upper_hit_indices = active.index[active][hit_upper_mask]

        if not upper_hit_indices.empty:
            labels.loc[upper_hit_indices] = 1
            
            hit_times.loc[upper_hit_indices] = active_time_k[hit_upper_mask]
            active.loc[upper_hit_indices] = False  

        
        
        still_active_indices = active.index[active]
        if still_active_indices.empty:
            if active.sum() == 0:  
                logging.info(f"All labels determined by period {k}. Stopping early.")
                break
            else:
                continue  

        
        future_lows_k_active = future_lows_k.loc[still_active_indices]
        active_lower_barrier_now = lower_barrier.loc[still_active_indices]
        active_time_k_now = current_time_k[
            active
        ]  

        hit_lower_mask = future_lows_k_active <= active_lower_barrier_now
        lower_hit_indices = still_active_indices[hit_lower_mask]

        if not lower_hit_indices.empty:
            labels.loc[lower_hit_indices] = -1
            hit_times.loc[lower_hit_indices] = active_time_k_now[hit_lower_mask]
            active.loc[lower_hit_indices] = False  

        if k % (window_periods // 10 + 1) == 0:  
            logging.info(
                f"Processed period {k}/{window_periods}. Active rows remaining: {active.sum()}"
            )

    logging.info(
        f"Finished label calculation. {len(labels[labels==1])} [+1], {len(labels[labels==-1])} [-1], {len(labels[labels==0])} [0]."
    )
    return labels, hit_times





def main():
    try:
        con = duckdb.connect(DB_PATH)
        logging.info(f"Connected to database: {DB_PATH}")

        
        con.execute(
            f"""
        CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
            open_time TIMESTAMP,
            symbol VARCHAR,
            label INTEGER,
            window_minutes INTEGER,
            barrier_lambda DOUBLE,
            target_hit_time TIMESTAMP, 
            PRIMARY KEY (symbol, open_time, window_minutes, barrier_lambda)
        );
        """
        )
        logging.info(f"Ensured target table '{TARGET_TABLE}' exists.")

        for asset in ASSETS:
            logging.info(f"--- Processing asset: {asset} ---")

            
            logging.info(f"Fetching HLC prices for {asset} from '{SOURCE_TABLE}'...")
            df = con.execute(
                f"SELECT open_time, high, low, close FROM {SOURCE_TABLE} WHERE symbol = ? ORDER BY open_time",
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

            df[["high", "low", "close"]] = df[["high", "low", "close"]].astype(
                float
            )  
            logging.info(f"Loaded {len(df)} data points for {asset}.")

            
            labels, hit_times = calculate_triple_barrier_labels(
                df, WINDOW_PERIODS, BARRIER_LAMBDA
            )

            
            result_df = pd.DataFrame(
                {
                    
                    "symbol": asset,
                    "label": labels,
                    "window_minutes": WINDOW_DURATION_MINUTES,
                    "barrier_lambda": BARRIER_LAMBDA,
                    "target_hit_time": hit_times,
                }
            ).reset_index()  

            
            
            
            

            if result_df.empty:
                logging.warning(
                    f"Resulting labeled series for {asset} is empty. Skipping insertion."
                )
                continue

            
            logging.info(
                f"Inserting/updating {len(result_df)} rows into '{TARGET_TABLE}' for {asset}..."
            )
            
            con.register("result_df_temp", result_df)
            
            con.execute(
                f"""
                INSERT INTO {TARGET_TABLE}
                SELECT open_time, symbol, label, window_minutes, barrier_lambda, target_hit_time
                FROM result_df_temp
                ON CONFLICT (symbol, open_time, window_minutes, barrier_lambda) DO UPDATE SET
                    label = excluded.label,
                    target_hit_time = excluded.target_hit_time;
            """
            )
            con.unregister("result_df_temp")
            logging.info(
                f"Successfully inserted/updated labels for {asset} with window={WINDOW_DURATION_MINUTES}min, lambda={BARRIER_LAMBDA}."
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
    logging.info("Triple barrier labeling script finished.")
