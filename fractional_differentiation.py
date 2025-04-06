import pandas as pd
import numpy as np
import duckdb
import optuna
from statsmodels.tsa.stattools import adfuller
import logging
import sys


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


DB_PATH = "database/crypto_data.db"
SOURCE_TABLE = "candles_5m"
TARGET_TABLE = "candles_5m_fracdiff"
ASSETS = ["ETHUSDT", "BTCUSDT", "LTCUSDT"]
ADF_PVALUE_THRESHOLD = 0.01  
OPTUNA_N_TRIALS = 5  
FRACDIFF_THRESHOLD = 1e-4  
MAX_D_ORDER = 1.5  




def get_weights_fixed_window(d, threshold):
    """Calculates weights for fixed window fractional differentiation."""
    w = [1.0]
    k = 1
    while True:
        w_ = -w[-1] * (d - k + 1) / k
        if abs(w_) < threshold:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1])  


def frac_diff_fixed_window(series, d, threshold=FRACDIFF_THRESHOLD):
    """
    Computes fractionally differentiated series using fixed window method.
    Ensures the output series length matches the input series length.
    Initial values where differentiation is not possible will be NaN.

    Args:
        series (pd.Series): Input time series.
        d (float): Differentiation order.
        threshold (float): Threshold to stop weight calculation.

    Returns:
        pd.Series: Fractionally differentiated series.
    """
    weights = get_weights_fixed_window(d, threshold)
    w_len = len(weights)
    n = len(series)

    
    diff_series = pd.Series(index=series.index, dtype="float64")

    
    for i in range(w_len - 1, n):
        window = series.iloc[i - w_len + 1 : i + 1]
        if len(window) == w_len:  
            diff_series.iloc[i] = np.dot(weights, window)
        

    return diff_series





def check_stationarity(series, threshold=ADF_PVALUE_THRESHOLD):
    """Performs ADF test and checks if p-value is below threshold."""
    series = series.dropna()  
    if len(series) < 20:  
        logging.warning("Series too short after dropping NaNs for ADF test.")
        return False, 1.0  
    try:
        result = adfuller(series)
        p_value = result[1]
        return p_value <= threshold, p_value
    except Exception as e:
        logging.error(f"ADF test failed: {e}")
        return False, 1.0  





def objective(trial, series):
    """
    Optuna objective function.
    Minimizes d while ensuring stationarity (p-value <= threshold).
    If stationary, returns d. If not, returns a penalty (e.g., MAX_D_ORDER + p-value).
    """
    d = trial.suggest_float("d", 0, MAX_D_ORDER)

    diff_series = frac_diff_fixed_window(series, d)
    is_stationary, p_value = check_stationarity(diff_series)

    if is_stationary:
        
        
        trial.set_user_attr("p_value", p_value)  
        return d
    else:
        
        
        return MAX_D_ORDER + p_value





def main():
    try:
        con = duckdb.connect(DB_PATH)
        logging.info(f"Connected to database: {DB_PATH}")

        
        con.execute(
            f"""
        CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
            open_time TIMESTAMP,
            symbol VARCHAR,
            close_fracdiff DOUBLE,
            d_order DOUBLE,
            PRIMARY KEY (symbol, open_time)
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
            price_series = df["close"].astype(float)  
            logging.info(f"Loaded {len(price_series)} data points for {asset}.")

            
            is_orig_stationary, orig_p_value = check_stationarity(price_series)
            if is_orig_stationary:
                logging.info(
                    f"Original series for {asset} is already stationary (p={orig_p_value:.4f}). Setting d=0."
                )
                best_d = 0.0
                frac_diff_result = price_series  
            else:
                logging.info(
                    f"Original series for {asset} not stationary (p={orig_p_value:.4f}). Running Optuna..."
                )
                
                study = optuna.create_study(direction="minimize")
                
                study.optimize(
                    lambda trial: objective(trial, price_series),
                    n_trials=OPTUNA_N_TRIALS,
                    show_progress_bar=True,
                )

                best_trial = study.best_trial
                best_d = (
                    best_trial.value
                )  
                best_p_value = best_trial.user_attrs.get("p_value", None)

                
                if best_d > MAX_D_ORDER:  
                    logging.error(
                        f"Optuna failed to find a value of d for {asset} that makes the series stationary after {OPTUNA_N_TRIALS} trials. Last p-value tried might have been {best_p_value}. Skipping insertion."
                    )
                    continue

                logging.info(
                    f"Optuna finished for {asset}. Best d: {best_d:.4f} (achieved p-value: {best_p_value:.4f}) "
                )

                
                logging.info(
                    f"Applying fractional differentiation with d={best_d:.4f}..."
                )
                frac_diff_result = frac_diff_fixed_window(price_series, best_d)

            
            result_df = pd.DataFrame(
                {
                    "open_time": frac_diff_result.index,
                    "symbol": asset,
                    "close_fracdiff": frac_diff_result.values,
                    "d_order": best_d,
                }
            )

            
            result_df = result_df.dropna(subset=["close_fracdiff"])

            if result_df.empty:
                logging.warning(
                    f"Resulting differentiated series for {asset} is empty after dropping NaNs. Skipping insertion."
                )
                continue

            
            logging.info(
                f"Inserting {len(result_df)} rows into '{TARGET_TABLE}' for {asset}..."
            )
            con.register("result_df_temp", result_df)
            con.execute(
                f"""
                INSERT INTO {TARGET_TABLE}
                SELECT open_time, symbol, close_fracdiff, d_order
                FROM result_df_temp
                ON CONFLICT (symbol, open_time) DO UPDATE SET
                    close_fracdiff = excluded.close_fracdiff,
                    d_order = excluded.d_order;
            """
            )  
            con.unregister("result_df_temp")
            logging.info(f"Successfully inserted/updated data for {asset}.")

    except Exception as e:
        logging.error(f"An error occurred during the process: {e}", exc_info=True)
    finally:
        if "con" in locals() and con:
            con.close()
            logging.info("Database connection closed.")


if __name__ == "__main__":
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    main()
    logging.info("Fractional differentiation script finished.")
