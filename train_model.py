import pandas as pd
import numpy as np
import duckdb
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from model import create_ae_mlp  


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


DB_PATH = "database/crypto_data.db"
FRACDIFF_TABLE = "candles_5m_fracdiff"
LABEL_TABLE = "candles_5m_labels"
ALL_ASSETS = ["ETHUSDT", "BTCUSDT", "LTCUSDT"]


WINDOW_MINUTES_PARAM = 15  
BARRIER_LAMBDA_PARAM = 0.001  


TRAIN_SPLIT_RATIO = 0.8  



HIDDEN_UNITS = [128, 64, 64, 32]  
DROPOUT_RATES = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]  
LABEL_SMOOTHING = 0.01
LEARNING_RATE = 1e-3
EPOCHS = 50  
BATCH_SIZE = 256  
MODEL_SAVE_DIR = "trained_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


def load_data(con):
    """Loads features (FracDiff) and labels from DuckDB."""
    logging.info("Loading fractional differentiation data...")
    
    fd_query = f"SELECT open_time, symbol, close_fracdiff FROM {FRACDIFF_TABLE}"
    all_fd_data = con.execute(fd_query).fetchdf()

    if all_fd_data.empty:
        raise ValueError(
            f"No data found in {FRACDIFF_TABLE}. Run fractional_differentiation.py first."
        )

    
    features_df = all_fd_data.pivot(
        index="open_time", columns="symbol", values="close_fracdiff"
    )
    features_df.columns = [f"{col}_fd" for col in features_df.columns]  
    logging.info(
        f"Loaded {len(features_df)} rows of feature data with columns: {features_df.columns.tolist()}"
    )

    logging.info(
        f"Loading triple barrier labels (window={WINDOW_MINUTES_PARAM}, lambda={BARRIER_LAMBDA_PARAM})..."
    )
    
    label_query = f"SELECT open_time, symbol, label FROM {LABEL_TABLE} WHERE window_minutes = ? AND barrier_lambda = ?"
    all_label_data = con.execute(
        label_query, [WINDOW_MINUTES_PARAM, BARRIER_LAMBDA_PARAM]
    ).fetchdf()

    if all_label_data.empty:
        raise ValueError(
            f"No labels found in {LABEL_TABLE} for window={WINDOW_MINUTES_PARAM}, lambda={BARRIER_LAMBDA_PARAM}. Run triple_barrier_labeling.py first."
        )
    logging.info(f"Loaded {len(all_label_data)} rows of label data.")

    return features_df, all_label_data


def prepare_asset_data(features_df, all_label_data, target_asset):
    """Prepares features (X) and targets (y) for a specific target asset."""
    logging.info(f"Preparing data for target asset: {target_asset}")

    
    asset_labels = all_label_data[all_label_data["symbol"] == target_asset][
        ["open_time", "label"]
    ]
    asset_labels = asset_labels.set_index("open_time")

    
    merged_df = features_df.join(
        asset_labels, how="inner"
    )  

    
    initial_rows = len(merged_df)
    merged_df = merged_df.dropna()
    rows_after_na = len(merged_df)
    if initial_rows > rows_after_na:
        logging.warning(
            f"Dropped {initial_rows - rows_after_na} rows due to NaNs after merging features and labels for {target_asset}."
        )

    if merged_df.empty:
        logging.warning(
            f"No data remaining for {target_asset} after merging and dropping NaNs. Skipping."
        )
        return None, None, None

    
    labeled_df = merged_df[merged_df["label"] != 0].copy()
    logging.info(
        f"Filtered out label 0. Kept {len(labeled_df)} rows for {target_asset}."
    )

    if labeled_df.empty:
        logging.warning(f"No non-zero labels found for {target_asset}. Skipping.")
        return None, None, None

    
    feature_cols = [col for col in labeled_df.columns if col.endswith("_fd")]
    X = labeled_df[feature_cols]

    
    y_action = labeled_df["label"].map({1: 1, -1: 0}).astype(int)

    
    y_decoder = X.copy()

    logging.info(
        f"Prepared data shapes for {target_asset}: X={X.shape}, y_action={y_action.shape}, y_decoder={y_decoder.shape}"
    )

    return X, y_action, y_decoder


def main():
    try:
        con = duckdb.connect(DB_PATH)
        logging.info(f"Connected to database: {DB_PATH}")
        features_df, all_label_data = load_data(con)
        con.close()  
        logging.info("Database connection closed.")

    except (ValueError, Exception) as e:
        logging.error(f"Failed to load data: {e}", exc_info=True)
        if "con" in locals() and con:
            con.close()
        return

    for target_asset in ALL_ASSETS:
        logging.info(
            f"--- Starting training process for target asset: {target_asset} ---"
        )
        X, y_action, y_decoder = prepare_asset_data(
            features_df, all_label_data, target_asset
        )

        if X is None:
            continue  

        
        split_index = int(len(X) * TRAIN_SPLIT_RATIO)
        if split_index == 0 or split_index == len(X):
            logging.error(
                f"Cannot perform train/val split for {target_asset}. Insufficient data or invalid split ratio. Need at least 2 samples."
            )
            continue

        X_train, X_val = X.iloc[:split_index], X.iloc[split_index:]
        y_action_train, y_action_val = (
            y_action.iloc[:split_index],
            y_action.iloc[split_index:],
        )
        y_decoder_train, y_decoder_val = (
            y_decoder.iloc[:split_index],
            y_decoder.iloc[split_index:],
        )

        logging.info(
            f"Data split for {target_asset}: Train={len(X_train)}, Val={len(X_val)}"
        )

        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        y_decoder_train_scaled = scaler.transform(y_decoder_train)
        y_decoder_val_scaled = scaler.transform(y_decoder_val)
        logging.info(f"Features scaled for {target_asset}.")

        
        num_columns = X_train_scaled.shape[1]
        num_labels = 1  

        if len(DROPOUT_RATES) < len(HIDDEN_UNITS) + 2:
            raise ValueError(
                "Length of DROPOUT_RATES must be at least len(HIDDEN_UNITS) + 2"
            )

        model = create_ae_mlp(
            num_columns=num_columns,
            num_labels=num_labels,
            hidden_units=HIDDEN_UNITS,
            dropout_rates=DROPOUT_RATES,
            ls=LABEL_SMOOTHING,
            lr=LEARNING_RATE,
        )
        model.summary(print_fn=logging.info)

        
        model_filename = os.path.join(
            MODEL_SAVE_DIR, f"{target_asset.lower()}_ae_mlp_model.keras"
        )
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            model_filename,
            monitor="val_action_AUC",  
            save_best_only=True,
            save_weights_only=False,  
            mode="max",  
            verbose=1,
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_action_AUC",
            patience=10,  
            mode="max",
            restore_best_weights=True,  
            verbose=1,
        )

        
        train_targets = {
            "decoder": y_decoder_train_scaled,
            "ae_action": y_action_train,
            "action": y_action_train,
        }
        val_targets = {
            "decoder": y_decoder_val_scaled,
            "ae_action": y_action_val,
            "action": y_action_val,
        }

        
        logging.info(f"Starting model training for {target_asset}...")
        history = model.fit(
            X_train_scaled,
            train_targets,
            validation_data=(X_val_scaled, val_targets),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[checkpoint, early_stopping],
            verbose=1,  
        )

        logging.info(
            f"Training finished for {target_asset}. Best model saved to {model_filename}"
        )

    logging.info("--- Model training script finished for all assets --- ")


if __name__ == "__main__":
    main()
