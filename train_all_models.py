import pandas as pd
import numpy as np
import duckdb
import logging
import os
import time
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import optuna
from models import create_base_regression_model, create_binary_classification_model, create_ae_mlp, CustomSAEMetric

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

DB_PATH = "database/crypto_data.db"
FRACDIFF_TABLE = "candles_5m_fracdiff"
LABEL_TABLE = "candles_5m_labels"

BASIC_LABEL_TABLE_NAME = "candles_5m_basic_labels"
BASIC_LABEL_WINDOW_MINUTES = 5

TARGET_ASSETS = ["BTCUSDT", "ES", "M6E"]
WALK_FORWARD_END_YEAR = 2019

WINDOW_MINUTES_PARAM = 30
BARRIER_LAMBDA_PARAM = 0.002

N_OPTUNA_TRIALS = 50
N_WALK_FORWARD_SPLITS = 5
EPOCHS = 100

MODEL_ROOT_DIR = "trained_models_all"
PREDICTIONS_ROOT_DIR = "predictions_all"
os.makedirs(MODEL_ROOT_DIR, exist_ok=True)

def load_raw_data(con):
    """Loads features (FracDiff), triple barrier labels, and basic labels from DuckDB."""
    logging.info("Loading fractional differentiation data...")
    fd_query = f"SELECT open_time, symbol, close_fracdiff FROM {FRACDIFF_TABLE}"
    all_fd_data = con.execute(fd_query).fetchdf()

    if all_fd_data.empty:
        raise ValueError(f"No data found in {FRACDIFF_TABLE}. Run fractional_differentiation.py first.")

    features_df = all_fd_data.pivot(
        index="open_time", columns="symbol", values="close_fracdiff"
    )
    features_df.columns = [f"{col}_fd" for col in features_df.columns]
    features_df.index = pd.to_datetime(features_df.index, unit='s')
    logging.info(f"Loaded {len(features_df)} rows of feature data. Index type: {features_df.index.dtype}")

    logging.info(f"Loading triple barrier labels (window={WINDOW_MINUTES_PARAM}, lambda={BARRIER_LAMBDA_PARAM}) from {LABEL_TABLE}...")
    tb_label_query = f"SELECT open_time, symbol, label FROM {LABEL_TABLE} WHERE window_minutes = ? AND barrier_lambda = ?"
    triple_barrier_label_data = con.execute(
        tb_label_query, [WINDOW_MINUTES_PARAM, BARRIER_LAMBDA_PARAM]
    ).fetchdf()

    if triple_barrier_label_data.empty:
        logging.warning( # Changed to warning as some models might not need it
            f"No triple barrier labels found in {LABEL_TABLE} for window={WINDOW_MINUTES_PARAM}, lambda={BARRIER_LAMBDA_PARAM}."
        )
    else:
        triple_barrier_label_data['open_time'] = pd.to_datetime(triple_barrier_label_data['open_time'], unit='s')
        logging.info(f"Loaded {len(triple_barrier_label_data)} rows of triple barrier label data.")

    logging.info(f"Loading basic labels (window={BASIC_LABEL_WINDOW_MINUTES} minutes) from {BASIC_LABEL_TABLE_NAME}...")
    basic_label_query = f"SELECT open_time, symbol, future_return, future_direction FROM {BASIC_LABEL_TABLE_NAME} WHERE window_minutes = ?"
    basic_label_data = con.execute(basic_label_query, [BASIC_LABEL_WINDOW_MINUTES]).fetchdf()

    if basic_label_data.empty:
        logging.warning( # Changed to warning
            f"No basic labels found in {BASIC_LABEL_TABLE_NAME} for window_minutes = {BASIC_LABEL_WINDOW_MINUTES}."
        )
    else:
        basic_label_data['open_time'] = pd.to_datetime(basic_label_data['open_time'], unit='s')
        logging.info(f"Loaded {len(basic_label_data)} rows of basic label data.")

    return features_df.sort_index(), triple_barrier_label_data.sort_values(by='open_time') if not triple_barrier_label_data.empty else pd.DataFrame(), basic_label_data.sort_values(by='open_time') if not basic_label_data.empty else pd.DataFrame()


def prepare_data_for_regression(features_df, basic_label_data, target_asset):
    logging.info(f"Preparing regression data for {target_asset} using basic labels (future_return)...")
    
    if basic_label_data.empty:
        logging.warning(f"Basic label data is empty. Skipping regression for {target_asset}.")
        return None, None, None

    asset_basic_labels = basic_label_data[basic_label_data["symbol"] == target_asset].copy()
    if asset_basic_labels.empty:
        logging.warning(f"No basic labels found for asset {target_asset}. Skipping regression.")
        return None, None, None
    
    asset_basic_labels.set_index("open_time", inplace=True)
    
    merged_df = features_df.join(asset_basic_labels[['future_return']], how="inner")
    
    initial_rows = len(merged_df)
    merged_df.dropna(inplace=True) # Drops rows with NaN features or NaN future_return
    if initial_rows > len(merged_df):
        logging.info(f"Dropped {initial_rows - len(merged_df)} rows due to NaNs for regression on {target_asset}.")

    if merged_df.empty:
        logging.warning(f"No data after merging/NA drop for regression on {target_asset}.")
        return None, None, None

    X = merged_df.drop(columns=["future_return"])
    y = merged_df["future_return"]
    timestamps = merged_df.index
    
    logging.info(f"Regression data for {target_asset}: X shape {X.shape}, y shape {y.shape}")
    return X, y, timestamps


def prepare_data_for_classification(features_df, basic_label_data, target_asset):
    logging.info(f"Preparing classification data for {target_asset} using basic labels (future_direction)...")

    if basic_label_data.empty:
        logging.warning(f"Basic label data is empty. Skipping classification for {target_asset}.")
        return None, None, None
        
    asset_basic_labels = basic_label_data[basic_label_data["symbol"] == target_asset].copy()
    if asset_basic_labels.empty:
        logging.warning(f"No basic labels found for asset {target_asset}. Skipping classification.")
        return None, None, None
        
    asset_basic_labels.set_index("open_time", inplace=True)
    
    merged_df = features_df.join(asset_basic_labels[['future_direction']], how="inner")
    
    initial_rows = len(merged_df)
    merged_df.dropna(inplace=True)
    if initial_rows > len(merged_df):
        logging.info(f"Dropped {initial_rows - len(merged_df)} rows due to NaNs for classification on {target_asset}.")

    if merged_df.empty:
        logging.warning(f"No data after merging/NA drop for classification on {target_asset}.")
        return None, None, None

    labeled_df = merged_df[merged_df["future_direction"] != 0].copy() # Filter out 'hold' signals
    if labeled_df.empty:
        logging.warning(f"No non-zero future_direction labels found for {target_asset} after filtering. Skipping.")
        return None, None, None
        
    logging.info(f"Filtered out future_direction 0. Kept {len(labeled_df)} rows for {target_asset}.")

    X = labeled_df.drop(columns=['future_direction'])
    y = labeled_df["future_direction"].map({1: 1, -1: 0}).astype(int) # Map labels to 0 and 1
    timestamps = labeled_df.index

    logging.info(f"Classification data for {target_asset}: X shape {X.shape}, y shape {y.shape}")
    return X, y, timestamps


def prepare_data_for_sae_mlp(features_df, triple_barrier_label_data, target_asset):
    logging.info(f"Preparing SAE-MLP data for {target_asset} using triple barrier labels...")

    if triple_barrier_label_data.empty:
        logging.warning(f"Triple barrier label data is empty. Skipping SAE-MLP for {target_asset}.")
        return None, None, None, None
    
    asset_tb_labels = triple_barrier_label_data[triple_barrier_label_data["symbol"] == target_asset].copy()
    if asset_tb_labels.empty:
        logging.warning(f"No triple barrier labels found for asset {target_asset}. Skipping SAE-MLP.")
        return None, None, None, None
        
    asset_tb_labels.set_index("open_time", inplace=True)
    
    merged_df = features_df.join(asset_tb_labels[['label']], how="inner") # 'label' is from triple_barrier_label_data
    
    initial_rows = len(merged_df)
    merged_df.dropna(inplace=True)
    if initial_rows > len(merged_df):
        logging.info(f"Dropped {initial_rows - len(merged_df)} rows due to NaNs for SAE-MLP on {target_asset}.")

    if merged_df.empty:
        logging.warning(f"No data after merging/NA drop for SAE-MLP on {target_asset}.")
        return None, None, None, None

    # For 3-class: {-1 (sell), 0 (hold), 1 (buy)} -> {0, 1, 2}
    # Do NOT filter out label 0 (hold) anymore.
    labeled_df = merged_df.copy() # Use all data after merging and NA drop
    # logging.info(f"Filtered out triple barrier label 0. Kept {len(labeled_df)} rows for SAE-MLP {target_asset}.") # Old message
    logging.info(f"Using {len(labeled_df)} rows for SAE-MLP {target_asset} (including holds).")

    X = labeled_df.drop(columns=['label'])
    y_action = labeled_df["label"].map({-1: 0, 0: 1, 1: 2}).astype(int) # Map to 0, 1, 2
    timestamps = labeled_df.index
    
    y_decoder = X.copy()

    logging.info(f"SAE-MLP data for {target_asset}: X={X.shape}, y_action={y_action.shape}, y_decoder={y_decoder.shape}")
    return X, y_action, y_decoder, timestamps

def suggest_regression_hps(trial):
    params = {}
    params['learning_rate'] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_units = [trial.suggest_int(f"units_layer_{i}", 2, 16, log=True) for i in range(n_layers)]
    params['hidden_units'] = hidden_units
    params['batch_size'] = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    return params

def suggest_classification_hps(trial):
    params = {}
    params['learning_rate'] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_units = [trial.suggest_int(f"units_layer_{i}", 2, 32, log=True) for i in range(n_layers)]
    params['hidden_units'] = hidden_units
    params['batch_size'] = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    return params

def suggest_sae_mlp_hps(trial):
    params = {}
    params['lr'] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    params['ls'] = trial.suggest_float("ls", 0.0, 0.2)
    
    hu_encoder = trial.suggest_categorical("hu_encoder", [1, 2, 4, 8])
    hu_ae_path = trial.suggest_categorical("hu_ae_path", [1, 2, 4, 8])
    n_mlp_layers = trial.suggest_int("n_mlp_layers", 1, 3)
    mlp_hidden_units = [trial.suggest_categorical(f"hu_mlp_{i}", [1, 2, 4, 8]) for i in range(n_mlp_layers)]
    params['hidden_units'] = [hu_encoder, hu_ae_path] + mlp_hidden_units
    
    num_dropout_rates = len(params['hidden_units']) + 2
    dropout_rates = [trial.suggest_float(f"dropout_rate_{i}", 0.0, 0.5) for i in range(num_dropout_rates)]
    params['dropout_rates'] = dropout_rates
    params['batch_size'] = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    return params

def objective(trial, model_spec, X_hpo, y_hpo, y_decoder_hpo=None):
    """Optuna objective function for a single model type and asset."""
    tf.keras.backend.clear_session()
    
    hyperparams = model_spec["hyperparameters_fn"](trial)
    num_columns = X_hpo.shape[1]
    num_labels = 1
    
    fold_metrics = []
    
    tscv = TimeSeriesSplit(n_splits=N_WALK_FORWARD_SPLITS)
    for train_idx, val_idx in tscv.split(X_hpo):
        X_train_fold, X_val_fold = X_hpo.iloc[train_idx], X_hpo.iloc[val_idx]
        y_train_fold, y_val_fold = y_hpo.iloc[train_idx], y_hpo.iloc[val_idx]
        
        if y_decoder_hpo is not None:
            y_decoder_train_fold, y_decoder_val_fold = y_decoder_hpo.iloc[train_idx], y_decoder_hpo.iloc[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)
        
        y_decoder_train_scaled, y_decoder_val_scaled = None, None
        if model_spec["is_sae"] and y_decoder_hpo is not None:
            y_decoder_train_scaled = scaler.transform(y_decoder_train_fold)
            y_decoder_val_scaled = scaler.transform(y_decoder_val_fold)

        try:
            if model_spec["is_sae"]:
                num_action_classes_for_sae = 3 # For {-1, 0, 1} mapped to {0, 1, 2}
                model = model_spec["model_create_fn"](
                    num_columns=num_columns,
                    num_labels=num_action_classes_for_sae, # Pass 3 for num_labels
                    hidden_units=hyperparams['hidden_units'],
                    dropout_rates=hyperparams['dropout_rates'],
                    ls=hyperparams['ls'],
                    lr=hyperparams['lr'],
                    batch_size=hyperparams['batch_size']
                )
                train_targets = {
                    "decoder": y_decoder_train_scaled,
                    "ae_action": y_train_fold.values,
                    "action": y_train_fold.values
                }
                val_targets = {
                    "decoder": y_decoder_val_scaled,
                    "ae_action": y_val_fold.values,
                    "action": y_val_fold.values
                }
            else:
                model = model_spec["model_create_fn"](
                    num_columns=num_columns,
                    hidden_units=hyperparams['hidden_units'],
                    learning_rate=hyperparams['learning_rate'],
                    batch_size=hyperparams['batch_size']
                )
                train_targets = y_train_fold.values
                val_targets = y_val_fold.values

            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor=model_spec["early_stopping_metric"],
                patience=10,
                mode='max' if model_spec["early_stopping_metric"] == "val_action_custom_metric" else model_spec["optuna_direction"],
                restore_best_weights=True
            )

            history = model.fit(
                X_train_scaled,
                train_targets,
                validation_data=(X_val_scaled, val_targets),
                epochs=EPOCHS,
                batch_size=hyperparams['batch_size'],
                callbacks=[early_stopping],
                verbose=0 # Suppress verbose logging during HPO
            )
            
            val_metric_value = history.history[model_spec["early_stopping_metric"]][early_stopping.best_epoch]
            fold_metrics.append(val_metric_value)

        except Exception as e:
            logging.warning(f"Trial failed for {model_spec['name']} asset {model_spec['asset']} with params {hyperparams}: {e}", exc_info=True)
            return float('inf') if model_spec["optuna_direction"] == "minimize" else float('-inf')

    if not fold_metrics:
        logging.warning(f"All walk-forward folds failed for trial of {model_spec['name']} asset {model_spec['asset']}.")
        return float('inf') if model_spec["optuna_direction"] == "minimize" else float('-inf')
        
    avg_metric = np.mean(fold_metrics)
    logging.info(f"Trial completed for {model_spec['name']} asset {model_spec['asset']}. Avg {model_spec['optuna_metric']}: {avg_metric:.4f} with params: {hyperparams}")
    return avg_metric


# --- Main Training Loop ---

def main():
    logging.info("Starting main training script for all models and assets.")
    
    db_connection = None
    try:
        db_connection = duckdb.connect(DB_PATH)
        logging.info(f"Connected to database: {DB_PATH}")
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS raw_predictions (
            open_time TIMESTAMP,
            asset VARCHAR,
            model_name VARCHAR,
            prediction_value DOUBLE,
            predicted_class INTEGER,
            source_window_minutes INTEGER,
            source_barrier_lambda DOUBLE,
            training_walk_forward_end_year INTEGER
        );
        """
        db_connection.execute(create_table_sql)
        logging.info("Ensured 'raw_predictions' table exists.")

        features_df, triple_barrier_label_data, basic_label_data = load_raw_data(db_connection)
        
        if features_df.empty:
            logging.error("Features DataFrame is empty after loading. Exiting.")
            return
        # Basic_label_data or triple_barrier_label_data can be empty if not found, handled by prepare_data functions

    except Exception as e:
        logging.error(f"Failed to load data or initialize table: {e}", exc_info=True)
        if db_connection:
            db_connection.close()
        return

    hpo_train_end_dt = pd.Timestamp(f'{WALK_FORWARD_END_YEAR}-12-31 23:59:59')

    MODEL_SPECS_TEMPLATE = [
        # {
        #     "name": "regression_base",
        #     "model_create_fn": create_base_regression_model,
        #     "prepare_data_fn": prepare_data_for_regression, # Uses basic_label_data
        #     "optuna_direction": "minimize",
        #     "optuna_metric": "val_loss",
        #     "early_stopping_metric": "val_loss",
        #     "hyperparameters_fn": suggest_regression_hps,
        #     "is_sae": False,
        # },
        # {
        #     "name": "classification_base",
        #     "model_create_fn": create_binary_classification_model,
        #     "prepare_data_fn": prepare_data_for_classification, # Uses basic_label_data
        #     "optuna_direction": "maximize",
        #     "optuna_metric": "val_auc",
        #     "early_stopping_metric": "val_auc", 
        #     "hyperparameters_fn": suggest_classification_hps,
        #     "is_sae": False,
        # },
        {
            "name": "sae_mlp", 
            "model_create_fn": create_ae_mlp,
            "prepare_data_fn": prepare_data_for_sae_mlp, # Uses triple_barrier_label_data
            "optuna_direction": "maximize",
            "optuna_metric": "val_action_custom_metric",
            "early_stopping_metric": "val_action_custom_metric",
            "hyperparameters_fn": suggest_sae_mlp_hps,
            "is_sae": True,
        }
    ]

    for asset in TARGET_ASSETS:
        asset_model_dir = os.path.join(MODEL_ROOT_DIR, asset)
        os.makedirs(asset_model_dir, exist_ok=True)

        for spec_template in MODEL_SPECS_TEMPLATE:
            model_spec = spec_template.copy()
            model_spec["asset"] = asset
            
            logging.info(f"--- Processing model: {model_spec['name']} for asset: {asset} ---")

            X_all, y_all, y_decoder_all, timestamps_all = None, None, None, None
            
            if model_spec["name"] == "regression_base":
                if basic_label_data.empty:
                    logging.warning(f"Skipping {model_spec['name']} for {asset} due to missing basic_label_data.")
                    continue
                X_all, y_all, timestamps_all = model_spec["prepare_data_fn"](features_df, basic_label_data, asset)
            elif model_spec["name"] == "classification_base":
                if basic_label_data.empty:
                    logging.warning(f"Skipping {model_spec['name']} for {asset} due to missing basic_label_data.")
                    continue
                X_all, y_all, timestamps_all = model_spec["prepare_data_fn"](features_df, basic_label_data, asset)
            elif model_spec["is_sae"]: # sae_mlp
                if triple_barrier_label_data.empty:
                    logging.warning(f"Skipping {model_spec['name']} for {asset} due to missing triple_barrier_label_data.")
                    continue
                X_all, y_all, y_decoder_all, timestamps_all = model_spec["prepare_data_fn"](features_df, triple_barrier_label_data, asset)
            else:
                logging.error(f"Unknown model spec type or missing label data for: {model_spec['name']}")
                continue

            if X_all is None or X_all.empty:
                logging.warning(f"No data prepared for {model_spec['name']} on {asset}. Skipping.")
                continue
            
            hpo_mask = timestamps_all <= hpo_train_end_dt
            test_mask = timestamps_all > hpo_train_end_dt

            X_hpo = X_all[hpo_mask]
            y_hpo = y_all[hpo_mask]
            
            X_test = X_all[test_mask]
            # y_test = y_all[test_mask] # Not explicitly used later, but good to have if needed
            timestamps_test = timestamps_all[test_mask]

            y_decoder_hpo, y_decoder_test = None, None
            if model_spec["is_sae"] and y_decoder_all is not None: # y_decoder_all could be None if prep failed
                y_decoder_hpo = y_decoder_all[hpo_mask]
                # y_decoder_test = y_decoder_all[test_mask] # Not explicitly used

            if X_hpo.empty or len(X_hpo) < N_WALK_FORWARD_SPLITS * 2 :
                logging.warning(f"Not enough data for HPO for {model_spec['name']} on {asset} (samples: {len(X_hpo)}). Skipping.")
                continue
            
            logging.info(f"Starting Optuna HPO for {model_spec['name']} on {asset}...")
            study = optuna.create_study(direction=model_spec["optuna_direction"])
            
            study.optimize(
                lambda trial: objective(trial, model_spec, X_hpo, y_hpo, y_decoder_hpo),
                n_trials=N_OPTUNA_TRIALS,
                timeout=18000
            )
            
            best_hyperparams = study.best_params
            logging.info(f"Best HPs for {model_spec['name']} on {asset}: {best_hyperparams}")
            logging.info(f"Best value ({model_spec['optuna_metric']}): {study.best_value}")

            logging.info(f"Training final model for {model_spec['name']} on {asset} with best HPs...")
            tf.keras.backend.clear_session()

            final_scaler = StandardScaler()
            X_hpo_scaled = final_scaler.fit_transform(X_hpo)
            
            final_y_decoder_hpo_scaled = None
            if model_spec["is_sae"] and y_decoder_hpo is not None:
                final_y_decoder_hpo_scaled = final_scaler.transform(y_decoder_hpo)

            num_columns = X_hpo.shape[1]
            # num_labels = 1 # Old

            # Retrieve full hyperparameter set from the best trial for model creation
            full_best_hyperparams = model_spec["hyperparameters_fn"](study.best_trial)

            if model_spec["is_sae"]:
                num_final_action_classes = 3 # For {-1, 0, 1} mapped to {0, 1, 2}
                final_model = model_spec["model_create_fn"](
                    num_columns=num_columns, 
                    num_labels=num_final_action_classes, # Pass 3 for num_labels
                    hidden_units=full_best_hyperparams['hidden_units'], 
                    dropout_rates=full_best_hyperparams['dropout_rates'],
                    ls=full_best_hyperparams['ls'], 
                    lr=full_best_hyperparams['lr'], 
                    batch_size=full_best_hyperparams['batch_size']
                )
                final_train_targets = {
                    "decoder": final_y_decoder_hpo_scaled,
                    "ae_action": y_hpo.values,
                    "action": y_hpo.values
                }
            else: # Regression or Classification
                final_model = model_spec["model_create_fn"](
                    num_columns=num_columns,
                    hidden_units=full_best_hyperparams['hidden_units'],
                    learning_rate=full_best_hyperparams['learning_rate'],
                    batch_size=full_best_hyperparams['batch_size']
                )
                final_train_targets = y_hpo.values
            
            final_model.fit(
                X_hpo_scaled,
                final_train_targets,
                epochs=EPOCHS,
                batch_size=full_best_hyperparams['batch_size'],
                verbose=1
            )

            model_save_path = os.path.join(asset_model_dir, f"{model_spec['name']}_model.keras")
            final_model.save(model_save_path)
            logging.info(f"Saved final model for {model_spec['name']} on {asset} to {model_save_path}")

            if not X_test.empty:
                logging.info(f"Making predictions for {model_spec['name']} on {asset} for test period...")
                X_test_scaled = final_scaler.transform(X_test)
                
                raw_predictions_output = final_model.predict(X_test_scaled)

                if model_spec["is_sae"]:
                    # raw_predictions_output[2] will be (num_samples, 3) with probabilities for each class
                    predicted_class_indices = np.argmax(raw_predictions_output[2], axis=1) # 0, 1, or 2
                    prediction_probabilities = np.max(raw_predictions_output[2], axis=1) # Probability of the predicted class
                    predictions_values = prediction_probabilities # Store the max probability as the 'prediction_value'
                    # The actual predicted class (0, 1, 2) will go into 'predicted_class' column
                else:
                    predictions_values = raw_predictions_output.flatten()
                
                current_source_window_minutes = None
                current_source_barrier_lambda = pd.NA

                if model_spec["name"] == "sae_mlp":
                    current_source_window_minutes = WINDOW_MINUTES_PARAM
                    current_source_barrier_lambda = BARRIER_LAMBDA_PARAM
                else: # regression_base or classification_base
                    current_source_window_minutes = BASIC_LABEL_WINDOW_MINUTES
                    # current_source_barrier_lambda remains pd.NA from initialization

                predictions_df = pd.DataFrame({
                    'open_time': timestamps_test,
                    'asset': asset,
                    'model_name': model_spec['name'],
                    'prediction_value': predictions_values,
                    # Values assigned based on model type
                    'source_window_minutes': current_source_window_minutes,
                    'source_barrier_lambda': current_source_barrier_lambda,
                    'training_walk_forward_end_year': WALK_FORWARD_END_YEAR
                })
                
                if model_spec["is_sae"]:
                    # For SAE-MLP, predicted_class_indices already contains 0, 1, or 2
                    predictions_df['predicted_class'] = predicted_class_indices
                elif model_spec["name"] != "regression_base": # For binary classification_base
                    predictions_df['predicted_class'] = (predictions_df['prediction_value'] > 0.5).astype(int)
                else: # For regression_base
                    predictions_df['predicted_class'] = pd.NA
                
                column_order = [
                    'open_time', 'asset', 'model_name', 'prediction_value', 'predicted_class',
                    'source_window_minutes', 'source_barrier_lambda', 'training_walk_forward_end_year'
                ]
                predictions_df = predictions_df[column_order]

                try:
                    db_connection.register('predictions_batch_df', predictions_df)
                    db_connection.execute("INSERT INTO raw_predictions SELECT * FROM predictions_batch_df")
                    db_connection.unregister('predictions_batch_df')
                    logging.info(f"Saved predictions for {model_spec['name']} on {asset} to 'raw_predictions' table.")

                except Exception as e_db:
                    logging.error(f"Failed to save predictions to DuckDB for {model_spec['name']} on {asset}: {e_db}", exc_info=True)
            else:
                logging.info(f"No test data to make predictions for {model_spec['name']} on {asset}.")
            
            logging.info(f"--- Finished processing model: {model_spec['name']} for asset: {asset} ---")
            time.sleep(1) # Reduced sleep

    logging.info("--- Model training script finished for all assets and models --- ")

    if db_connection:
        db_connection.close()
        logging.info("Database connection closed at the end of main.")

if __name__ == "__main__":
    main() 