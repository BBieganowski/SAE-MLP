# SAE-MLP for financial time series forecasting

This project implements a pipeline for analyzing cryptocurrency time series data (ETH, BTC, LTC). It involves downloading data, performing feature engineering (fractional differentiation), generating labels (triple barrier method), and training a supervised autoencoder model for price movement prediction.

## Features

*   **Data Acquisition:** Downloads 5-minute candle data from Binance for specified assets (ETHUSDT, BTCUSDT, LTCUSDT) over a configurable period (e.g., past year).
*   **Efficient Storage:** Stores downloaded and processed data in a local DuckDB database (`database/crypto_data.db`).
*   **Fractional Differentiation:** Applies fractional differentiation to price series to achieve stationarity while aiming to preserve maximal memory, using the fixed-window method.
*   **Optimal Differentiation Order:** Uses Optuna to find the minimum differentiation order (`d`) required to pass the Augmented Dickey-Fuller (ADF) stationarity test for each asset.
*   **Triple Barrier Labeling:** Generates supervised learning labels based on the triple barrier method (configurale window duration and barrier thresholds).
*   **Supervised Autoencoder:** Implements and trains a Keras-based supervised autoencoder model designed for noisy features. The model learns simultaneously to reconstruct the input features (FracDiff prices) and classify the triple barrier labels.
*   **Modular Pipeline:** Scripts are organized for each distinct step: download, differentiation, labeling, and training.

## Project Structure

```
.
├── database/ # Directory for DuckDB database file
│ └── crypto_data.db # DuckDB database file (created by scripts)
├── trained_models/ # Directory for saved Keras models
│ ├── ethusdt_ae_mlp_model.keras # Example saved model
│ └── ...
├── .venv/ # Python virtual environment (created by user)
├── data_download.py # Script for downloading Binance data
├── fractional_differentiation.py # Script for calculating fractional differentiation
├── triple_barrier_labeling.py # Script for generating triple barrier labels
├── model.py # Defines the Keras supervised autoencoder model
├── train_model.py # Script for training the model
└── README.md # This file
```

## Setup

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create and activate a Python virtual environment:**
    *   Requires Python 3.10 or newer recommended.
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    *(On Windows, use `.venv\Scripts\activate`)*

3.  **Install dependencies:**
    ```bash
    pip install pandas numpy duckdb python-binance aiohttp statsmodels optuna tensorflow scikit-learn
    ```
    *(Alternatively, create a `requirements.txt` file with the packages listed above and run `pip install -r requirements.txt`)*

## Usage

Run the scripts sequentially using the Python interpreter from your activated virtual environment.

1.  **Download Data:**
    *   Modify `ASSETS` or `DAYS_TO_FETCH` in `data_download.py` if needed.
    ```bash
    .venv/bin/python data_download.py
    ```
    *   This creates `database/crypto_data.db` and populates the `candles_5m` table.

2.  **Calculate Fractional Differentiation:**
    *   Modify `ASSETS`, `ADF_PVALUE_THRESHOLD`, or `OPTUNA_N_TRIALS` in `fractional_differentiation.py` if needed.
    ```bash
    .venv/bin/python fractional_differentiation.py
    ```
    *   This populates the `candles_5m_fracdiff` table.

3.  **Generate Triple Barrier Labels:**
    *   **Important:** Adjust `WINDOW_DURATION_MINUTES` and `BARRIER_LAMBDA` in `triple_barrier_labeling.py` to your desired configuration. These parameters *must* match the ones used later in `train_model.py`.
    ```bash
    .venv/bin/python triple_barrier_labeling.py
    ```
    *   This populates the `candles_5m_labels` table.

4.  **Train the Model:**
    *   **Important:** Ensure `WINDOW_MINUTES_PARAM` and `BARRIER_LAMBDA_PARAM` in `train_model.py` match the parameters used in the previous labeling step.
    *   Modify model hyperparameters (`HIDDEN_UNITS`, `DROPOUT_RATES`, `EPOCHS`, `BATCH_SIZE`, etc.) if desired.
    ```bash
    .venv/bin/python train_model.py
    ```
    *   This trains a model for each asset using the FracDiff features and corresponding labels, saving the best models to the `trained_models/` directory.

## Database Schema (`database/crypto_data.db`)

*   **`candles_5m`**: Stores raw 5-minute candle data from Binance.
    *   `open_time` (TIMESTAMP, PK)
    *   `open` (DOUBLE)
    *   `high` (DOUBLE)
    *   `low` (DOUBLE)
    *   `close` (DOUBLE)
    *   `volume` (DOUBLE)
    *   `close_time` (TIMESTAMP)
    *   `quote_asset_volume` (DOUBLE)
    *   `number_of_trades` (INTEGER)
    *   `taker_buy_base_asset_volume` (DOUBLE)
    *   `taker_buy_quote_asset_volume` (DOUBLE)
    *   `symbol` (VARCHAR, PK)
*   **`candles_5m_fracdiff`**: Stores fractionally differentiated closing prices.
    *   `open_time` (TIMESTAMP, PK)
    *   `symbol` (VARCHAR, PK)
    *   `close_fracdiff` (DOUBLE): The differentiated series value.
    *   `d_order` (DOUBLE): The optimal differentiation order `d` found.
*   **`candles_5m_labels`**: Stores triple barrier labels.
    *   `open_time` (TIMESTAMP, PK)
    *   `symbol` (VARCHAR, PK)
    *   `label` (INTEGER): The outcome label (1: upper hit, -1: lower hit, 0: vertical barrier hit).
    *   `window_minutes` (INTEGER, PK): Look-forward window duration used for labeling.
    *   `barrier_lambda` (DOUBLE, PK): Barrier threshold used for labeling.
    *   `target_hit_time` (TIMESTAMP): Timestamp when the first horizontal barrier was hit (NULL if vertical barrier hit first).

## Model Architecture (`model.py`)

The project utilizes a supervised autoencoder architecture built with Keras/TensorFlow. Key aspects:

*   **Input:** Fractionally differentiated price series for multiple assets.
*   **Encoder:** Compresses the input features into a lower-dimensional representation. Includes Gaussian Noise for regularization.
*   **Decoder Branch:** Attempts to reconstruct the original (scaled) input features from the encoder's output. Loss is typically Mean Squared Error.
*   **Classification Branch:** Takes the encoder output (optionally concatenated with the initial input) through further dense layers to predict the target label (derived from triple barrier). Loss is typically Binary Crossentropy.
*   **Dual Objective:** The model is trained simultaneously on both the reconstruction loss and the classification loss, encouraging the encoder to learn representations useful for both tasks.

## Future Work

*   Hyperparameter tuning for the model (e.g., using Optuna or KerasTuner).
*   Adding more input features (e.g., volume, other technical indicators).
*   Implementing more sophisticated data splitting/validation techniques (e.g., walk-forward validation).
*   Detailed performance evaluation (classification metrics, backtesting).
*   Experimenting with different model architectures.