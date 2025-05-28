import pandas as pd
import numpy as np
import duckdb
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants
DB_PATH = "database/crypto_data.db"
CANDLES_TABLE = "candles_5m"  # Assuming 5-minute candles
PREDICTIONS_TABLE = "raw_predictions"
EQUITY_TABLE = "backtest_equity"
INITIAL_CAPITAL = 100000.0
TRANSACTION_COST = 0.0001  # 0.1% per trade

def load_predictions_and_candles(con, model_name, asset, window_minutes):
    """Load predictions and corresponding candle data for backtesting."""
    logging.info(f"Loading predictions for {model_name} on {asset} with window_minutes={window_minutes}...")
    
    # Load predictions
    pred_query = f"""
    SELECT open_time, prediction_value, predicted_class, source_window_minutes
    FROM {PREDICTIONS_TABLE}
    WHERE model_name = ? AND asset = ? AND source_window_minutes = ?
    ORDER BY open_time
    """
    predictions = con.execute(pred_query, [model_name, asset, window_minutes]).fetchdf()
    
    if predictions.empty:
        logging.warning(f"No predictions found for {model_name} on {asset}")
        return None, None
    
    # Get the time range
    start_time = predictions['open_time'].min()
    end_time = predictions['open_time'].max()
    
    # Load candle data for the period
    candles_query = f"""
    SELECT open_time, symbol, open, high, low, close, volume
    FROM {CANDLES_TABLE}
    WHERE symbol = ? AND open_time >= ? AND open_time <= ?
    ORDER BY open_time
    """
    candles = con.execute(candles_query, [asset, start_time, end_time]).fetchdf()
    
    if candles.empty:
        logging.warning(f"No candle data found for {asset} in the prediction period")
        return predictions, None
    
    logging.info(f"Loaded {len(predictions)} predictions and {len(candles)} candles for {asset}")
    return predictions, candles


def get_position_signal(model_name, prediction_value, predicted_class):
    """Convert model predictions to position signals."""
    if model_name == "regression_base":
        # Regression: negative -> short (-1), positive -> long (1)
        return -1 if prediction_value < 0 else 1
    
    elif model_name == "classification_base":
        # Classification: 0 -> short (-1), 1 -> long (1)
        return -1 if predicted_class == 0 else 1
    
    elif model_name == "sae_mlp":
        # SAE-MLP: 0 -> short (-1), 1 -> hold (0), 2 -> long (1)
        if pd.isna(predicted_class):
            return 0
        class_mapping = {0: -1, 1: 0, 2: 1}
        return class_mapping.get(int(predicted_class), 0)
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def backtest_strategy(predictions, candles, model_name, asset, initial_capital):
    """Run backtest for a single asset and model."""
    logging.info(f"Starting backtest for {model_name} on {asset}...")
    
    # Merge predictions with candle data
    predictions['open_time'] = pd.to_datetime(predictions['open_time'])
    candles['open_time'] = pd.to_datetime(candles['open_time'])
    
    # Create a complete timeline with candle data
    backtest_df = candles.copy()
    backtest_df = backtest_df.merge(
        predictions[['open_time', 'prediction_value', 'predicted_class', 'source_window_minutes']], 
        on='open_time', 
        how='left'
    )
    
    # Initialize tracking variables
    equity = initial_capital
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0
    entry_time = None
    min_hold_minutes = None
    
    equity_curve = []
    trades = []
    
    for idx, row in backtest_df.iterrows():
        current_time = row['open_time']
        current_price = row['close']
        
        # Check if we have a new prediction
        if not pd.isna(row['prediction_value']):
            new_signal = get_position_signal(model_name, row['prediction_value'], row['predicted_class'])
            min_hold_minutes = row['source_window_minutes']
            
            # Check if we need to close existing position
            if position != 0:
                # Check if minimum holding period has passed
                if entry_time is not None:
                    time_held = (current_time - entry_time).total_seconds() / 60
                    
                    if time_held >= min_hold_minutes or new_signal != position:
                        # Close position
                        exit_price = current_price
                        pnl = calculate_pnl(position, entry_price, exit_price, equity, TRANSACTION_COST)
                        equity += pnl
                        
                        trades.append({
                            'entry_time': entry_time,
                            'exit_time': current_time,
                            'direction': 'long' if position == 1 else 'short',
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'return': pnl / (equity - pnl)
                        })
                        
                        position = 0
                        entry_price = 0
                        entry_time = None
            
            # Open new position if signal is not flat
            if position == 0 and new_signal != 0:
                position = new_signal
                entry_price = current_price
                entry_time = current_time
                # Deduct transaction cost
                equity *= (1 - TRANSACTION_COST)
        
        # Update equity for current position
        if position != 0 and entry_price > 0:
            current_value = calculate_position_value(position, entry_price, current_price, equity)
            equity_curve.append({
                'timestamp': current_time,
                'equity': current_value,
                'position': position
            })
        else:
            equity_curve.append({
                'timestamp': current_time,
                'equity': equity,
                'position': 0
            })
    
    # Close any remaining position at the end
    if position != 0 and entry_time is not None:
        exit_price = backtest_df.iloc[-1]['close']
        pnl = calculate_pnl(position, entry_price, exit_price, equity, TRANSACTION_COST)
        equity += pnl
        
        trades.append({
            'entry_time': entry_time,
            'exit_time': backtest_df.iloc[-1]['open_time'],
            'direction': 'long' if position == 1 else 'short',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'return': pnl / (equity - pnl)
        })
    
    # Calculate performance metrics
    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    metrics = calculate_metrics(equity_df, trades_df, initial_capital)
    
    return equity_df, trades_df, metrics


def calculate_pnl(position, entry_price, exit_price, current_equity, transaction_cost):
    """Calculate P&L for a closed position."""
    if position == 1:  # Long
        gross_return = (exit_price - entry_price) / entry_price
    else:  # Short
        gross_return = (entry_price - exit_price) / entry_price
    
    # Apply transaction cost for exit
    net_return = gross_return - transaction_cost
    return current_equity * net_return


def calculate_position_value(position, entry_price, current_price, equity_at_entry):
    """Calculate current value of an open position."""
    if position == 1:  # Long
        return equity_at_entry * (current_price / entry_price)
    else:  # Short
        return equity_at_entry * (2 - current_price / entry_price)


def calculate_metrics(equity_df, trades_df, initial_capital):
    """Calculate performance metrics."""
    metrics = {}
    
    # Basic metrics
    final_equity = equity_df['equity'].iloc[-1]
    metrics['total_return'] = (final_equity - initial_capital) / initial_capital
    metrics['final_equity'] = final_equity
    
    # Calculate daily returns for Sharpe ratio
    equity_df['returns'] = equity_df['equity'].pct_change()
    daily_returns = equity_df.set_index('timestamp')['returns'].resample('D').sum()
    
    if len(daily_returns) > 1:
        metrics['sharpe_ratio'] = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        metrics['max_drawdown'] = calculate_max_drawdown(equity_df['equity'])
    else:
        metrics['sharpe_ratio'] = 0
        metrics['max_drawdown'] = 0
    
    # Trade metrics
    if not trades_df.empty:
        metrics['num_trades'] = len(trades_df)
        metrics['win_rate'] = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df)
        metrics['avg_trade_return'] = trades_df['return'].mean()
        metrics['best_trade'] = trades_df['return'].max()
        metrics['worst_trade'] = trades_df['return'].min()
    else:
        metrics['num_trades'] = 0
        metrics['win_rate'] = 0
        metrics['avg_trade_return'] = 0
        metrics['best_trade'] = 0
        metrics['worst_trade'] = 0
    
    return metrics


def calculate_max_drawdown(equity_series):
    """Calculate maximum drawdown."""
    cummax = equity_series.expanding().max()
    drawdown = (equity_series - cummax) / cummax
    return drawdown.min()


def save_equity_to_db(con, equity_data, model_name, asset, window_minutes):
    """Save equity curve to database."""
    # Create table if it doesn't exist
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {EQUITY_TABLE} (
        timestamp TIMESTAMP,
        model_name VARCHAR,
        asset VARCHAR,
        window_minutes INTEGER,
        equity DOUBLE,
        position INTEGER
    );
    """
    con.execute(create_table_sql)
    
    # Prepare data for insertion with only the required columns
    save_data = pd.DataFrame({
        'timestamp': equity_data['timestamp'],
        'model_name': model_name,
        'asset': asset,
        'window_minutes': window_minutes,
        'equity': equity_data['equity'],
        'position': equity_data['position']
    })
    
    # Insert data
    con.register('equity_df', save_data)
    con.execute(f"INSERT INTO {EQUITY_TABLE} SELECT * FROM equity_df")
    con.unregister('equity_df')
    
    logging.info(f"Saved {len(save_data)} equity records for {model_name} on {asset} with window_minutes={window_minutes}")


def main():
    logging.info("Starting backtesting script...")
    
    try:
        con = duckdb.connect(DB_PATH)
        logging.info(f"Connected to database: {DB_PATH}")
        
        # Clear existing equity data to avoid schema conflicts
        try:
            con.execute(f"DROP TABLE IF EXISTS {EQUITY_TABLE}")
            logging.info("Cleared existing equity table")
        except:
            pass
        
        # Get unique model-asset-window combinations from predictions
        combo_query = f"""
        SELECT DISTINCT model_name, asset, source_window_minutes 
        FROM {PREDICTIONS_TABLE}
        ORDER BY model_name, asset, source_window_minutes
        """
        combinations = con.execute(combo_query).fetchdf()
        
        if combinations.empty:
            logging.error("No predictions found in the database")
            return
        
        all_results = []
        
        # Run backtest for each combination
        for _, combo in combinations.iterrows():
            model_name = combo['model_name']
            asset = combo['asset']
            window_minutes = combo['source_window_minutes']
            
            logging.info(f"\n{'='*60}")
            logging.info(f"Backtesting {model_name} on {asset} (window={window_minutes}min)")
            logging.info(f"{'='*60}")
            
            # Load data
            predictions, candles = load_predictions_and_candles(con, model_name, asset, window_minutes)
            
            if predictions is None or candles is None:
                logging.warning(f"Skipping {model_name} on {asset} with window={window_minutes} due to missing data")
                continue
            
            # Run backtest
            equity_df, trades_df, metrics = backtest_strategy(
                predictions, candles, model_name, asset, INITIAL_CAPITAL
            )
            
            # Save equity curve to database
            save_equity_to_db(con, equity_df, model_name, asset, window_minutes)
            
            # Store results
            result = {
                'model': model_name,
                'asset': asset,
                'window_minutes': window_minutes,
                **metrics
            }
            all_results.append(result)
            
            # Print metrics
            logging.info(f"\nPerformance Metrics for {model_name} on {asset}:")
            logging.info(f"Final Equity: ${metrics['final_equity']:,.2f}")
            logging.info(f"Total Return: {metrics['total_return']:.2%}")
            logging.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            logging.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
            logging.info(f"Number of Trades: {metrics['num_trades']}")
            logging.info(f"Win Rate: {metrics['win_rate']:.2%}")
            logging.info(f"Avg Trade Return: {metrics['avg_trade_return']:.2%}")
        
        # Create summary table
        if all_results:
            results_df = pd.DataFrame(all_results)
            
            # Save summary to database
            create_summary_table = """
            CREATE TABLE IF NOT EXISTS backtest_summary (
                model VARCHAR,
                asset VARCHAR,
                window_minutes INTEGER,
                total_return DOUBLE,
                final_equity DOUBLE,
                sharpe_ratio DOUBLE,
                max_drawdown DOUBLE,
                num_trades INTEGER,
                win_rate DOUBLE,
                avg_trade_return DOUBLE,
                best_trade DOUBLE,
                worst_trade DOUBLE
            );
            """
            con.execute(create_summary_table)
            
            # Clear existing summary
            con.execute("DELETE FROM backtest_summary")
            
            # Insert new summary
            con.register('summary_df', results_df)
            con.execute("INSERT INTO backtest_summary SELECT * FROM summary_df")
            con.unregister('summary_df')
            
            logging.info("\n" + "="*80)
            logging.info("BACKTEST SUMMARY")
            logging.info("="*80)
            print(results_df.to_string(index=False))
        
        con.close()
        logging.info("\nBacktesting completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during backtesting: {e}", exc_info=True)
        if 'con' in locals() and con:
            con.close()


if __name__ == "__main__":
    main() 