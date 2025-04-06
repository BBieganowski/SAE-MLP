import asyncio
import pandas as pd
from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException
from datetime import datetime, timedelta
import time
import duckdb
import logging
import os


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


ASSETS = ["ETHUSDT", "BTCUSDT", "LTCUSDT"]
INTERVAL = AsyncClient.KLINE_INTERVAL_5MINUTE
DB_PATH = "database/crypto_data.db"
TABLE_NAME = "candles_5m"
DAYS_TO_FETCH = 365  


os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


con = duckdb.connect(DB_PATH)


con.execute(
    f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    open_time TIMESTAMP,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE,
    close_time TIMESTAMP,
    quote_asset_volume DOUBLE,
    number_of_trades INTEGER,
    taker_buy_base_asset_volume DOUBLE,
    taker_buy_quote_asset_volume DOUBLE,
    symbol VARCHAR,
    PRIMARY KEY (symbol, open_time)
);
"""
)


async def get_historical_klines(client, symbol, interval, start_str, end_date_dt):
    """Helper function to get historical klines and handle rate limits."""
    klines = []
    
    
    current_start_dt = pd.to_datetime(start_str).tz_localize(None)

    while True:
        
        if current_start_dt >= end_date_dt:
            logging.info(
                f"Reached end date {end_date_dt} for {symbol}. Stopping fetch."
            )
            break
        try:
            
            current_start_api_str = current_start_dt.strftime("%Y-%m-%d %H:%M:%S")
            logging.info(
                f"Fetching klines for {symbol} from {current_start_api_str}..."
            )
            
            end_api_str = end_date_dt.strftime("%Y-%m-%d %H:%M:%S")
            new_klines = await client.get_historical_klines(
                symbol, interval, current_start_api_str, end_api_str, limit=1000
            )

            if not new_klines:
                
                logging.info(
                    f"No more klines returned for {symbol} starting from {current_start_api_str}. Stopping fetch."
                )
                break

            
            first_kline_time_ms = new_klines[0][0]
            first_kline_dt = pd.to_datetime(first_kline_time_ms, unit="ms").tz_localize(
                None
            )

            if first_kline_dt >= end_date_dt:
                logging.info(
                    f"First kline time {first_kline_dt} is on or after end date {end_date_dt} for {symbol}. Stopping fetch before adding this batch."
                )
                break

            
            klines.extend(new_klines)

            
            last_kline_time_ms = klines[-1][
                0
            ]  
            next_start_dt = pd.to_datetime(
                last_kline_time_ms, unit="ms"
            ) + pd.Timedelta(milliseconds=1)
            next_start_dt = next_start_dt.tz_localize(None)  

            logging.info(
                f"Fetched {len(new_klines)} klines for {symbol}. Last kline time: {pd.to_datetime(last_kline_time_ms, unit='ms')}. Next start: {next_start_dt}"
            )

            
            current_start_dt = next_start_dt

            
            await asyncio.sleep(0.1)  
        except BinanceAPIException as e:
            logging.error(f"Binance API Exception for {symbol}: {e}")
            if e.code == -1003:  
                logging.warning("Rate limit exceeded. Sleeping for 60 seconds...")
                await asyncio.sleep(60)
            else:
                
                await asyncio.sleep(5)  
        except Exception as e:
            logging.error(f"An unexpected error occurred for {symbol}: {e}")
            await asyncio.sleep(5)  
            
    return klines


def process_and_insert_klines(klines, symbol):
    """Processes klines data and inserts it into DuckDB."""
    if not klines:
        logging.info(f"No klines data to process for {symbol}.")
        return

    df = pd.DataFrame(
        klines,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )

    
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
    df["number_of_trades"] = df["number_of_trades"].astype(int)

    
    df["symbol"] = symbol

    
    df = df[
        [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "symbol",
        ]
    ]

    
    
    try:
        con.register("df_temp", df)
        con.execute(
            f"""
            INSERT INTO {TABLE_NAME}
            SELECT * FROM df_temp
            ON CONFLICT (symbol, open_time) DO NOTHING;
        """
        )
        con.unregister("df_temp")
        logging.info(
            f"Successfully inserted/updated {len(df)} rows for {symbol} into {TABLE_NAME}."
        )
    except Exception as e:
        logging.error(f"Error inserting data for {symbol}: {e}")


async def main():
    client = await AsyncClient.create()

    
    
    today_utc = datetime.utcnow().date()
    end_date = datetime(today_utc.year, today_utc.month, today_utc.day, tzinfo=None)

    
    start_date = end_date - timedelta(days=DAYS_TO_FETCH)
    start_str = start_date.strftime("%Y-%m-%d %H:%M:%S")

    tasks = []
    for asset in ASSETS:
        logging.info(
            f"Initiating download for {asset} from {start_str} up to {end_date}..."
        )
        
        tasks.append(
            asyncio.create_task(
                get_historical_klines(client, asset, INTERVAL, start_str, end_date)
            )
        )

    all_klines_results = await asyncio.gather(*tasks)

    await client.close_connection()  

    
    for i, asset in enumerate(ASSETS):
        klines_data = all_klines_results[i]
        if klines_data:
            process_and_insert_klines(klines_data, asset)
        else:
            logging.warning(f"No data retrieved for {asset}.")

    con.close()  
    logging.info("Data download and insertion complete.")


if __name__ == "__main__":
    asyncio.run(main())
