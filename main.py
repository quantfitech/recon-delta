import numpy as np
import pandas as pd
import threading
import time
import os
import sys
import warnings
import urllib
import http
import redis
import asyncio
import datetime
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Ignore warnings
warnings.filterwarnings("ignore")
load_dotenv('.env')

#GLOBALS
REDIS_HOST = os.getenv('REDIS_HOST')
REDIS_PORT = os.getenv('REDIS_PORT')

ALERT_TOKEN = os.getenv('ALERT_TOKEN')
ALERT_USER = os.getenv('ALERT_USER')

USERNAME = os.getenv('MYSQL_USER')
PASSWORD = os.getenv('MYSQL_PASSWORD')
HOST = os.getenv('MYSQL_HOST')
DATABASE = os.getenv('MYSQL_DATABASE')

sql = create_engine(f'mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}/{DATABASE}')

set_universal_delta_limit = 100 #SET UNIVERSAL LIMITS FOR TESTING
set_universal_pnl_limit = -1 #SET UNIVERSAL LIMITS FOR TESTING


# Async Redis client initializer
async def init_redis_client():
    return redis.asyncio.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

# Async function to fetch data from Redis
async def get_data_from_redis(redis_key: str, field: str) -> float:
    try:
        redis_client = await init_redis_client()
        data = await redis_client.hget(redis_key, field)
        if data is not None:
            return float(data)
        else:
            return 0.0
    except Exception as e:
        print(f"Error fetching data from Redis: {e}")
        return 0.0

# Send alerts to Pushover
def send_alert(title, message, priority=1):
    # data = urllib.parse.urlencode({
    #     "token": ALERT_TOKEN,
    #     "user": ALERT_USER,
    #     "title": title,
    #     "message": message,
    #     "priority": priority
    # })
    # conn = http.client.HTTPSConnection("api.pushover.net:443")
    # conn.request("POST", "/1/messages.json", data, {"Content-type": "application/x-www-form-urlencoded"})
    # print(conn.getresponse())
    print(title + ' ' + message)

# Pull positions from MySQL
def pull_positions():
    query_volumes = text("""
                         SELECT * FROM RequestForQuotes
                         WHERE acceptedAt IS NOT NULL 
                         AND executedAt IS NULL
                         GROUP BY baseAsset;
                         """)

    with sql.connect() as db:
        result = db.execute(query_volumes)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df

# Pull PnL threshold
def pull_treshold_pnl(ticker):
    try:
        limit = float(set_universal_pnl_limit)
    except (NameError, TypeError):
        limit = -1
    return limit

# Pull Delta threshold
def pull_treshold_delta(ticker):
    try:
        limit = float(set_universal_delta_limit)
    except (NameError, TypeError):
        limit = 100
    return limit

# Calculate Delta
def check_delta(basket):
    delta = np.nansum(basket.quoteQuantity[basket.side == 'BUY']) - np.nansum(basket.quoteQuantity[basket.side == 'SELL'])
    return delta

# Calculate VWAP
def check_vwap(basket):
    value = np.nansum(basket.quoteQuantity[basket.side == 'BUY']) - np.nansum(basket.quoteQuantity[basket.side == 'SELL'])
    quantity = np.nansum(basket.quantity[basket.side == 'BUY']) - np.nansum(basket.quantity[basket.side == 'SELL'])
    vwap = value / quantity
    return float(vwap)

# Async price fetcher using Redis
async def pull_price(ticker):
    current_price = await get_data_from_redis("FALLBACK_PRICES", f"{ticker}/EUR")
    return current_price

# Create baskets of positions
def create_baskets(table):
    baskets = []
    for i in table.baseAsset.unique():
        baskets.append(table[table['baseAsset'] == i])
    return baskets

# Create baskets of positions
def insert_into(ticker, time, current_delta, vwap, current_pnl, treshold_pnl, treshold_delta):
    query_volumes = text(f"""
                        INSERT INTO OpenInventory (ticker, time, current_delta, 
                        vwap, current_pnl, treshold_pnl, treshold_delta)
                        VALUES ('{ticker}', '{time}', {current_delta}, {vwap}, 
                        {current_pnl}, {treshold_pnl}, {treshold_delta});
                        """)
    print('Query Volumes', query_volumes)
    with sql.connect() as db:
        result = db.execute(query_volumes)
    return result

# Async checker function
async def checker(baskets):
    for i in baskets:
        try:
            ticker = i.baseAsset.iloc[0]
            price = await pull_price(ticker)  # Await price retrieval

            current_delta = check_delta(i)
            vwap = check_vwap(i)            
            current_pnl = np.round(((price - vwap) / vwap) * int(current_delta / abs(current_delta)) * 100, 2)

            treshold_pnl = pull_treshold_pnl(ticker)
            treshold_delta = pull_treshold_delta(ticker)
            print(ticker)
            print(vwap)
            print(price)
            if current_pnl < treshold_pnl and abs(current_delta) < treshold_delta:
                title = f'SIM Alert - {ticker} - PNL'
                message = f"Asset {ticker}: PnL Position {current_pnl}. Value Position {current_delta}."
                send_alert(title, message)

            elif current_pnl > treshold_pnl and abs(current_delta) > treshold_delta:
                title = f'SIM Alert - {ticker} - DELTA'
                message = f"Asset {ticker}: PnL Position {current_pnl}. Value Position {current_delta}."
                send_alert(title, message)

            elif current_pnl < treshold_pnl and abs(current_delta) > treshold_delta:
                title = f'SIM Alert - {ticker} - PNL AND DELTA'
                message = f"Asset {ticker}: PnL Position {current_pnl}. Value Position {current_delta}."
                send_alert(title, message)
            
            #insert into database
            insert_into(ticker,  datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S'), current_delta, vwap, current_pnl, treshold_pnl, treshold_delta)

        except Exception as e:
            print(f'Could not calculate {ticker}: {e}')

# Main async function
async def main():
    tik = time.time()
    table = pull_positions()

    table.to_csv('positions.csv', index=False)

    baskets = create_baskets(table)

    for basket in baskets:
        if 'USDT' in basket['baseAsset'].values:  # Check if 'USDT' exists in the DataFrame
            usdt_basket = basket
            usdt_basket.to_csv('usdt_basket.csv', index=False)  # Save as CSV
            print("Saved usdt_basket.csv")
            break

    await checker(baskets)  
    tok = time.time()
    await asyncio.sleep(max(0, 60 * 5 - (tok - tik)))  

if __name__ == "__main__":
    asyncio.run(main()) 