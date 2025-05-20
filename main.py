import sys

import numpy as np
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import shutil
from check_breaks import *
from sklearn.datasets import load_iris

from datetime import datetime, timezone, timedelta

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

stable = ['EUR', 'FDUSD', 'USD', 'USDC', 'USDT']
n_data= 20

def printdf(df: pd.DataFrame) -> None:
    # Get terminal width dynamically
    max_width = shutil.get_terminal_size().columns

    # Adjust display settings temporarily
    with pd.option_context('display.width', max_width,       # Set max width to terminal size
                           'display.max_columns', None,      # Display all columns
                           'display.expand_frame_repr', True):  # Avoid splitting columns
        # df = pd.DataFrame(df, columns=df.feature_names)
        print(df)

def pull_positions_raw(nday):
    # Target timestamp: previous day at 22:00:00
    target_time = datetime.now().replace(hour=22, minute=0, second=0, microsecond=0) - timedelta(days=nday)

    query = text("""
        SELECT * FROM Reconciliations r
        WHERE r.timestamp = (
            SELECT MAX(timestamp)
            FROM Reconciliations
            WHERE timestamp <= :target_time
        )
        ORDER BY asset;
    """)

    with sql.connect() as db:
        result = db.execute(query, {"target_time": target_time})
        rows = result.fetchall()
        df = pd.DataFrame(rows, columns=result.keys()) if rows else pd.DataFrame()

    return df

def data_process(df):

    df = df.reset_index(drop=True)

    keep_columns = [
               "account", "asset", "price", "current_quantity", "desired_quantity",
               "expected_quantity","native_difference","nominal_difference","nominal_quantity","epoch","timestamp"
    ]

    df = df[keep_columns]

    float_columns = [
        "price", "current_quantity", "desired_quantity", "expected_quantity","native_difference","nominal_difference","nominal_quantity"
    ]

    df[float_columns] = df[float_columns].apply(pd.to_numeric, errors='coerce')
    # df[float_columns] = df[float_columns].applymap(lambda x: f"{x:,.0f}" if pd.notna(x) else "0")
    return df

def overwrite_values(df, df_correction):

    # Iterate through df2 and update df1 where asset and epoch match
    for _, row in df_correction.iterrows():
        mask = (df['epoch'] == row['epoch']) & (df['asset'] == row['asset'])
        for col in df_correction.columns:
            if col not in ['epoch', 'asset'] and pd.notna(row[col]):
                df.loc[mask, col] = row[col]
    return df

def compute_diff_native(df):
    df_filtered = df[~df['account'].isin(['SI', 'YIELD_FARM'])]

    desired_quantity_sum = df_filtered.groupby(['asset','price', 'timestamp', 'epoch'])['desired_quantity'].sum().reset_index()
    desired_quantity_sum.rename(columns={'desired_quantity': 'desired_quantity'}, inplace=True)

    current_quantity_sum = df.groupby(['asset', 'price', 'timestamp', 'epoch'])['current_quantity'].sum().reset_index()
    current_quantity_sum.rename(columns={'current_quantity': 'current_quantity'}, inplace=True)

    df_merged = current_quantity_sum.merge(desired_quantity_sum, on=['asset', 'price', 'timestamp', 'epoch'], how='left')

    df_merged['diff_native'] = df_merged['current_quantity'] - df_merged['desired_quantity'].fillna(0)
    df_merged['diff_nominal'] = df_merged['diff_native'] * df_merged['price']
    df_merged['current_nominal'] = df_merged['current_quantity'] * df_merged['price']

    keep_columns = [
        "asset", "price", "current_quantity",
        "current_nominal", "diff_native", "diff_nominal", "epoch", "timestamp"
    ]

    df = df_merged[keep_columns]
    # df_merged['diff_native'] = df_merged['diff_native'].apply(lambda x: f"{x:,.0f}")
    return df

def df_split(df, stable):
    #  df_crypto: asset not in stable
    df_crypto = df[~df['asset'].isin(stable)]

    #  df_stable: asset in stable
    df_stable = df[df['asset'].isin(stable)]

    #  df_stable_usd: asset in stable excluding 'EUR'
    df_stable_usd = df[df['asset'].isin(stable[1:])]

    #  df_stable_eur: asset == 'EUR'
    df_stable_eur = df[df['asset'] == stable[0]]

    return df_crypto, df_stable, df_stable_usd, df_stable_eur

def delta_overview(df, eur_usd_t,eur_usd_t_1):

    aum_t = df['current_nominal_t'].sum() / eur_usd_t
    equity_t = (df['diff_native_t'] * df['price_t']).sum()/ eur_usd_t

    df["diff_nominal_t"] = df['diff_native_t'] * df['price_t']
    long_equity = df.loc[df["diff_nominal_t"] > 0, "diff_nominal_t"].sum() / eur_usd_t
    short_equity = df.loc[df["diff_nominal_t"] < 0, "diff_nominal_t"].sum() / eur_usd_t
    equity_lag = (df['diff_native_t_1'] * df['price_t']).sum() / eur_usd_t

    aum_t_1 = df['current_nominal_t_1'].sum() / eur_usd_t_1
    equity_t_1 = (df['diff_native_t_1'] * df['price_t_1']).sum() / eur_usd_t_1
    delta_equity = (equity_t - equity_t_1)

    delta_aum = (aum_t - aum_t_1)
    delta_price_diff = ((df['price_t'] - df['price_t_1']) * df['diff_native_t_1']).sum() / eur_usd_t
    delta_pos_diff = ((df['diff_native_t'] - df['diff_native_t_1']) * df['price_t']).sum() / eur_usd_t
    delta_fx_diff = (df['diff_native_t_1'] * df['price_t_1']).sum() / eur_usd_t - (df['diff_native_t_1'] * df['price_t_1']).sum() / eur_usd_t_1
    delta_market_diff = delta_price_diff + delta_fx_diff
    delta_dict = {
        # "aum_t": int(aum_t),
        # "aum_t_1": int(aum_t_1),
        # "delta_aum": int(delta_aum),

        "equity_t": int(equity_t),
        "equity_t_1": int(equity_t_1),

        "long_equity": int(long_equity),
        "short_equity": int(short_equity),
        "equity_lag": int(equity_lag),
        "delta_equity": int(delta_equity),
        "delta_price_diff": int(delta_price_diff),
        "delta_pos_diff": int(delta_pos_diff),
        "delta_fx_diff": int(delta_fx_diff),
        "delta_market_diff": int(delta_market_diff)

    }

    for key, value in delta_dict.items():
        print(f"{key}: {value:,}")
    return delta_dict

def process_dataframe(df,df_correction):
    df = data_process(df)
    df = compute_diff_native(df)

    df = overwrite_values(df, df_correction)
    eur_usd = df.loc[df['asset'] == 'EUR', 'price'].values
    return df, eur_usd

def extract_diff_native_column(df):
    try:
        date_str = pd.to_datetime(df['timestamp'].iloc[0]).strftime('%Y%m%d')
        column_name = f'diff_native_{date_str}'
        return df[['asset', 'diff_native']].rename(columns={'diff_native': column_name})
    except Exception:
        return df

def get_processed_df(df,df_correction):
    timestamp = df['timestamp'].iloc[0]
    df, fx = process_dataframe(df, df_correction)
    return df, fx, timestamp


def main():

    df_t_raw = pull_positions_raw(nday=1)
    df_t_1_raw = pull_positions_raw(nday=2)
    epoch_t = df_t_raw['epoch'][0]
    epoch_t_1 = df_t_1_raw['epoch'][0]
    df_correction = pd.read_csv('recon_corrections.csv', delimiter=';')
    # print(df_correction[df_correction['epoch'].isin([epoch_t, epoch_t_1])][['asset', 'diff_native', 'epoch', 'comments']])
    df_t, fx_t, time_t = get_processed_df(df_t_raw, df_correction)
    df_t_1, fx_t_1, time_t_1 = get_processed_df(df_t_1_raw, df_correction)


####################### SPLIT THE DATA TO CRYPTO AND STABLES #################################################################

    df_crypto_t, df_stable_t, df_stable_usd_t, df_stable_eur_t = df_split(df_t, stable)
    df_crypto_t_1, df_stable_t_1, df_stable_usd_t_1, df_stable_eur_t_1 = df_split(df_t_1, stable)

    print(f'EUR/USD_T: {fx_t},\nEUR/USD_T-1: {fx_t_1}')


####################### DELTA OVERVIEW #############################################################################################

    df_crypto_2t = df_crypto_t.merge(df_crypto_t_1, on="asset", how="outer", suffixes=("_t", "_t_1")).fillna(0)

    datasets = [
        ("crypto", df_crypto_t, df_crypto_t_1),
        ("stable", df_stable_t, df_stable_t_1),
        ("USD stable", df_stable_usd_t, df_stable_usd_t_1),
        ("EUR stable", df_stable_eur_t, df_stable_eur_t_1),
    ]

    for label, df_t_p, df_t_1_p in datasets:
        df_merged = df_t_p.merge(df_t_1_p, on="asset", how="outer", suffixes=("_t", "_t_1")).fillna(0)
        print(f"\nThe {label} delta between {time_t_1} and {time_t} is:")
        delta_overview(df_merged, fx_t, fx_t_1)

####################### BREAKS CHECK #############################################################################################

    asset_to_check.extend(check_breaks_income(df_crypto_2t, time_t_1, time_t+timedelta(days=1)))


if __name__ == "__main__":
    main()