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
from forecast import forecast
from sklearn.datasets import load_iris

from datetime import datetime, timezone


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
stablecoins = ['EUR', 'FDUSD', 'USD', 'USDC', 'USDT']


def printdf(df: pd.DataFrame) -> None:
    # Get terminal width dynamically
    max_width = shutil.get_terminal_size().columns

    # Adjust display settings temporarily
    with pd.option_context('display.width', max_width,       # Set max width to terminal size
                           'display.max_columns', None,      # Display all columns
                           'display.expand_frame_repr', True):  # Avoid splitting columns
        # df = pd.DataFrame(df, columns=df.feature_names)
        print(df)


def pull_positions_raw(epoch_nr):
    query_recon = text("""
                    select * from Reconciliations r
                    where epoch = :epoch_nr
                    order by asset;
                         """)

    with sql.connect() as db:
        result = db.execute(query_recon, {"epoch_nr": epoch_nr})
        rows = result.fetchall()
        df = pd.DataFrame(rows, columns=result.keys()) if rows else pd.DataFrame()
    return df

def plot(df):
    x_column = 'epoch'  # Last column
    y_column = 'diff_native' # Second column

    plt.figure(figsize=(10, 5))
    plt.plot(df[x_column], df[y_column], linestyle='-', label=y_column)  # Removed marker='o'

    # plt.xticks([df[x_column].iloc[0], df[x_column].iloc[-1]],
    #            [df[x_column].iloc[0], df[x_column].iloc[-1]])
    additional_ticks = np.linspace(df[x_column].min(), df[x_column].max(), num=15)  # 5 + first & last

    # Set x-ticks including first, last, and additional ones
    plt.xticks(additional_ticks.astype(int), additional_ticks.astype(int))

    # Set first and last y-ticks
    plt.yticks([df[y_column].iloc[0], df[y_column].iloc[-1]],
               [df[y_column].iloc[0], df[y_column].iloc[-1]])
    # plt.xlabel(x_column)
    # plt.ylabel(y_column)
    plt.title(f'The Native Diff For { df['asset'].iloc[0] }')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

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

def df_split(df, stablecoins):

    df_stable = df[df['asset'].isin(stablecoins)]
    df_crypto = df[~df['asset'].isin(stablecoins)]

    # print("Stablecoins DataFrame:")
    # printdf(df_stable)
    #
    # print("\nCryptocurrency DataFrame:")
    # printdf(df_crypto)
    return df_stable, df_crypto

def delta_overview(df, eur_usd_t,eur_usd_t_1):
    aum_t = df['current_nominal_t'].sum() / eur_usd_t
    equity_t = (df['diff_native_t'] * df['price_t']).sum()/ eur_usd_t
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
    df_stable, df_crypto = df_split(df, stablecoins)
    return df, eur_usd, df_stable, df_crypto

def main():
    epoch_t = 26447
    epoch_t_1 = 26161

    df_t = pull_positions_raw(epoch_t)
    df_t_1 = pull_positions_raw(epoch_t_1)

    df_t.to_csv('epoch_t.csv', index=False)
    df_t_1.to_csv('df_t_1.csv', index=False)

    df_correction = pd.read_csv('recon_corrections.csv', delimiter=';')

    df_t, eur_usd_t, df_stable_t, df_crypto_t = process_dataframe(df_t, df_correction)

    df_t_1, eur_usd_t_1, df_stable_t_1, df_crypto_t_1 = process_dataframe(df_t_1, df_correction)

    print(f'EUR/USD_T: {eur_usd_t},\nEUR/USD_T-1: {eur_usd_t_1}')
    df_crypto_2t = df_crypto_t.merge(df_crypto_t_1, on="asset", how="outer", suffixes=("_t", "_t_1")).fillna(0)

    print(f'\nThe crypto delta between {df_crypto_2t['timestamp_t_1'][0]} and {df_crypto_2t['timestamp_t'][0]}is:')
    delta_dict = delta_overview(df_crypto_2t, eur_usd_t,eur_usd_t_1)
    # print(delta_dict)

    df_stable_2t = df_stable_t.merge(df_stable_t_1, on="asset", how="outer", suffixes=("_t", "_t_1")).fillna(0)
    print(f'\nThe stable delta between {df_crypto_2t['timestamp_t_1'][0]} and {df_crypto_2t['timestamp_t'][0]}is:')
    delta_dict = delta_overview(df_stable_2t, eur_usd_t ,eur_usd_t_1 )







if __name__ == "__main__":
    main()