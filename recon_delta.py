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
    df.columns = df.iloc[0]
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

def compute_diff_native(df):
    df_filtered = df[~df['account'].isin(['SI', 'YIELD_FARM'])]

    desired_quantity_sum = df_filtered.groupby(['asset','price', 'timestamp', 'epoch'])['desired_quantity'].sum().reset_index()
    desired_quantity_sum.rename(columns={'desired_quantity': 'sum_desired_quantity'}, inplace=True)

    current_quantity_sum = df.groupby(['asset', 'price', 'timestamp', 'epoch'])['current_quantity'].sum().reset_index()
    current_quantity_sum.rename(columns={'current_quantity': 'sum_current_quantity'}, inplace=True)

    df_merged = current_quantity_sum.merge(desired_quantity_sum, on=['asset', 'price', 'timestamp', 'epoch'], how='left')

    df_merged['diff_native'] = df_merged['sum_current_quantity'] - df_merged['sum_desired_quantity'].fillna(0)
    df_merged['diff_nominal'] = df_merged['diff_native'] * df_merged['price']
    df_merged['current_nominal'] = df_merged['sum_current_quantity'] * df_merged['price']


    keep_columns = [
         "asset", "price", "sum_current_quantity",
        "current_nominal", "diff_native", "diff_nominal", "epoch", "timestamp"
    ]

    df = df_merged[keep_columns]
    # df_merged['diff_native'] = df_merged['diff_native'].apply(lambda x: f"{x:,.0f}")
    printdf(df.head(10))

    return df

def df_split(df,stablecoins):

    df_stable = df[df['asset'].isin(stablecoins)]
    df_crypto = df[~df['asset'].isin(stablecoins)]

    print("Stablecoins DataFrame:")
    printdf(df_stable)

    print("\nCryptocurrency DataFrame:")
    printdf(df_crypto)
    return df_stable, df_crypto

def delta_overview(df_t, df_t_1, eur_usd_t,eur_usd_t_1):
    aum_t = df_t['current_nominal'].sum()
    equity_t = df_t['diff_nominal'].sum()
    aum_t_1 = df_t_1['current_nominal'].sum()
    equity_t_1 = df_t_1['diff_nominal'].sum()
    delta = (equity_t - equity_t_1) / eur_usd_t

# def corrections

def main():
    stablecoins = ['EUR', 'FDUSD', 'USD', 'USDC', 'USDT']
    epoch_t = 24143
    epoch_t_1 = 23838

    # df_t = pull_positions_raw(epoch_t)
    # df_t_1 = pull_positions_raw(epoch_t_1)

    # df.to_csv('epoch_t.csv', index=False)
    # sys.exit(1)
    df_t = pd.read_csv('epoch_t.csv', header = None)
    df_t_1 = pd.read_csv('epoch_t_1.csv', header = None)


    df_t = data_process(df_t)
    df_t = compute_diff_native(df_t)
    eur_usd_t = df_t.loc[df_t['asset'] == 'EUR', 'price'].values
    df_stable_t, df_crypto_t = df_split(df_t,stablecoins)

    df_t_1 = data_process(df_t_1)
    df_t_1 = compute_diff_native(df_t_1)
    eur_usd_t_1 = df_t_1.loc[df_t_1['asset'] == 'EUR', 'price'].values
    df_stable_t_1, df_cryptot_1 = df_split(df_t_1,stablecoins)

    print(eur_usd_t,eur_usd_t_1)
    delta_overview(df_crypto_t, df_cryptot_1, eur_usd_t,eur_usd_t_1)








if __name__ == "__main__":
    main()