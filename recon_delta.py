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
from SIM_YF_income import *
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
epoch_t = 39735
epoch_t_1 =39641


# threshold = 2000
n_data= 20
asset_to_check = []

def printdf(df: pd.DataFrame) -> None:
    # Get terminal width dynamically
    max_width = shutil.get_terminal_size().columns

    # Adjust display settings temporarily
    with pd.option_context('display.width', max_width,       # Set max width to terminal size
                           'display.max_columns', None,      # Display all columns
                           'display.expand_frame_repr', True):  # Avoid splitting columns
        # df = pd.DataFrame(df, columns=df.feature_names)
        print(df)


def pull_positions_raw1(epoch_nr):
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

def pull_yf_mutations_raw(time_t, time_t_1):
    query_recon = text("""
        SELECT * FROM LedgerMutations l
        WHERE timestamp < :time_t 
          AND timestamp > :time_t_1
          AND asset = 'USDT' 
          AND subType = 'FUNDING_FEE';
    """)

    with sql.connect() as db:
        result = db.execute(query_recon, {"time_t": time_t, "time_t_1": time_t_1})
        rows = result.fetchall()
        df = pd.DataFrame(rows, columns=result.keys()) if rows else pd.DataFrame()
    return df

def get_rfqs(start_time, end_time):

    query_volumes = text("""
        SELECT * FROM RequestForQuotes
        WHERE acceptedAt IS NOT NULL
        AND executedAt IS NOT NULL
        AND executedAt BETWEEN :start_time AND :end_time;
    """)

    with sql.connect() as db:
        result = db.execute(query_volumes, {
            "start_time": start_time,
            "end_time": end_time
        })
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df


def plot(df, asset_name, n_data):
    asset = df[df['asset'] == asset_name]

    native_values = asset.drop(columns='asset').squeeze()

    dates = [col.replace('diff_native_', '') for col in native_values.index]
    dates = pd.to_datetime(dates, format='%Y%m%d')

    plt.figure(figsize=(10, 5))
    # sort by date
    sorted_pairs = sorted(zip(dates, native_values.values))
    sorted_pairs = sorted_pairs[-n_data:]
    sorted_dates, sorted_values = zip(*sorted_pairs)
    plt.plot(sorted_dates, sorted_values, marker='o')
    plt.title(f'{asset_name} Native Difference Records')
    plt.xlabel('Date')
    plt.ylabel('Native Value')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
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

def recon_breaks(df, threshold):

    df['break_native'] = df['diff_native_t'] - df['diff_native_t_1']
    df['break_nominal'] = df['break_native'] * df['price_t']

    df['breaks']  = abs(df['break_nominal']) > threshold

    df_breaks = df[df['breaks']][['asset', 'diff_native_t', 'diff_native_t_1', 'breaks', 'break_native', 'break_nominal', 'timestamp_t']].copy()
    df_breaks = df_breaks.reindex(df_breaks['break_nominal'].abs().sort_values(ascending=False).index)

    return df_breaks

def historical_diff(df, df_new):
    new_col_df = extract_diff_native_column(df_new)
    df = extract_diff_native_column(df)

    if new_col_df.empty:
        return df  # skip if nothing to merge

    # Get the name of the new column (besides 'asset')
    new_col_name = [col for col in new_col_df.columns if col != 'asset'][0]

    if new_col_name in df.columns:
        # Replace values in that column for matching assets
        df = df.set_index('asset')
        new_col_df = new_col_df.set_index('asset')

        df.update(new_col_df)  # only updates the existing column
        df = df.reset_index()
    else:
        # New date — merge as new column
        df = df.merge(new_col_df, on='asset', how='outer')

    return df

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

    df_correction = pd.read_csv('recon_corrections.csv', delimiter=';')
    print(df_correction[df_correction['epoch'].isin([epoch_t, epoch_t_1])][['asset', 'diff_native', 'epoch', 'comments']])
    df_t_raw = pull_positions_raw1(epoch_t)

    df_t_1_raw = pull_positions_raw1(epoch_t_1)

    df_t, fx_t, time_t = get_processed_df(df_t_raw, df_correction)
    df_t_1, fx_t_1, time_t_1 = get_processed_df(df_t_1_raw, df_correction)

####################### YIELD FARMING AND SIM PROFITS #################################################################
    print('-------------------------------')
    df_orders= get_rfqs(time_t_1, time_t)
    total_volume = sim_volumne(df_orders)
    sim_profit = sim_profit_cal_assets(df_t_raw, df_t_1_raw, epoch_t, epoch_t_1)
    sim_profit_usdt = sim_profit_cal_usdt(df_t_raw, df_t_1_raw, epoch_t, epoch_t_1)

    print(f'the SIM profit between {time_t_1} and {time_t} in USD is: {sim_profit} (ref: {sim_profit_usdt})')
    print(f'the estimate for SIM profit based on trades volume [{total_volume}] in USD is between {total_volume *0.8*0.01} and {total_volume *1.0*0.01})')
    df_yf = pull_yf_mutations_raw(time_t, time_t_1)
    yf_profit = YF_profit_cal(df_yf)
    yf_assets_delta = yf_profit_cal_assets(df_t_raw, df_t_1_raw, epoch_t, epoch_t_1)

    print(f'the YIELD FARMING profit between {time_t_1} and {time_t} in USD is: {yf_profit}')
    print(f'the YIELD FARMING assets value movement between {time_t_1} and {time_t} in USD is: {yf_assets_delta}')

    YF_t, SI_t, general_bank_t = get_stable_pos_stable(df_t_raw, stable)
    YF_t_1, SI_t_1, general_bank_t_1 = get_stable_pos_stable(df_t_1_raw, stable)

    print(f'the GENERAL_BANK, YIELD FARMING and SIM balance at {time_t} in USD is: {general_bank_t/fx_t}, {YF_t/fx_t}, {SI_t/fx_t} ')
    print(f'the GENERAL_BANK, YIELD FARMING and SIM balance at {time_t_1} in USD is: {general_bank_t_1/fx_t_1}, {YF_t_1/fx_t_1}, {SI_t_1/fx_t_1}')
    print('-------------------------------')

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
    print(asset_to_check)
    df = pd.read_csv('historical_diff.csv')

    df = historical_diff(df, df_t)
    df.to_csv('historical_diff.csv', index=False)
    # df = pd.read_csv('historical_diff.csv')
    for asset in asset_to_check:
        plot(df, asset, n_data)

if __name__ == "__main__":
    main()