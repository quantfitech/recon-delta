import sys

import pandas as pd
from recon_delta import printdf as printdf
threshold = 1000

def check_delta(df):
    df['delta_native'] = df['diff_native_t'] - df['diff_native_t_1']
    df['delta_nominal'] = df['delta_native'] * df['price_t']

    df['delta']  = abs(df['delta_nominal']) > threshold/2

    df_breaks = df[df['delta']][['asset', 'diff_native_t', 'diff_native_t_1', 'delta', 'delta_native', 'price_t', 'delta_nominal', 'timestamp_t']].copy()
    df_breaks = df_breaks.reindex(df_breaks['delta_nominal'].abs().sort_values(ascending=False).index)
    df_breaks.to_csv('recon_break_list.csv', index=False)

    return df_breaks

def expected_flow(start_date, end_date):
    df_in = pd.read_csv('daily_incoming_rewards_march.csv', header=0)
    df_in['flow_native'] = df_in['Earn In Native Cm'] + df_in['Staking In Native Cm']
    df_in = df_in.rename(columns={
        'Transaction Date': "date",
        'Asset': "asset",
    })

    df_out = pd.read_csv('daily_outgoing_rewards_march.csv', header= 0)
    df_out['flow_native'] = -df_out['Sum Earn Reward Crypto'] - df_out['Sum Stake Reward Crypto']

    df_out = df_out.rename(columns={
        'Deposit Time Date': "date",
        'Coin': "asset",
    })

    keep_columns = ["date", "asset", "flow_native"]
    df_in = df_in[keep_columns]
    df_out = df_out[keep_columns]

    df_flow = pd.concat([df_in, df_out], ignore_index=True)
    df_flow = df_flow[
        (df_flow['date'] >= start_date) &
        (df_flow['date'] <= end_date)
        ].copy()
    df_total = df_flow.groupby(['asset'], as_index=False)['flow_native'].sum()
    df_total.to_csv('income_rewards.csv', index=False)

    return df_total

def check_breaks_income(df, start_date, end_date):
    df_in_out_flow = expected_flow(start_date, end_date).copy()

    # Compute deltas and merge income flows
    df_breaks = check_delta(df).copy()
    df_breaks = df_breaks.merge(df_in_out_flow, on='asset', how='left')
    df_breaks['in_out_flow'] = df_breaks['flow_native'].fillna(0)
    df_breaks.drop(columns=['flow_native'], inplace=True)
    df_breaks['breaks_native'] = df_breaks['delta_native'] - df_breaks['in_out_flow']
    df_breaks['breaks_nominal'] = df_breaks['breaks_native'] * df_breaks['price_t']
    df_breaks['breaks']  = abs(df_breaks['breaks_nominal']) > threshold
    keep_columns = [ "asset", "diff_native_t", "diff_native_t_1", "delta_native", "in_out_flow", "breaks_native" , "breaks_nominal", "timestamp_t"]
    df_breaks = df_breaks[keep_columns]

    df_breaks = df_breaks.reindex(df_breaks['breaks_nominal'].abs().sort_values(ascending=False).index)

    printdf(df_breaks)
    return df_breaks

