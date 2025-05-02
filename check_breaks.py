import sys

import pandas as pd
from recon_delta import printdf as printdf
from google.cloud import bigquery

threshold = 2000

def check_delta(df):
    df['delta_native'] = df['diff_native_t'] - df['diff_native_t_1']
    df['delta_nominal'] = df['delta_native'] * df['price_t']

    df['delta']  = abs(df['delta_nominal']) > threshold

    df_breaks = df[df['delta']][['asset', 'diff_native_t', 'diff_native_t_1', 'delta', 'delta_native', 'price_t', 'delta_nominal', 'timestamp_t']].copy()
    df_breaks = df_breaks.reindex(df_breaks['delta_nominal'].abs().sort_values(ascending=False).index)
    # df_breaks.to_csv('recon_break_list.csv', index=False)

    return df_breaks

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
    filtered = df_breaks[df_breaks['breaks_nominal'].abs() > threshold]

    return filtered['asset']

def expected_flow(start_date, end_date):

    client = bigquery.Client(project="bigdata-staging-cm")
    query = "SELECT * FROM `bigdata-staging-cm.treasury_ml.coinmerce_incoming_outgoing_rewards_daily`"
    df = client.query(query).to_dataframe()

    df_flow = df[
        (df['date'] >= start_date) &
        (df['date'] <= end_date)
        ].copy()
    df_flow = df_flow.rename(columns={
        'manual_input_finance_earn_in_native_cm': "earn_in",
        'manual_input_finance_staking_in_native_cm': "stake_in",
        'rewards_bonus_sum_earn_reward_crypto': "earn_out",
        'rewards_bonus_sum_stake_reward_crypto': "stake_out",

    })

    df_flow['flow_in'] = df_flow['earn_in'] + df_flow['stake_in']
    df_flow['flow_out'] = df_flow['earn_out'] + df_flow['stake_out']
    df_flow['fiat_in'] = df_flow['manual_input_finance_earn_in_fiat_cm'] + df_flow['manual_input_finance_staking_in_fiat_cm']
    df_flow['fiat_out'] = df_flow['rewards_bonus_sum_earn_reward_fiat'] + df_flow['rewards_bonus_sum_stake_reward_fiat']

    df_flow['flow_native'] = (
            df_flow['flow_in'].fillna(0).astype(float) - df_flow['flow_out'].fillna(0).astype(float)
    )

    df_flow['flow_fiat'] = (
            df_flow['fiat_in'].fillna(0).astype(float) - df_flow['fiat_out'].fillna(0).astype(float)
    )
    keep_columns = ["date", "asset", "flow_native", "flow_fiat"]

    df_flow = df_flow[keep_columns]
    df_total = df_flow.groupby(['asset'], as_index=False)[['flow_native','flow_fiat']].sum()
    print(f'\nThe total net income between {start_date} and { end_date} is : {df_total['flow_fiat'].sum()} USD')
    return df_total