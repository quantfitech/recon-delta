import numpy as np

def sim_profit_cal_assets (df_t, df_t_1, epoch_t, epoch_t_1):

    sim_balance_t = df_t.loc[(df_t['account'] == 'SI') & (df_t['epoch'] == epoch_t), 'nominal_difference'].sum()

    sim_balance_t_1 = df_t_1.loc[(df_t_1['account'] == 'SI') & (df_t_1['epoch'] == epoch_t_1), 'nominal_difference'].sum()
    sim_profit = (sim_balance_t - sim_balance_t_1)
    return sim_profit

def sim_profit_cal_usdt (df_t, df_t_1, epoch_t, epoch_t_1):

    sim_balance_t = ((df_t.loc[(df_t['account'] == 'SI') & (df_t['epoch'] == epoch_t) & (df_t['asset'] == 'USDT'), 'nominal_difference'].sum())
                    + (df_t.loc[(df_t['account'] == 'SI') & (df_t['epoch'] == epoch_t) & (df_t['asset'] == 'USDT'), 'nominal_pending'].sum()))
    sim_balance_t_1 = (df_t_1.loc[(df_t_1['account'] == 'SI') & (df_t_1['epoch'] == epoch_t_1)& (df_t['asset'] == 'USDT'), 'nominal_difference'].sum())
    + (df_t.loc[
           (df_t['account'] == 'SI') & (df_t['epoch'] == epoch_t) & (df_t['asset'] == 'USDT'), 'nominal_pending'].sum())

    sim_profit = (sim_balance_t - sim_balance_t_1)
    return sim_profit

def sim_volumne(df):
    total_volume = df['quoteQuantity'].sum()
    return total_volume

def YF_profit_cal (df):
    yf_profit = df['quantity'].sum()
    yf_profit = float(yf_profit)
    return yf_profit

