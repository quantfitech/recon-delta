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


def yf_profit_cal_assets (df_t, df_t_1, epoch_t, epoch_t_1):

    yf_balance_t = df_t.loc[(df_t['account'] == 'YIELD_FARM') & (df_t['epoch'] == epoch_t), 'nominal_difference'].sum()

    yf_balance_t_1 = df_t_1.loc[(df_t_1['account'] == 'YIELD_FARM') & (df_t_1['epoch'] == epoch_t_1), 'nominal_difference'].sum()
    yf_profit = (yf_balance_t - yf_balance_t_1)
    return yf_profit

def get_stable_pos_stable(df, stable):

    df_eur = df[df['asset'] == 'EUR']
    coinmerce_account = "COINMERCE" if "COINMERCE" in df_eur['account'].values else "COINMERCE_CUSTODY"
    eur_bank = (
            df_eur.loc[df_eur['account'] == 'BANK', 'current_quantity'].iloc[0] -
            df_eur.loc[df_eur['account'] == coinmerce_account, 'desired_quantity'].iloc[0]
    )

    df_stable = df[df['asset'].isin(stable)]
    df_filtered = df_stable[df_stable['account'].isin(['SI', 'YIELD_FARM'])]
    df_sim_yf = df_filtered.groupby(['account'])['nominal_difference'].sum().reset_index()

    si = df_sim_yf[df_sim_yf['account'] == 'SI']['nominal_difference'].iloc[0]
    yf = df_sim_yf[df_sim_yf['account'] == 'YIELD_FARM']['nominal_difference'].iloc[0]
    # df_merged['diff_native'] = df_merged['diff_native'].apply(lambda x: f"{x:,.0f}")
    return yf, si, eur_bank
