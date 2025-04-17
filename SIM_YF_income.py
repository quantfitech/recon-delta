

def sim_profit_cal (df_t, df_t_1, epoch_t, epoch_t_1, fx):
    sim_balance_t = (df_t.loc[(df_t['account'] == 'SI') & (df_t['epoch'] == epoch_t), 'nominal_difference'].sum()
                     + df_t.loc[(df_t['account'] == 'SI') & (df_t['epoch'] == epoch_t) , 'pending_quantity'].sum())
    sim_balance_t_1 = (df_t_1.loc[(df_t_1['account'] == 'SI') & (df_t_1['epoch'] == epoch_t_1), 'nominal_difference'].sum() +
                    df_t_1.loc[(df_t_1['account'] == 'SI') & (df_t_1['epoch'] == epoch_t_1), 'pending_quantity'].sum())
    sim_profit = (sim_balance_t - sim_balance_t_1) / fx
    return sim_profit

def YF_profit_cal (df,fx):
    yf_profit = df['quantity'].sum()
    yf_profit = float(yf_profit) / fx
    return yf_profit

