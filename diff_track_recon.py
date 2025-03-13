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
        df = pd.DataFrame(df.data,
                          columns=df.feature_names)
        print(df)


def pull_positions():
    query_recon = text("""
                    select asset, round(sum(current_quantity) -
                    (select sum(desired_quantity) from Reconciliations r2
                    where r.epoch = r2.epoch and r.asset = r2.asset and r2.account not in ('SI', 'YIELD_FARM')),0) as diff_native,
                    price, epoch, timestamp
                    from Reconciliations r
                    where asset = 'USDT' and epoch > 9200
                    group by asset, epoch
                    order by
                    epoch asc ,
                    asset;
                         """)
    # query_recon = text("""
    #                     select * from Reconciliations
    #                      """)
    with sql.connect() as db:
        result = db.execute(query_recon)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df

def pull_positions_yf():
    query_recon = text("""
                select asset, epoch, current_quantity , account, timestamp
                from Reconciliations
                where asset = 'USDT' and epoch >12200 and account = 'YIELD_FARM'
                order by  epoch asc;

                         """)
    # query_recon = text("""
    #                     select * from Reconciliations
    #                      """)
    with sql.connect() as db:
        result = db.execute(query_recon)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
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


def main():
    # df = pull_positions()
    # df = pull_positions_yf()
    #
    # df.to_csv('usdt_21feb_yf.csv', index=False)
    df = pd.read_csv('usdt_21febv1.csv', header = None)
    # df = pd.read_csv('usdc_20feb.csv', header = None)

    df.columns = df.iloc[0]  # Set first row as column names
    df = df[1:].reset_index(drop=True)  # Remove the first row and reset index
    # df["current_quantity"] = df["current_quantity"].astype(float)
    df["diff_native"] = df["diff_native"].astype(float)

    df["epoch"] = df["epoch"].astype(float)
    # print(len(df))
    # sys.exit(1)
    plot(df)





if __name__ == "__main__":
    main()