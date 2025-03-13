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
    query_volumes = text("""
                         SELECT * FROM RequestForQuotes
                         WHERE acceptedAt IS NOT NULL 
                         AND executedAt IS NOT NULL;
                         """)

    with sql.connect() as db:
        result = db.execute(query_volumes)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df

def get_orders(df):
    filtered_df = df[df['acceptedAt'].notna()]

    dates = filtered_df['acceptedAt'].values
    side = filtered_df['side']
    # Calculate 'USDT' based on 'side'
    usdt = np.where(side == 'BUY',
                    -1 * filtered_df['quoteQuantity'] * filtered_df['exchangeRate'],
                    filtered_df['quoteQuantity'] * filtered_df['exchangeRate'])

    # Create DataFrame with 'Date' and 'USDT'
    df = pd.DataFrame({'Date': dates, 'USDT': usdt, 'side': side})
    return df


def data_visualization(df):
    # Ensure 'Date' column is treated as datetime
    time_col = 'Date'
    side_col = 'side'
    df[time_col] = pd.to_datetime(df[time_col])

    for col in df.columns:
        if col not in [time_col, side_col]:  # Skip 'Date' column
            plt.plot(df[time_col], df[col])
            plt.xlabel('Date')
            plt.ylabel('USDT')
            plt.title('USDT Flow Over Time')
            plt.show()
            plt.plot(df['Date'], df[col], 'o')
            plt.show()
            plt.hist(df[col])
            plt.show()
    # Basic Information
    print("Basic Information:")
    print(df.info(), "\n")

    # Descriptive Statistics
    print("Descriptive Statistics (Numerical Columns):")
    print(df.describe(), "\n")

    # 1. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()
    #
    # sys.exit(1)
    #
    # # 2. Histogram for Numeric Columns
    # plt.figure(figsize=(8, 6))  # Set the figure size
    # plt.hist(df['USDT'], edgecolor='black')  # 30 bins and black edges for bars
    # plt.title('USDT')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.show()
    #
    # # 3. Boxplot for Outlier Detection
    # plt.figure(figsize=(12, 6))
    # sns.boxplot(data=df.select_dtypes(include='number'))
    # plt.title("Boxplot for Outliers")
    # plt.xticks(rotation=45)
    # plt.show()
    #
    # print('check point 1')
    #
    # # 4. Line Plot (Time-Series) - Use 'Date' as the x-axis
    # plt.figure(figsize=(12, 6))
    # print(df.columns)
    #
    # print('check point 2')
    # plt.title("Line Plot with Time as X-axis")
    # plt.xlabel(time_col)
    # plt.ylabel("Values")
    # plt.legend()
    # plt.xticks(rotation=45)
    # plt.grid(True)
    # plt.show()
    #
    # # Clear all plots after showing
    # plt.close('all')


def group_data(data: pd.DataFrame, date_col: str, interval: str = '1T') -> pd.DataFrame:
    """
    Groups data based on the specified time interval.

    """
    # Ensure 'Date' column is in datetime format
    data[date_col] = pd.to_datetime(data[date_col], utc=True)

    if 'side' in data.columns:  # Check if 'side' column exists
        df_grouped = (data.groupby('side')  # Group by 'side'
                      .resample(interval, on=date_col)
                      .sum()
                      .drop(columns=['side'])  # Drop date column
                      .reset_index())  # Reset index
    else:
        df_grouped = (data.resample(interval, on=date_col)  # Resample without grouping
                      .sum()
                      .reset_index())  # Reset index

    # Order by date_col
    data = df_grouped.sort_values(by=date_col)
    return data

def merge_data(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the 'close' column of df2 into df1 based on 'date' in df1 and 'time' in df2.

    """
    # Ensure 'Date' and 'time' columns are in datetime format
    df1['date'] = pd.to_datetime(df1['Date'])
    df2['time'] = pd.to_datetime(df2['time'])

    # Merge the dataframes based on matching dates
    merged_df = pd.merge(df1, df2[['time', 'close']],
                         left_on='date',
                         right_on='time',
                         how='left')

    merged_df.drop(columns=['time', 'date'], inplace=True)

    return merged_df


def main():

    # df = pull_positions()
    # df = get_orders(df)
    # df.to_csv('usdt_flow.csv', index=False)  # Save as CSV
    df = pd.read_csv('usdt_flow.csv')
    df = df.sort_values(by='Date')

    df_mkt= pd.read_csv('CRYPTOCAP_TOTAL, hourly.csv')
    # printdf(df_mkt.head(10))
    # df_mkt['time'] = pd.to_datetime(df_mkt['time'], utc=True)

    df_mkt = group_data(df_mkt.copy(), 'time', interval='1H')
    print(f'the market hourly price is \n {df_mkt.head(20)}')

    df_grouped = group_data(df.copy(), 'Date','1H')
    # print(df_grouped[df_grouped['USDT']<0])

    forecast(df_grouped,frequency = 'D' )
    # sys.exit(1)


    data_visualization(df)
    # data_visualization(df_grouped)
    print(df_grouped)

    sys.exit(1)

    # printdf(df_grouped.head(20))
    df_merged = merge_data(df_grouped, df_mkt)
    # printdf(f'the merged hourly data \n {df_merged[:1].head(10)}')
    printdf(df_merged.head(10) )



    sys.exit(1)



if __name__ == "__main__":
    main()