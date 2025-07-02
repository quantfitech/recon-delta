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

def get_corrections():
    query_recon = text("""
        SELECT * FROM ReconciliationCorrections;
    """)


    with sql.connect() as db:
        result = db.execute(query_recon)
        rows = result.fetchall()
        df = pd.DataFrame(rows, columns=result.keys()) if rows else pd.DataFrame()

    return df

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
