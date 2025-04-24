Asset Delta Reconciliation & Break Detection

This project contains a set of Python scripts designed to assist with crypto asset delta reconciliation, and break detection using SQLAlchemy, and BigQuery. It helps identify inconsistencies in crypto balances across time/epochs and highlights significant changes.

1. Project Structure

-  recon_delta.py:
Main script for reconciling asset deltas, querying databases, plotting charts, and integrating other modules.
-  check_breaks.py:
Contains functions to detect large changes (“breaks”) in asset deltas and summarize anomalies.
-  SIM_YF_income.py:
Implements profit calculation logic based on account balance changes between two epochs.

2. Requirements

Make sure you have Python 3.8+ and the following packages installed:

pandas
numpy
matplotlib
seaborn
sqlalchemy
python-dotenv
google-cloud-bigquery

3. Highlights

Delta Reconciliation:
Compares native and nominal asset differences across epochs to verify consistency.
Break Detection (check_breaks.py):
Flags large jumps in nominal balances exceeding a defined threshold (default: 2000).
Profit Simulation (SIM_YF_income.py):
Calculates net profit change over time from SI accounts, usable for income estimation or reconciliation.

4. Sample Output

Output includes:

-Extensive overview of asset deltas, calculated in both native and nominal terms
-Delta Overview Summary via delta_overview(df, eur_usd_t, eur_usd_t_1): This function breaks down the drivers of portfolio equity changes between two time points:
. Total Equity at t / t-1
. Long / Short Exposure
. Lagged Equity
. Δ Equity (Total change)
. Δ Price Effect – change in market price
. Δ Position Effect – change in position size
. Δ FX Effect – change in exchange rate
. Δ Market Effect – combination of price and FX shifts

