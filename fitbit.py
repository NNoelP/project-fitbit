from module import *
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import streamlit as st

# INITIALIZING
conn = connect("./data/fitbit_database.db")
daily_activity = pd.read_sql("SELECT * FROM daily_activity", conn)
daily_activity["Id"] = daily_activity["Id"].astype(str)
daily_activity["ActivityDate"] = pd.to_datetime(daily_activity["ActivityDate"])

# PRINTING UNIQUE USERS


def unique_users(df):
    unique_users = df["Id"].nunique()
    st.write(f"Total unique users: {unique_users}")
