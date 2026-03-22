import streamlit as st
import pandas as pd
import sqlite3
import os
import matplotlib.pyplot as plt
import seaborn as sns
from fitbit import *

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "fitbit_database.db")
CSV_PATH = os.path.join(BASE_DIR, "data", "daily_activity.csv")


@st.cache_resource
def get_connection():
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)

    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    if not tables:
        st.info("Database is empty!")
        if os.path.exists(CSV_PATH):
            df_import = pd.read_csv(CSV_PATH)
            df_import.to_sql("daily_activity", conn, index=False)
            st.success("Database imported!")
        else:
            st.error(f"CSV file not found at {CSV_PATH}.")

    return conn


@st.cache_data
def load_data():
    conn = get_connection()
    try:
        df = pd.read_sql("SELECT * FROM daily_activity", conn)
        df["Id"] = df["Id"].astype(str)
        df["ActivityDate"] = pd.to_datetime(df["ActivityDate"])
        return df
    except Exception as e:
        return pd.DataFrame()


st.set_page_config(page_title="Fitbit Dashboard", layout="wide")
st.title("Fitbit Activity Dashboard")

data = load_data()
connect = get_connection()

pages = st.sidebar.radio(
    "User",
    [
        "General User Statistics",
        "User Analysis",
        "Activity Analysis",
        "Health Analysis",
    ],
)
if not data.empty:
    # users can choose the user id from the sidebar
    id = st.sidebar.selectbox("User ID", data["Id"].unique())
    start_date = st.sidebar.date_input("Start Date")
    end_date = st.sidebar.date_input("End Date")

    if pages == "General User Statistics":
        st.subheader("Total Distance per User")
        distance_per_user(data)

        st.subheader("General User Classification")
        classify(connect)

    elif pages == "User Analysis":
        st.subheader(f"User {id}")

        st.write("User Overview")
        # user info
        st.dataframe(data[data["Id"] == id].head(10), use_container_width=True)

    elif pages == "Activity Analysis":
        st.write("CALORIES BURNT")
        calories_per_day_user(data, id, start_date, end_date)

        st.write("STEPS AND ACTICITY COMPARISON")
        plot_linear_regression(data, id)

        st.subheader("Workout Patterns")

        workout_frequency(data)

        st.subheader("SLEEP AND ACTIVITY COMPARISON")
        sleep_active_minutes(connect)

    elif pages == "Health Analysis":
        st.subheader("Sleep Analysis")
        sleep_duration(connect)
        bedtime_vs_duration(connect, id)
        sleep_distribution(connect, id)

else:
    print("There is no data, please try again.")
