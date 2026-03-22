import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tsa.stattools as smt
from statsmodels.tsa.seasonal import STL
import sqlite3
from module import *  # Assuming your custom functions are here

DATE_FMT = "%m/%d/%Y %I:%M:%S %p"

conn = sqlite3.connect("./data/fitbit_database.db")

tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';",conn)['name'].values
ids = []
for table in tables:
    ids = np.union1d(ids, pd.read_sql(f"SELECT DISTINCT Id FROM {table}", conn)['Id'].astype(int).astype(str).values)

# Set page config
st.set_page_config(page_title="Fitbit Data Dashboard", layout="wide")

# Title
st.title("Fitbit Activity, Heart Rate & Sleep Analysis Dashboard")

# Sidebar for global controls
st.sidebar.header("Global Filters")
selected_ids = st.sidebar.multiselect("Select User IDs", options=ids, default=ids[0])

# Get min/max dates across datasets
@st.cache_data
def get_date_range():
    dates = []
    tables_to_check = ['hourly_calories', 'heart_rate', 'minute_sleep']
    for table in tables_to_check:
        try:
            if table == 'heart_rate':
                df = pd.read_sql(f"SELECT MIN(Time) as min_date, MAX(Time) as max_date FROM {table}", conn)
            elif table == 'minute_sleep':
                df = pd.read_sql(f"SELECT MIN(date) as min_date, MAX(date) as max_date FROM {table}", conn)
            else:
                df = pd.read_sql(f"SELECT MIN(ActivityHour) as min_date, MAX(ActivityHour) as max_date FROM {table}", conn)
            dates.append(pd.to_datetime(df['min_date'][0], format=DATE_FMT))
            dates.append(pd.to_datetime(df['max_date'][0], format=DATE_FMT))
        except:
            pass
    min_date = min(dates).date()
    max_date = max(dates).date()
    return min_date, max_date

min_date, max_date = get_date_range()
date_range = st.sidebar.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

# Load data (similar to notebook)
@st.cache_data
def load_data(selected_ids, date_range):
    # Convert date_range to datetime for filtering
    if len(date_range) == 2:
        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
    else:
        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[0]) + pd.Timedelta(days=1)
    
    # Activity data
    cis_df = pd.read_sql("""
                     SELECT *
                     FROM hourly_calories
                     NATURAL JOIN hourly_intensity
                     NATURAL JOIN hourly_steps
                     """, conn)
    cis_df['Id'] = cis_df['Id'].astype(int).astype(str)
    cis_df = cis_df[cis_df['Id'].isin(selected_ids)]
    cis_df['Time'] = pd.to_datetime(cis_df['ActivityHour'], format=DATE_FMT)
    cis_df = cis_df[(cis_df['Time'] >= start_date) & (cis_df['Time'] < end_date)]
    cis_df = cis_df.set_index(['Id', 'Time']).drop(columns=['ActivityHour','AverageIntensity'])
    cis_means = cis_df.groupby('Time').mean().iloc[:-1] if not cis_df.empty else pd.DataFrame()

    # Heart rate
    id_list = "', '".join(selected_ids)
    hr_df = pd.read_sql(f"""
                        SELECT Id, Time, value
                        FROM heart_rate
                        WHERE Id IN ('{id_list}')
                        """, conn)
    if not hr_df.empty:
        hr_df['Id'] = hr_df['Id'].astype(int).astype(str)
        hr_df['Time'] = pd.to_datetime(hr_df['Time'], format=DATE_FMT)
        hr_df = hr_df[(hr_df['Time'] >= start_date) & (hr_df['Time'] < end_date)]
    hr_stats = hr_df.groupby('Time')['value'].describe() if not hr_df.empty else pd.DataFrame()
    
    # Sleep
    minute_sleep = pd.read_sql("SELECT * from minute_sleep", conn)
    minute_sleep[['Id', 'logId']] = minute_sleep[['Id', 'logId']].astype(int).astype(str)
    minute_sleep = minute_sleep[minute_sleep['Id'].isin(selected_ids)]
    minute_sleep['date'] = pd.to_datetime(minute_sleep['date'], format=DATE_FMT)
    minute_sleep = minute_sleep[(minute_sleep['date'] >= start_date) & (minute_sleep['date'] < end_date)]
    minute_sleep['Duration'] = 1
    sleep_df = minute_sleep.groupby(['Id', 'date'])['Duration'].sum() if not minute_sleep.empty else pd.Series()
    sleep_stats = sleep_df.reset_index().groupby('date')['Duration'].describe() if not sleep_df.empty else pd.DataFrame()
    
    # Align data
    if cis_means.empty or hr_stats.empty or sleep_stats.empty:
        aligned_data = pd.DataFrame()
    else:
        aligned_data = cis_means[['TotalIntensity']].join(sleep_stats[['mean']].rename(columns={'mean': 'Sleep'}), how='left')
        aligned_data['Sleep'] = aligned_data['Sleep'].fillna(0)
        if 'mean' in hr_stats.columns:
            aligned_data = aligned_data.join(hr_stats[['mean']].rename(columns={'mean': 'HeartRate'}), how='inner')
    
    return cis_means, hr_stats, sleep_stats, aligned_data

cis_means, hr_stats, sleep_stats, aligned_data = load_data(selected_ids, date_range)

# Tabs for pages
tab1, tab2, tab3, tab4 = st.tabs(["Activity", "Heart Rate", "Sleep", "Relations"])

with tab1:
    st.header("Activity Analysis")
    if cis_means.empty:
        st.warning("No activity data available for selected date range and users.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Correlations")
            fig, ax = plt.subplots()
            sns.heatmap(cis_means.corr(), annot=True, ax=ax)
            st.pyplot(fig)
        with col2:
            st.subheader("Stats Plot")
            fig, ax = plt.subplots()
            cis_means.plot(subplots=True, sharex=True, ax=ax)
            st.pyplot(fig)

with tab2:
    st.header("Heart Rate Analysis")
    if hr_stats.empty:
        st.warning("No heart rate data available for selected date range and users.")
    else:
        st.subheader("Stats Plot")
        fig, ax = plt.subplots()
        hr_stats.loc[:,::-1].plot(subplots=True, sharex=True, ax=ax)
        st.pyplot(fig)

with tab3:
    st.header("Sleep Analysis")
    if sleep_stats.empty:
        st.warning("No sleep data available for selected date range and users.")
    else:
        st.subheader("Stats Plot")
        fig, ax = plt.subplots()
        sleep_stats.loc[:,::-1].plot(subplots=True, sharex=True, ax=ax)
        st.pyplot(fig)

with tab4:
    st.header("Relations Analysis")
    if aligned_data.empty:
        st.warning("No aligned data available for selected date range and users.")
    else:
        lag_range = st.slider("CCF Lag Range", -12, 12, (-12, 12))
        selected_y = st.selectbox("Select Target for CCF", ["Sleep", "HeartRate"])
        lags = range(lag_range[0], lag_range[1] + 1)
        ccf_results = {}
        for col in aligned_data.drop(columns=selected_y).columns:
            ccf = smt.ccf(aligned_data[selected_y], aligned_data[col], adjusted=False)
            ccf_results[col] = ccf[:len(lags)]
        fig, ax = plt.subplots()
        for col, vals in ccf_results.items():
            ax.plot(lags, vals, label=col)
        ax.axhline(0, color='black', linestyle='--')
        ax.set_xlabel('Lag (hours)')
        ax.set_ylabel('Cross-Correlation')
        ax.set_title(f'CCF with {selected_y}')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        st.subheader("OLS Results")
        st.write("**Sleep ~ TotalIntensity (lag -11):**")
        lag = -11
        X = aligned_data[['TotalIntensity']].shift(lag).dropna()
        y = aligned_data['Sleep'][X.index]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        st.text(model.summary())
        st.write("**HeartRate ~ TotalIntensity (lag 0):**")
        X = aligned_data[['TotalIntensity']]
        y = aligned_data['HeartRate']
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        st.text(model.summary())

st.info("Dashboard updated with tabs and filters. Run with `streamlit run streamlit_app.py`")