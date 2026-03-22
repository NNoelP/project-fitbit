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
    # Format dates for SQL query
    start_str = start_date.strftime("%m/%d/%Y 12:00:00 AM")
    end_str = end_date.strftime("%m/%d/%Y 12:00:00 AM")
    hr_df = pd.read_sql(f"""
                        SELECT Id, Time, Value
                        FROM heart_rate
                        WHERE Id IN ('{id_list}')
                          AND Time >= '{start_str}'
                          AND Time < '{end_str}'
                        """, conn)
    if not hr_df.empty:
        hr_df['Id'] = hr_df['Id'].astype(int).astype(str)
        hr_df['Time'] = pd.to_datetime(hr_df['Time'], format=DATE_FMT)
    hr_stats = hr_df.groupby('Time')['Value'].describe() if (not hr_df.empty and 'Value' in hr_df.columns) else pd.DataFrame()
    
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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Statistics", "Activity", "Heart Rate", "Sleep", "Relations"])

with tab1:
    st.header("Data Statistics")
    
    # Load raw data for statistics
    @st.cache_data
    def get_data_stats(selected_ids):
        stats = {}
        
        # Activity data
        cis_df = pd.read_sql("""
                         SELECT *
                         FROM hourly_calories
                         NATURAL JOIN hourly_intensity
                         NATURAL JOIN hourly_steps
                         """, conn)
        cis_df['Id'] = cis_df['Id'].astype(int).astype(str)
        cis_df = cis_df[cis_df['Id'].isin(selected_ids)]
        stats['Activity Records'] = len(cis_df)
        stats['Avg Calories'] = cis_df['Calories'].mean() if not cis_df.empty else 0
        stats['Avg Intensity'] = cis_df['TotalIntensity'].mean() if not cis_df.empty else 0
        stats['Avg Steps'] = cis_df['StepTotal'].mean() if not cis_df.empty else 0
        
        # Heart rate data
        id_list = "', '".join(selected_ids)
        hr_df = pd.read_sql(f"""
                            SELECT *
                            FROM heart_rate
                            WHERE Id IN ('{id_list}')
                            """, conn)
        hr_df['Id'] = hr_df['Id'].astype(int).astype(str) if not hr_df.empty else pd.Series()
        stats['Heart Rate Records'] = len(hr_df)
        stats['Avg Heart Rate'] = hr_df['Value'].mean() if not hr_df.empty else 0
        stats['Min Heart Rate'] = hr_df['Value'].min() if not hr_df.empty else 0
        stats['Max Heart Rate'] = hr_df['Value'].max() if not hr_df.empty else 0
        
        # Sleep data
        sleep_df = pd.read_sql("SELECT * from minute_sleep", conn)
        sleep_df['Id'] = sleep_df['Id'].astype(int).astype(str)
        sleep_df = sleep_df[sleep_df['Id'].isin(selected_ids)]
        stats['Sleep Records'] = len(sleep_df)
        stats['Avg Sleep Duration'] = (len(sleep_df) / len(sleep_df['date'].unique())) if not sleep_df.empty and len(sleep_df['date'].unique()) > 0 else 0
        
        return stats, cis_df, hr_df, sleep_df
    
    stats, cis_df, hr_df, sleep_df = get_data_stats(selected_ids)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Availability Heatmap")
        # Create availability matrix by user and metric type
        if not cis_df.empty or not hr_df.empty or not sleep_df.empty:
            availability_data = []
            for user_id in selected_ids:
                row = {
                    'User ID': user_id,
                    'Activity': len(cis_df[cis_df['Id'] == user_id]) if not cis_df.empty else 0,
                    'Heart Rate': len(hr_df[hr_df['Id'] == user_id]) if not hr_df.empty else 0,
                    'Sleep': len(sleep_df[sleep_df['Id'] == user_id]) if not sleep_df.empty else 0
                }
                availability_data.append(row)
            
            avail_df = pd.DataFrame(availability_data).set_index('User ID')
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(avail_df, annot=True, fmt='d', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Record Count'})
            ax.set_title('Data Availability by User and Metric')
            st.pyplot(fig)
        else:
            st.warning("No data available for selected users.")
    
    with col2:
        st.subheader("Summary Statistics")
        summary_stats = pd.DataFrame([
            {'Metric': 'Total Users', 'Value': len(selected_ids)},
            {'Metric': 'Activity Records', 'Value': int(stats['Activity Records'])},
            {'Metric': 'Heart Rate Records', 'Value': int(stats['Heart Rate Records'])},
            {'Metric': 'Sleep Records', 'Value': int(stats['Sleep Records'])},
            {'Metric': 'Avg Daily Calories', 'Value': f"{stats['Avg Calories']:.1f}"},
            {'Metric': 'Avg Daily Intensity', 'Value': f"{stats['Avg Intensity']:.1f}"},
            {'Metric': 'Avg Daily Steps', 'Value': f"{stats['Avg Steps']:.0f}"},
            {'Metric': 'Avg Heart Rate (bpm)', 'Value': f"{stats['Avg Heart Rate']:.1f}"},
            {'Metric': 'Heart Rate Range', 'Value': f"{int(stats['Min Heart Rate'])}-{int(stats['Max Heart Rate'])} bpm"},
            {'Metric': 'Avg Sleep (min/day)', 'Value': f"{stats['Avg Sleep Duration']:.0f}"},
        ])
        st.dataframe(summary_stats, use_container_width=True, hide_index=True)

with tab2:
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

with tab3:
    st.header("Heart Rate Analysis")
    if hr_stats.empty:
        st.warning("No heart rate data available for selected date range and users.")
    else:
        st.subheader("Stats Plot")
        fig, ax = plt.subplots()
        hr_stats.loc[:,::-1].plot(subplots=True, sharex=True, ax=ax)
        st.pyplot(fig)

with tab4:
    st.header("Sleep Analysis")
    if sleep_stats.empty:
        st.warning("No sleep data available for selected date range and users.")
    else:
        st.subheader("Bedtime vs. Duration")

        # Load raw minute_sleep data for jointplot
        minute_sleep_raw = pd.read_sql("SELECT * from minute_sleep", conn)
        minute_sleep_raw['Id'] = minute_sleep_raw['Id'].astype(int).astype(str)
        minute_sleep_raw = minute_sleep_raw[minute_sleep_raw['Id'].isin(selected_ids)]
        minute_sleep_raw['date'] = pd.to_datetime(minute_sleep_raw['date'], format=DATE_FMT)
        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
        minute_sleep_raw = minute_sleep_raw[(minute_sleep_raw['date'] >= start_date) & (minute_sleep_raw['date'] < end_date)]
        
        if not minute_sleep_raw.empty:
            # Get bedtime and duration per sleep session (logId)
            sleep_sessions = minute_sleep_raw.groupby('logId').agg({
                'date': 'min',
                'logId': 'count'
            }).reset_index(drop=True)
            sleep_sessions.columns = ['Bedtime', 'Duration_minutes']
            sleep_sessions['Bedtime'] = pd.to_datetime(sleep_sessions['Bedtime'])
            # Convert duration from minutes to hours
            sleep_sessions['Duration'] = sleep_sessions['Duration_minutes'] / 60
            # Extract hour for x-axis
            sleep_sessions['Bedtime_Hour'] = sleep_sessions['Bedtime'].dt.hour + sleep_sessions['Bedtime'].dt.minute / 60
            
            # Create KDE jointplot
            fig = sns.jointplot(data=sleep_sessions, x='Bedtime_Hour', y='Duration', kind='kde', height=6, fill=True)
            fig.set_axis_labels('Bedtime (Hour of Day)', 'Duration (hours)')
            st.pyplot(fig)
        
        st.subheader("Stats Plot")
        fig, ax = plt.subplots()
        sleep_stats.loc[:,::-1].plot(subplots=True, sharex=True, ax=ax)
        st.pyplot(fig)

with tab5:
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
