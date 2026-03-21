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


############## PART 1

# TOTAL DISTANCE PER USER


def distance_per_user(df):
    total_distance_per_user = (
        df.groupby("Id")["TotalDistance"].sum().sort_values(ascending=False)
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=total_distance_per_user.index, y=total_distance_per_user.values
    )  # using a barplot
    plt.xticks(rotation=90)  # without adding this ID's were not readable
    plt.title("Total Distance Registered for Each User")
    plt.xlabel("User ID")
    plt.ylabel("Total Distance")
    plt.tight_layout()
    plt.show()
    st.pyplot(plt.gcf())
    plt.close()
