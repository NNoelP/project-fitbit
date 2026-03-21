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


# CALORIES PER DAY FOR USERS


def calories_per_day_user(df, id, start, end):
    user = df[df["Id"] == str(id)].copy()

    user["ActivityDate"] = pd.to_datetime(user["ActivityDate"])
    # making sure that the data exists
    if start and end:
        user = user[user["ActivityDate"] >= pd.to_datetime(start)]
        user = user[user["ActivityDate"] <= pd.to_datetime(end)]

    user = user.sort_values("ActivityDate")

    plt.figure(figsize=(10, 4))
    plt.plot(user["ActivityDate"], user["Calories"])
    plt.title(f"Calories per day {id}")
    plt.xlabel("Date")
    plt.ylabel(" Burnt Calories")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    st.pyplot(plt.gcf())
    plt.close()


# WORKOUT FREQUENCY FOR EACH DAY OF THE WEEK


def workout_frequency(df):
    df["ActivityDate"] = pd.to_datetime(df["ActivityDate"])
    daily_activity = df.groupby("ActivityDate")["Id"].nunique().reset_index()
    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    daily_activity["DayOfWeek"] = daily_activity["ActivityDate"].dt.day_name()

    plt.figure(figsize=(12, 6))
    sns.countplot(daily_activity, x="DayOfWeek", order=days)
    plt.xlabel("Day of Week")
    plt.ylabel("Activity")
    plt.title("Workout Frequency by Day of Week Users")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    st.pyplot(plt.gcf())
    plt.close()
