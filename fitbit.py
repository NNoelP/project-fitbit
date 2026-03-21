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


# LINEAR REGRESSION MODEL


def linear_regression(df, id):
    user_data = df[df["Id"] == str(id)]
    model = smf.ols("Calories ~ TotalSteps", data=user_data).fit()
    # st.write(model.summary())


def plot_linear_regression(df, id):
    plt.figure(figsize=(1, 6))
    sns.regplot(
        data=daily_activity[daily_activity["Id"] == str(id)],
        x="TotalSteps",
        y="Calories",
        line_kws={"color": "grey"},
    )
    plt.title(f"Linear Regression: Calories vs. Steps (User {id})")
    plt.xlabel("Total Steps")
    plt.ylabel("Calories")
    plt.show()
    st.pyplot(plt.gcf())
    plt.close()

    linear_regression(df, id)


# CLASSIFYING USERS


def class_of_user(count):
    if count <= 10:
        return "Light user"
    elif count <= 15:
        return "Moderate user"
    return "Heavy user"


def classify(connect):
    query = """SELECT Id, COUNT(*) as activity_count FROM daily_activity GROUP BY Id"""
    df = pd.read_sql(query, connect)

    df["Class"] = df["activity_count"].apply(class_of_user)
    class_counts = df["Class"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        class_counts,
        labels=class_counts.index,
        autopct="%1.1f%%",
        colors=sns.color_palette("pastel"),
    )
    ax.set_title("User Base Classification")
    st.pyplot(plt.gcf())
    plt.close()


############## PART 3

# SLEEP DURATION OF USERS


def sleep_duration(connect):
    sleep_query = "SELECT Id, SUM(value) as total_sleep FROM minute_sleep GROUP BY Id"

    df = pd.read_sql(sleep_query, connect)
    df["Id"] = df["Id"].astype(str)

    plt.figure(figsize=(12, 6))
    sns.histplot(df["total_sleep"], kde=True)
    plt.xlabel("Sleeping Duration")
    plt.ylabel("Frequency")
    plt.title("Distribution for Sleep Duration")
    st.pyplot(plt.gcf())
    plt.close()

    return df


# SLEEP DISTRIBUTION


def sleep_distribution(conn, id):
    query = f"SELECT * FROM minute_sleep WHERE Id = '{id}'"
    minute_sleep = pd.read_sql(query, conn)

    if not minute_sleep.empty:
        dates = pd.to_datetime(minute_sleep["date"])

        T0 = 18
        SHIFT = 24 - T0
        time = (dates.dt.hour + SHIFT) % 24 + dates.dt.minute / 60

        fig, ax1 = plt.subplots(figsize=(10, 5))
        sns.histplot(time, binwidth=1, color="darkorange", ax=ax1)
        ax2 = ax1.twinx()
        sns.kdeplot(time, ax=ax2)

        ax1.set_xticks(range(24))
        ax1.set_xticklabels(
            [f"{h:02d}:00" for h in np.roll(range(24), SHIFT)], rotation=45
        )
        ax1.set_xlabel("Time of Day")
        plt.title(f"Sleep Distribution (User {id})")
        st.pyplot(plt.gcf())
        plt.close()


# COMPARING SLEEP AND ACTIVE MINUTES


def sleep_active_minutes(conn):
    merged_query = """
        SELECT a.Id, SUM(s.value) as sleep, 
        SUM(a.VeryActiveMinutes + a.FairlyActiveMinutes + a.LightlyActiveMinutes) as activity
    FROM minute_sleep s
    JOIN daily_activity a ON s.Id = a.Id
    GROUP BY s.Id
    """

    df = pd.read_sql(merged_query, conn).dropna()

    if not df.empty:
        # we can also print the models but it is taking too much space and also not very visual for the dashboard therfore i removed it
        # model = smf.ols("activity ~ sleep", data=df).fit()
        # st.text(model.summary())

        sns.regplot(x=df["sleep"], y=df["activity"])
        plt.ylabel("Total Active Time")
        plt.xlabel("Total Sleeping Time")
        plt.title("Comparing Sleep Duration and Active Minutes")
        st.pyplot(plt.gcf())


# BEDTIME VS DURATION


def bedtime_vs_duration(conn, id):
    query = f"SELECT *, COUNT(*) AS minutes FROM minute_sleep WHERE Id = '{id}' GROUP BY logId"

    bedtimes = pd.read_sql(query, conn)
    duration = bedtimes["minutes"] / 60

    if bedtimes.empty:
        st.write("No data.")
        return

    bedtimes["date"] = pd.to_datetime(bedtimes["date"])

    T0 = 18
    SHIFT = 24 - T0
    shifted_time = (
        (bedtimes["date"].dt.hour + SHIFT) % 24
        + bedtimes["date"].dt.minute / 60
        + bedtimes["date"].dt.second / 60**2
    )

    g = sns.jointplot(x=shifted_time, y=duration, s=5)
    g.plot_joint(sns.kdeplot, alpha=0.7, label="Sleep Duration")
    g.ax_joint.set(xlabel="Bedtime", ylabel="Duration")

    ticks = range(24)
    labels = [f"{h:02d}:00" for h in np.roll(range(24), SHIFT)]
    g.ax_joint.set_xticks(ticks)
    g.ax_joint.set_xticklabels(labels, rotation=45)

    plt.suptitle("Bedtime vs. Duration", y=1.02)
    st.pyplot(plt.gcf())
    plt.close()


# MISSING VALUES


def fill_weight(df):
    # assigning empty strings with nan so that it is easier to detect
    df.replace(["", " "], pd.NA, inplace=True)

    # this is used if theres any value that was logged before
    df["WeightKg"] = df.groupby("Id")["WeightKg"].transform(
        lambda x: x.fillna(x.mean())
    )

    # and take the median
    df["WeightKg"] = df["WeightKg"].fillna(df["WeightKg"].median())

    return df
