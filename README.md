# Fitbit Projet

## Overview
With this project, we worked on data visualization and analysis for Fitbit users by using Streamlit dashboard. 

## Project Structure
<pre>   Project-Fitbit/ # Root folder
├── data/ # All the data including daily activity and fitbit db
├── dashboard.py # Streamlit dashboard
├── fitbit.py # Main file for analysis
├── requirements.txt # Dependincies
├── README.md # Project description </pre>

## How to run the dashboard?

To run the dashboard, please follow these steps:
1. Ensure all dependencies are installed using pip install -r requirements.txt.

2. Place the daily_activity.csv in the data/ directory.

3. Run the command streamlit run dashboard.py on the terminal (make sure that you are in the root folder when running this command).

## Project Features
1. Getting acquainted with the data

- Counting unique users
- Calculating total distance for users
- Calculating calories per day for users
- Calculating workout frequency for users
- Linear Regression model between total steps and calories
- Classifying users based on activity

2. Interacting with the database

- Query for sleeping analysis including sleep duration/distribution, active minutes vs sleep minutes.

3. Data Wrangling

- Handling missing weight values.

4. Creating a dashboard
- Visualization of the data that was analysed before.


## Analysis
