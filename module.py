import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL, MSTL
import numpy as np
from sqlite3 import connect
import statsmodels.tsa.stattools as smt

DATE_FMT = "%m/%d/%Y %I:%M:%S %p"

class PercentileAgg:
    def __init__(self):
        self.values = []
    
    def step(self, value, p):
        if value is not None:
            self.values.append(value)
            self.p = p
    
    def finalize(self):
        if not self.values:
            return None
        return np.percentile(self.values, self.p * 100)
    
class StdevAgg:
    def __init__(self):
        self.values = []

    def step(self, v):
        if v is not None:
            self.values.append(v)

    def finalize(self):
        if len(self.values) < 2:
            return 0.0
        return np.std(self.values, ddof=1)  # sample stdev

conn = connect("./data/fitbit_database.db")
conn.create_aggregate("percentile", 2, PercentileAgg)
conn.create_aggregate("stdev", 1, StdevAgg)

def table_info(table):
    nrows = pd.read_sql(f"SELECT COUNT(*) FROM {table}", conn).values[0,0]
    print(f"nrows: {nrows}")
    sample = pd.read_sql(f"SELECT * FROM {table} LIMIT 5", conn)
    sample.info()
    return sample

def plot_linear_regression(X, y, title="Title"):
    model = sm.OLS(y, X)
    results = model.fit()
    beta = results.params.values
    y_pred = X.dot(beta) if X.ndim >= 2 else beta * X
    
    sns.lineplot(y, label='Data')
    sns.lineplot(y_pred, label='Regression', color="darkorange")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

    print(results.summary())
    
    errors = y - y_pred
    sns.histplot(errors)
    plt.title("Regression Errors")
    plt.show()

def plot_ccf(df, y, lags):
    # Compute cross-correlations for each variable with sleep
    ccf_results = {}
    for col in df.drop(columns=y):
        ccf = smt.ccf(df[y], df[col], adjusted=False)
        ccf_results[col] = ccf[:len(lags)]  # Take relevant lags

    # Plot
    plt.figure(figsize=(10, 6))
    for col, ccf_vals in ccf_results.items():
        plt.plot(lags, ccf_vals, label=col)
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Lag (hours)')
    plt.ylabel('Cross-Correlation')
    plt.title(f'Cross-Correlation with {y}')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Identify significant peaks (focus on lags -12 to 12 to avoid edge artifacts)
    significant_lags = [lag for lag in lags if abs(lag) <= 12]
    for col in ccf_results:
        max_corr = max(ccf_results[col][i] for i, lag in enumerate(lags) if lag in significant_lags)
        max_lag = lags[ccf_results[col].tolist().index(max_corr)]
        print(f"{col}: Peak correlation {max_corr:.3f} at lag {max_lag}")

def plot_stl(series, period=24, title="STL"):
    stl = STL(series.rename(title), period=period)
    res = stl.fit()
    fig = res.plot()
    plt.xticks(rotation=45)
    plt.show()