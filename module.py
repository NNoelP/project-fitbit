import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np
from sqlite3 import connect

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

class MinDateAgg:
    def __init__(self):
        self.values = []

    def step(self, v):
        if v is not None:
            self.values.append(v)

    def finalize(self):
        date = pd.to_datetime(self.values, format=DATE_FMT).sort_values()[0]
        return str(date)

class MaxDateAgg:
    def __init__(self):
        self.values = []

    def step(self, v):
        if v is not None:
            self.values.append(v)

    def finalize(self):
        date = pd.to_datetime(self.values, format=DATE_FMT).sort_values()[-1]
        return str(date)

conn = connect("./data/fitbit_database.db")
conn.create_aggregate("percentile", 2, PercentileAgg)
conn.create_aggregate("stdev", 1, StdevAgg)
conn.create_aggregate("min_date", 1, MinDateAgg)
conn.create_aggregate("max_date", 1, MaxDateAgg)
conn.create_function("to_date", 1, lambda x: str(pd.to_datetime(x, format=DATE_FMT)))

def table_info(table):
    nrows = pd.read_sql(f"SELECT COUNT(*) FROM {table}", conn).values[0,0]
    print(f"nrows: {nrows}")
    sample = pd.read_sql(f"SELECT * FROM {table} LIMIT 5", conn)
    sample.info()
    return sample