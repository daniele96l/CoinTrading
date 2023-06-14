import pandas as pd
from io import StringIO

data = "btc_hourly_data.csv"

df = pd.read_csv(data)

print(df)