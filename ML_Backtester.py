import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

data = "btc_weekly_data.csv"
#data = "btc_daily_data.csv"
#data = "btc_monthly_data.csv"

df = pd.read_csv(data)

# Assuming your DataFrame is named df
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') # Convert timestamp to datetime
df.set_index('timestamp', inplace=True) # Set the timestamp as the index

# Calculate the Simple Moving Average (SMA)
df['SMA_30'] = ta.trend.sma_indicator(df['close'], window=30)
df['SMA_100'] = ta.trend.sma_indicator(df['close'], window=100)

# Calculate the Exponential Moving Average (EMA)
df['EMA_30'] = ta.trend.ema_indicator(df['close'], window=30)
df['EMA_100'] = ta.trend.ema_indicator(df['close'], window=100)

# Calculate the Relative Strength Index (RSI)
df['RSI'] = ta.momentum.rsi(df['close'], window=14)

# Calculate the Moving Average Convergence Divergence (MACD)
df['MACD'] = ta.trend.macd_diff(df['close'])

# Bollinger Bands
bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
df['bollinger_hband'] = bollinger.bollinger_hband()
df['bollinger_lband'] = bollinger.bollinger_lband()


# On-Balance Volume (OBV)
df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])

df.drop(['open', 'high', 'low', 'date'], axis=1, inplace=True)

# Select columns to scale
if data == "btc_monthly_data.csv":
    cols_to_scale = ['volume', 'SMA_30',  'EMA_30',
                      'RSI', 'MACD', 'bollinger_hband', 'bollinger_lband',
                      'OBV']
else:
    cols_to_scale = ['volume', 'SMA_30', 'SMA_100', 'EMA_30',
                     'EMA_100', 'RSI', 'MACD', 'bollinger_hband', 'bollinger_lband',
                      'OBV']


df = df.dropna(axis=1, how='all')
df.dropna(inplace=True)

# Apply the scaler to the columns
df[cols_to_scale] = StandardScaler().fit_transform(df[cols_to_scale])
df['pc_change'] = df['close'].pct_change()

# Create quantile thresholds
upper_threshold = df['pc_change'].quantile(0.5000001)
lower_threshold = df['pc_change'].quantile(0.50)

# Create new column based on conditions
df['buy/sell'] = np.where(df['pc_change'] > upper_threshold, 1,(np.where(df['pc_change'] <= lower_threshold, -1,0)))
df['buy/sell'] = df['buy/sell'].shift(-1)
df.head()

# Remove NaN values
df = df.dropna(axis=1, how='all')
df = df.dropna()
df = df.drop(['close', 'pc_change'], axis=1)
# Set up features (X) and target (y)
X = df.drop('buy/sell', axis=1)
y = df['buy/sell']
# Map the labels
mapping = {-1: 0, 1: 1}

df.sort_index(inplace=True)

train_size = int(len(df) * 0.8)

# Split the dataset
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

y_test = y_test.map(mapping)
y_train = y_train.map(mapping)

# Set up XGBoost classifier

param_grid = [
    {'n_estimators': [10, 25, 50, 100], 'max_features': ['auto', 'sqrt', 'log2'],
     'max_depth' : [4,5,6,7,8], 'criterion' :['gini', 'entropy']}
]

# Perform the search
model = GridSearchCV(RandomForestClassifier(), param_grid, cv=5,scoring='accuracy', verbose = 2, n_jobs=-1)
#model = GridSearchCV(xgb.XGBClassifier(), param_grid, cv=5,scoring='accuracy', verbose = 2, n_jobs=-1)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

inverse_mapping = {0: -1, 1: 1}
y_pred = pd.Series(y_pred).map(inverse_mapping)
y_test = y_test.map(inverse_mapping)

importances = model.best_estimator_.feature_importances_

# Convert the importances into one-dimensional 1 darray with corresponding df column names as axis labels
f_importances = pd.Series(importances, df.columns[:-1])

# Sort the array in descending order of the importances
f_importances.sort_values(ascending=False, inplace=True)

print(f_importances)

# Evaluate the model
print(classification_report(y_test, y_pred))

