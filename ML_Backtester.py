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

def load_and_merge_data(data, sentiment):
    df = pd.read_csv(data)
    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'].dt.date  # convert to just date
    df = df.merge(sentiment, on='date', how='left')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


def calculate_indicators(df):
    df['pc_change'] = df['close'].pct_change()
    df['SMA_30'] = ta.trend.sma_indicator(df['close'], window=30)
    df['SMA_100'] = ta.trend.sma_indicator(df['close'], window=100)
    df['EMA_30'] = ta.trend.ema_indicator(df['close'], window=30)
    df['EMA_100'] = ta.trend.ema_indicator(df['close'], window=100)
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    df['MACD'] = ta.trend.macd_diff(df['close'])

    bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bollinger_hband'] = bollinger.bollinger_hband()
    df['bollinger_lband'] = bollinger.bollinger_lband()
    df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])

    return df


def preprocess_data(df, data):
    df.drop(['open', 'high', 'low', 'date'], axis=1, inplace=True)

    cols_to_scale = ['volume', 'SMA_30', 'SMA_100', 'EMA_30', 'EMA_100', 'RSI',
                     'MACD', 'bollinger_hband', 'bollinger_lband', 'OBV']

    if data == "btc_monthly_data.csv":
        cols_to_scale.remove('SMA_100')
        cols_to_scale.remove('EMA_100')

    df[cols_to_scale] = StandardScaler().fit_transform(df[cols_to_scale])

    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(inplace=True)

    upper_threshold = df['pc_change'].quantile(0.5000001)
    lower_threshold = df['pc_change'].quantile(0.50)

    df['buy/sell'] = np.where(df['pc_change'] > upper_threshold, 1,
                              (np.where(df['pc_change'] <= lower_threshold, -1, 0)))
    df['buy/sell'] = df['buy/sell'].shift(-1)
    pc_change = df['pc_change']

    df = df.drop(['pc_change', 'timestamp','volume','close','listing_close'], axis=1)

    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(inplace=True)
    return df


def split_data(df):
    X = df.drop('buy/sell', axis=1)
    y = df['buy/sell']

    mapping = {-1: 0, 1: 1}
    df.sort_index(inplace=True)

    train_size = int(len(df) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    y_test = y_test.map(mapping)
    y_train = y_train.map(mapping)

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    param_grid = [
        {'n_estimators': [10, 25, 50, 100], 'max_features': ['auto', 'sqrt', 'log2'],
         'max_depth': [4, 5, 6, 7, 8], 'criterion': ['gini', 'entropy']}
    ]
    model = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    inverse_mapping = {0: -1, 1: 1}
    y_pred = pd.Series(y_pred).map(inverse_mapping)
    y_test = y_test.map(inverse_mapping)

    importances = model.best_estimator_.feature_importances_
    f_importances = pd.Series(importances, df.columns[:-1])
    f_importances.sort_values(ascending=False, inplace=True)

    print(f_importances)
    print(classification_report(y_test, y_pred))


tech = 'W'
sen = 'D'
sentiment = 'sentiment_btc.csv'

df_sen = pd.read_csv(sentiment)

if tech == 'W':
    data = "btc_weekly_data.csv"
if tech == 'D':
    data = "btc_daily_data.csv"

df_sen['date'] = pd.to_datetime(df_sen['date'])

if sen == 'D':
    df_sen['date'] = df_sen['date'].dt.date
    sentiment = df_sen.groupby('date').sum().reset_index()
if sen == 'W':
    df_sen['date'] = pd.to_datetime(df_sen['date'])  # Ensure the date is in datetime format
    df_sen.set_index('date', inplace=True)  # Set 'date' as the DataFrame index
    df_sen = df_sen.resample('W').sum().reset_index()  # Resample to weekly frequency, summing the values
    df_sen['date'] = df_sen['date'].dt.date

df = load_and_merge_data(data, sentiment)
df = calculate_indicators(df)
df = preprocess_data(df, data)
X_train, X_test, y_train, y_test = split_data(df)
model = train_model(X_train, y_train)
evaluate_model(model, X_test, y_test)
