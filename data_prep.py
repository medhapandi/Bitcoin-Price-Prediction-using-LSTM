"""data_prep.py
Utilities to load BTC-USD CSV, create scaled sequences for LSTM.
Expected CSV: a 'Date' column and 'Close' (closing price) column.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess(csv_path, feature='Close', seq_len=60, split_ratio=0.9):
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    prices = df[[feature]].values.astype('float32')

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(prices)

    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(seq_len, len(data)):
            X.append(data[i-seq_len:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled, seq_len)
    # reshape to [samples, time_steps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(len(X) * split_ratio)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, y_train, X_test, y_test, scaler

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/BTC-USD.csv')
    parser.add_argument('--seq_len', type=int, default=60)
    args = parser.parse_args()
    X_train, y_train, X_test, y_test, scaler = load_and_preprocess(args.data, seq_len=args.seq_len)
    print('Shapes:')
    print(' X_train', X_train.shape)
    print(' y_train', y_train.shape)
    print(' X_test', X_test.shape)
    print(' y_test', y_test.shape)
