"""predict.py
Load saved model and make predictions on the test set. Save a comparison plot.
""" 
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from data_prep import load_and_preprocess

def main(csv_path='data/BTC-USD.csv', model_path='models/bitcoin_lstm.h5', seq_len=60):
    X_train, y_train, X_test, y_test, scaler = load_and_preprocess(csv_path, seq_len=seq_len)
    model = load_model(model_path)
    preds = model.predict(X_test)
    # inverse scale
    preds_inv = scaler.inverse_transform(preds)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))

    # simple metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    mse = mean_squared_error(y_test_inv, preds_inv)
    mae = mean_absolute_error(y_test_inv, preds_inv)
    print(f'MSE: {mse:.4f}, MAE: {mae:.4f}')

    # plot
    plt.figure(figsize=(10,5))
    plt.plot(y_test_inv, label='True Price')
    plt.plot(preds_inv, label='Predicted Price')
    plt.title('BTC Price — True vs Predicted (test set)')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    os.makedirs('models', exist_ok=True)
    plt.tight_layout()
    plt.savefig('models/prediction_vs_true.png')
    print('Saved models/prediction_vs_true.png')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/BTC-USD.csv')
    parser.add_argument('--model', default='models/bitcoin_lstm.h5')
    parser.add_argument('--seq_len', type=int, default=60)
    args = parser.parse_args()
    main(csv_path=args.data, model_path=args.model, seq_len=args.seq_len)
