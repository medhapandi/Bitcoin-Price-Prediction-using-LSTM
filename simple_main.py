import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

print("🚀 Bitcoin Price Prediction - Simplified Version")
print("=" * 50)

# Step 1: Load and verify data
print("📊 Loading data...")
try:
    df = pd.read_csv('data/bitcoin_data.csv', index_col='Date', parse_dates=True)
    print(f"✅ Loaded {len(df)} records")
    print(f"💰 Price range: ${df['Price'].min():.2f} - ${df['Price'].max():.2f}")
except Exception as e:
    print(f"❌ Failed to load data: {e}")
    exit()

# Step 2: Preprocess data
print("🔧 Preprocessing data...")

# Extract prices
prices = df['Price'].values.reshape(-1, 1)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(prices)

# Create sequences
lookback = 60
X, y = [], []

for i in range(lookback, len(scaled_data)):
    X.append(scaled_data[i-lookback:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

# Reshape for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split data
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"📊 Training data: {X_train.shape}")
print(f"📊 Testing data: {X_test.shape}")

# Step 3: Build LSTM model
print("🧠 Building LSTM model...")

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(50, return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(25),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
print("✅ Model built successfully")

# Step 4: Train model
print("🏋️ Training model...")
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=20,
    validation_data=(X_test, y_test),
    verbose=1
)

# Step 5: Make predictions
print("🔮 Making predictions...")

# Train predictions
train_predict = model.predict(X_train)
train_predict = scaler.inverse_transform(train_predict)

# Test predictions  
test_predict = model.predict(X_test)
test_predict = scaler.inverse_transform(test_predict)

# Actual prices for comparison
train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Step 6: Calculate accuracy
def calculate_accuracy(actual, predicted):
    return 100 - np.mean(np.abs((actual - predicted) / actual)) * 100

train_accuracy = calculate_accuracy(train_actual, train_predict)
test_accuracy = calculate_accuracy(test_actual, test_predict)

print(f"🎯 Training Accuracy: {train_accuracy:.2f}%")
print(f"🎯 Test Accuracy: {test_accuracy:.2f}%")

# Step 7: Create visualization
print("📈 Creating visualization...")

plt.figure(figsize=(15, 8))

# Create arrays for plotting
train_plot = np.empty_like(prices)
train_plot[:, :] = np.nan
train_plot[lookback:lookback + len(train_predict)] = train_predict

test_plot = np.empty_like(prices)  
test_plot[:, :] = np.nan
test_plot[lookback + len(train_predict):lookback + len(train_predict) + len(test_predict)] = test_predict

# Plot
plt.plot(df.index, prices, label='Actual Price', alpha=0.7)
plt.plot(df.index, train_plot, label='Training Predictions', alpha=0.7)
plt.plot(df.index, test_plot, label='Test Predictions', alpha=0.7)

plt.title('Bitcoin Price Prediction using LSTM')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('bitcoin_prediction_simple.png', dpi=300, bbox_inches='tight')
plt.show()

# Step 8: Save model
print("💾 Saving model...")
os.makedirs('models', exist_ok=True)
model.save('models/simple_model.h5')

print("\n✅ PROJECT COMPLETED SUCCESSFULLY!")
print(f"📊 Final Model Accuracy: {test_accuracy:.2f}%")
print("📁 Files created:")
print("   - data/bitcoin_data.csv")
print("   - models/simple_model.h5") 
print("   - bitcoin_prediction_simple.png")