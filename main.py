import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

class BitcoinPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.lookback = 60
        
    def load_and_validate_data(self):
        """Load and validate Bitcoin data"""
        print("📊 Loading Bitcoin data...")
        
        # Check if data file exists
        if not os.path.exists('data/bitcoin_data.csv'):
            print("❌ Data file not found. Creating sample data...")
            self.create_sample_data()
        
        try:
            df = pd.read_csv('data/bitcoin_data.csv', index_col='Date', parse_dates=True)
            
            if len(df) == 0:
                raise ValueError("Data file is empty")
                
            print(f"✅ Successfully loaded {len(df)} records")
            print(f"📅 Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
            print(f"💰 Price range: ${df['Price'].min():.2f} - ${df['Price'].max():.2f}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print("🔄 Creating sample data...")
            return self.create_sample_data()
    
    def create_sample_data(self):
        """Create realistic sample Bitcoin data"""
        dates = pd.date_range(start='2020-01-01', end=datetime.now(), freq='D')
        
        # Create realistic Bitcoin price pattern
        np.random.seed(42)
        n_points = len(dates)
        
        # Start around $30,000 with realistic volatility
        price = 30000
        prices = []
        
        for i in range(n_points):
            # Bitcoin-like volatility
            if i % 30 == 0:  # Monthly volatility
                change = np.random.normal(0, 0.08)
            else:
                change = np.random.normal(0, 0.03)
            
            # Long-term upward trend
            trend = 0.0005
            price = price * (1 + change + trend)
            price = max(price, 1000)  # Don't go below $1000
            
            prices.append(price)
        
        df = pd.DataFrame({'Price': prices}, index=dates)
        
        # Save the data
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/bitcoin_data.csv')
        print(f"✅ Created sample data with {len(df)} records")
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess data for LSTM model"""
        print("🔧 Preprocessing data...")
        
        # Extract prices
        prices = df['Price'].values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(prices)
        
        # Create sequences
        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i-self.lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Reshape for LSTM
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Split data (80% train, 20% test)
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        print(f"📊 Training data: {X_train.shape}")
        print(f"📊 Testing data: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, train_size
    
    def build_model(self, input_shape):
        """Build LSTM model"""
        print("🧠 Building LSTM model...")
        
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(25),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        print("✅ Model built successfully")
        return model
    
    def calculate_accuracy(self, actual, predicted):
        """Calculate prediction accuracy"""
        return 100 - np.mean(np.abs((actual - predicted) / actual)) * 100
    
    def plot_results(self, df, train_predict, test_predict, train_size):
        """Create comprehensive visualizations"""
        print("📈 Creating visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Overall predictions
        self._plot_overall_predictions(axes[0, 0], df, train_predict, test_predict, train_size)
        
        # Plot 2: Training predictions
        self._plot_training_predictions(axes[0, 1], df, train_predict, train_size)
        
        # Plot 3: Test predictions
        self._plot_test_predictions(axes[1, 0], df, test_predict, train_size)
        
        # Plot 4: Price distribution
        self._plot_price_distribution(axes[1, 1], df)
        
        plt.tight_layout()
        plt.savefig('bitcoin_prediction_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_overall_predictions(self, ax, df, train_predict, test_predict, train_size):
        """Plot overall actual vs predicted prices"""
        # Create arrays for plotting
        full_predict = np.empty_like(df['Price'])
        full_predict[:] = np.nan
        full_predict[self.lookback:train_size + self.lookback] = train_predict.flatten()
        full_predict[train_size + self.lookback:train_size + self.lookback + len(test_predict)] = test_predict.flatten()
        
        ax.plot(df.index, df['Price'], label='Actual Price', linewidth=1, alpha=0.8)
        ax.plot(df.index, full_predict, label='Predicted Price', linewidth=1, alpha=0.8)
        ax.set_title('Bitcoin Price: Actual vs Predicted')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_training_predictions(self, ax, df, train_predict, train_size):
        """Plot training data predictions"""
        train_dates = df.index[self.lookback:train_size + self.lookback]
        train_actual = df['Price'].values[self.lookback:train_size + self.lookback]
        
        ax.plot(train_dates, train_actual, label='Actual', linewidth=1)
        ax.plot(train_dates, train_predict.flatten(), label='Predicted', linewidth=1)
        ax.set_title('Training Data Predictions')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_test_predictions(self, ax, df, test_predict, train_size):
        """Plot test data predictions"""
        test_dates = df.index[train_size + self.lookback:train_size + self.lookback + len(test_predict)]
        test_actual = df['Price'].values[train_size + self.lookback:train_size + self.lookback + len(test_predict)]
        
        ax.plot(test_dates, test_actual, label='Actual', linewidth=1)
        ax.plot(test_dates, test_predict.flatten(), label='Predicted', linewidth=1)
        ax.set_title('Test Data Predictions')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_price_distribution(self, ax, df):
        """Plot price distribution"""
        sns.histplot(df['Price'], kde=True, ax=ax)
        ax.set_title('Bitcoin Price Distribution')
        ax.set_xlabel('Price (USD)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

def main():
    print("🚀 Bitcoin Price Prediction Project")
    print("=" * 50)
    
    # Initialize predictor
    predictor = BitcoinPredictor()
    
    try:
        # Step 1: Load data
        df = predictor.load_and_validate_data()
        
        # Step 2: Preprocess data
        X_train, X_test, y_train, y_test, train_size = predictor.preprocess_data(df)
        
        # Step 3: Build model
        model = predictor.build_model((X_train.shape[1], 1))
        
        # Step 4: Train model
        print("🏋️ Training model...")
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=50,
            validation_data=(X_test, y_test),
            verbose=1,
            shuffle=False
        )
        
        # Step 5: Make predictions
        print("🔮 Making predictions...")
        
        # Train predictions
        train_predict = model.predict(X_train)
        train_predict = predictor.scaler.inverse_transform(train_predict)
        
        # Test predictions
        test_predict = model.predict(X_test)
        test_predict = predictor.scaler.inverse_transform(test_predict)
        
        # Get actual prices for comparison
        train_actual = predictor.scaler.inverse_transform(y_train.reshape(-1, 1))
        test_actual = predictor.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Step 6: Calculate accuracy
        train_accuracy = predictor.calculate_accuracy(train_actual, train_predict)
        test_accuracy = predictor.calculate_accuracy(test_actual, test_predict)
        
        # Step 7: Create visualizations
        predictor.plot_results(df, train_predict, test_predict, train_size)
        
        # Step 8: Save model
        print("💾 Saving model...")
        os.makedirs('models', exist_ok=True)
        model.save('models/bitcoin_lstm_model.h5')
        
        # Final results
        print("\n🎯 MODEL PERFORMANCE SUMMARY:")
        print(f"   Training Accuracy: {train_accuracy:.2f}%")
        print(f"   Test Accuracy: {test_accuracy:.2f}%")
        print(f"   Final Training Loss: {history.history['loss'][-1]:.6f}")
        print(f"   Final Validation Loss: {history.history['val_loss'][-1]:.6f}")
        print(f"   Lookback Period: {predictor.lookback} days")
        print(f"   Training Samples: {len(X_train)}")
        print(f"   Testing Samples: {len(X_test)}")
        
        print("\n✅ PROJECT COMPLETED SUCCESSFULLY!")
        print("📁 Generated files:")
        print("   - data/bitcoin_data.csv")
        print("   - models/bitcoin_lstm_model.h5")
        print("   - bitcoin_prediction_comprehensive.png")
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()