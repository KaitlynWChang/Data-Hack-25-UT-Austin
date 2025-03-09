# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Step 1: Load and preprocess the data
def load_and_preprocess_data(file_path):
    """
    Reads the dataset, normalizes it, and splits it into training and testing sets.
    """
    # Load the dataset into a Pandas DataFrame
    data = pd.read_csv(file_path)
    
    # Filter only GANopolis data (if applicable)
    data = data[data['city'] == 'GANopolis']
    
    # Select the 'wind speed' column as the target variable
    wind_speed = data['wind speed'].values.reshape(-1, 1)
    
    # Normalize the data to the range [0, 1] for better performance with LSTMs
    scaler = MinMaxScaler(feature_range=(0, 1))
    wind_speed_scaled = scaler.fit_transform(wind_speed)
    
    # Split into training (70%) and testing (30%) sets
    train_size = int(len(wind_speed_scaled) * 0.7)
    train_data = wind_speed_scaled[:train_size]
    test_data = wind_speed_scaled[train_size:]
    
    return train_data, test_data, scaler

# Step 2: Create sequences for LSTM input
def create_sequences(data, look_back=5):
    """
    Converts a time series into sequences of input-output pairs for LSTM training.
    
    Parameters:
        - data: The normalized time series data.
        - look_back: Number of previous time steps to use as input features.
        
    Returns:
        - X: Input features (shape: [samples, look_back, 1]).
        - y: Target values (shape: [samples, 1]).
    """
    X, y = [], []
    
    for i in range(len(data) - look_back):
        # Extract the sequence of 'look_back' steps as input
        X.append(data[i:i + look_back])
        # The target is the next step after the sequence
        y.append(data[i + look_back])
    
    return np.array(X), np.array(y)

# Step 3: Build and train the LSTM model
def build_and_train_lstm(train_X, train_y, look_back):
    """
    Builds an LSTM model and trains it on the provided training data.
    
    Parameters:
        - train_X: Input features for training.
        - train_y: Target values for training.
        - look_back: Number of time steps in each input sequence.
        
    Returns:
        - model: The trained LSTM model.
    """
    # Initialize a sequential model
    model = Sequential()
    
    # Add an LSTM layer with 50 units (neurons) and input shape based on `look_back`
    model.add(LSTM(50, activation='relu', input_shape=(look_back, 1)))
    
    # Add a Dense layer to output a single value (regression task)
    model.add(Dense(1))
    
    # Compile the model with Mean Squared Error loss and Adam optimizer
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model for 50 epochs with batch size of 32
    model.fit(train_X, train_y, epochs=50, batch_size=32, verbose=2)
    
    return model

# Step 4: Evaluate the model and make predictions
def evaluate_and_predict(model, test_X, test_y, scaler):
    """
    Evaluates the model on test data and makes predictions.
    
    Parameters:
        - model: The trained LSTM model.
        - test_X: Input features for testing.
        - test_y: Target values for testing.
        - scaler: Scaler used to normalize the data (for inverse transformation).
        
    Returns:
        - predictions: Predicted values (in original scale).
        - true_values: Actual values (in original scale).
        - rmse: Root Mean Squared Error of predictions.
    """
    # Make predictions using the LSTM model
    predictions = model.predict(test_X)
    
    # Invert scaling to get predictions in original scale
    predictions = scaler.inverse_transform(predictions)
    
    # Invert scaling for true values as well
    true_values = scaler.inverse_transform(test_y)
    
    # Calculate RMSE between predictions and true values
    rmse = np.sqrt(mean_squared_error(true_values, predictions))
    
    print(f"Test RMSE: {rmse}")
    
    return predictions, true_values, rmse

# Step 5: Visualize results
def plot_predictions(true_values, predictions):
    """
    Plots the true values vs. predicted values for visualization.
    
    Parameters:
        - true_values: Actual target values from the test set.
        - predictions: Predicted values from the LSTM model.
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(true_values, label='True Values', color='blue')
    plt.plot(predictions, label='Predictions', color='red')
    
    plt.title('True vs Predicted Wind Speeds')
    plt.xlabel('Time Steps')
    plt.ylabel('Wind Speed')
    
    plt.legend()
    
# Main script execution
if __name__ == "__main__":
    


# Main script execution
if __name__ == "__main__":
    # Step 1: Load and preprocess the data
    print("Loading and preprocessing data...")
    train_data, test_data, scaler = load_and_preprocess_data('data/training_data.csv')

    # Step 2: Create sequences for LSTM input
    look_back = 24  # Number of previous time steps to use as input (e.g., 24 hours)
    print(f"Creating sequences with a look-back of {look_back} time steps...")
    train_X, train_y = create_sequences(train_data, look_back)
    test_X, test_y = create_sequences(test_data, look_back)

    # Reshape data for LSTM (samples, time steps, features)
    train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
    test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)

    # Step 3: Build and train the LSTM model
    print("Building and training the LSTM model...")
    model = build_and_train_lstm(train_X, train_y, look_back)

    # Step 4: Evaluate the model and make predictions
    print("Evaluating the model and making predictions...")
    predictions, true_values, rmse = evaluate_and_predict(model, test_X, test_y, scaler)

    # Step 5: Visualize results
    print("Visualizing results...")
    plot_predictions(true_values, predictions)