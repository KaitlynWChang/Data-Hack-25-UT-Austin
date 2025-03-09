import pandas as pd

df = pd.read_csv("datahack2025-Ilgneous/data/event_1.csv")

# print(df.head())

print(df.describe())

# for col in df.columns:
#     print(f"{col}: {df[col].dtype}")

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


# Creates a moving average model, where window_size = 3 means that it is using the previous 3 values. 

# class MovingAverageModel:
#     def __init__(self, window_size=3):
#         self.window_size = window_size
#         self.predictions = None  # Stores predicted values
    
#     def fit(self, X, y):
#         """ Moving Average is a non-trainable model, so fit does nothing. """
#         pass  
    
#     def predict(self, X):
#         """ Predict using the moving average of the past `window_size` values. """
#         self.predictions = pd.Series(X.flatten()).rolling(window=self.window_size, min_periods=1).mean().values
#         return self.predictions
    
#     def evaluate(self, y_true, y_pred):
#         """ Compute MSE between actual and predicted values. """
#         mse = mean_squared_error(y_true, y_pred)
#         return mse

# ## Using Model to Predict the Training Data

# # Define feature (X) and target (y)
# X = df['windspeed'].values.reshape(-1, 1)  # Reshape for consistency
# y = df['windspeed'].values  # Assuming we are comparing to actual wind speed

# # Initialize model
# ma_model = MovingAverageModel(window_size=3)

# # Fit model (not needed but for consistency)
# ma_model.fit(X, y)

# # Predict
# y_pred = ma_model.predict(X)

# # Evaluate
# mse = ma_model.evaluate(y, y_pred)

# # Add predictions to dataframe
# df['predicted_windspeed'] = y_pred

# # Display results
# print(df[['windspeed', 'predicted_windspeed']].head(10))
# print(f"Mean Squared Error: {mse:.4f}")


