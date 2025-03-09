import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Load the training data
train_data_path = "./data/training_data.csv"
train_df = pd.read_csv(train_data_path)

# Initialize a list to store predicted wind speeds for each event
all_predicted_windspeeds = []

# Loop through event_num 1 to 10
for event_num in range(1, 11):  # Loop from event_num 1 to 10
    # Load the test data for the current event
    test_data_path = f"./data/event_{event_num}.csv"
    test_df = pd.read_csv(test_data_path)

    # Use the windspeed values directly without shifting
    X_train = train_df[['windspeed']]
    y_train = train_df['windspeed']

    X_test = test_df[['windspeed']]
    y_test = test_df['windspeed']

    # Define and train the XGBoost model
    xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                              max_depth=5, alpha=10, n_estimators=10)
    xg_reg.fit(X_train, y_train)

    # Make initial predictions for the test data
    y_pred = xg_reg.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Event {event_num} - Mean Squared Error (MSE): {mse:.4f}")

    # Predict the next 120 windspeed values for the test data (recursive prediction)
    predicted_windspeeds = [y_pred[-1]]  # Start with the last predicted value as the initial value
    for i in range(1, 120):
        # Predict the next windspeed using the previous prediction
        next_prediction = xg_reg.predict([[predicted_windspeeds[-1]]])
        predicted_windspeeds.append(next_prediction[0])  # Corrected to append a scalar value

    # Append the predicted wind speeds for this event to the overall list
    all_predicted_windspeeds.append(predicted_windspeeds)

# Create the DataFrame
# Creating a dictionary where keys are the column names (event_number, 0, 1, 2, ..., 119)
data = {'event_number': [f'event_{event_num}' for event_num in range(1, 11)]}
for i in range(120):
    data[i] = [predicted_windspeeds[i] for predicted_windspeeds in all_predicted_windspeeds]

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv("windspeed_predictions.csv", index=False)

# Display the DataFrame
print(df)