import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# Load the dataset
df = pd.read_csv('./data/training_data.csv')

# Define the threshold for identifying outliers
damage_threshold = 500  # Example threshold to exclude outliers

# Filter out outliers
df_filtered = df[df['damage'] < damage_threshold]

# Define features and target variable
x_all = df_filtered['windspeed']
y_all = df_filtered['damage']

# Define the exponential function
def exp_func(x, a, b):
    return a * np.exp(b * x)

# Initial guess for parameters
p0 = [1, 0.1]

# Fit the curve for all cities
popt_all, pcov_all = curve_fit(exp_func, x_all, y_all, p0=p0)
# Initialize lists to store total damages and prices for each event
total_damages = []
prices = []

# Loop through each event
for event_num in range(1, 11):  # Loop from event_num 1 to 10
    # Load the test data for the current event
    test_data_path = f"./data/event_{event_num}.csv"
    test_df = pd.read_csv(test_data_path)
    
    # Extract wind speeds
    wind_speeds = test_df['windspeed'].tolist()
    
    # Predict damage using the fitted model
    predicted_damages = exp_func(np.array(wind_speeds), *popt_all)
    
    # Calculate total damage
    total_damage = np.sum(predicted_damages)
    
    # Calculate the price
    price = 250 + total_damage / 2
    
    # Append the results
    total_damages.append(total_damage)
    prices.append(price)
    
    print(f"Event {event_num}, Total Damage: {total_damage}, Price: {price}")

# Create a DataFrame with the results
results_df = pd.DataFrame({
    'event_number': [f'event_{event_num}' for event_num in range(1, 11)],
    'price': prices
})

# Save the results to a CSV file
results_df.to_csv("event_prices.csv", index=False)

# Display the results
print(results_df)
