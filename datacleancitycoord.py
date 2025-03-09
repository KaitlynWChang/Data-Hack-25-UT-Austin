event_num = 1

import pandas as pd
import ast

train_df = pd.read_csv("./data/training_data.csv")
test_df = pd.read_csv(f"./data/event_{event_num}.csv")

# Define the CITIES dictionary
CITIES = [
    dict(name="Sparseville", x=63, y=35),
    dict(name="Tensorburg", x=214, y=378),
    dict(name="Bayes Bay", x=160, y=262),
    dict(name="ReLU Ridge", x=413, y=23),
    dict(name="GANopolis", x=318, y=132),
    dict(name="Gradient Grove", x=468, y=158),
    dict(name="Offshore A", x=502, y=356),
    dict(name="Offshore B", x=660, y=184),
]

# Convert the CITIES list to a DataFrame for easy lookup
cities_df = pd.DataFrame(CITIES)

# Merge the 'city' column from train_df with the x and y values
train_df = train_df.merge(cities_df[['name', 'x', 'y']], left_on='city', right_on='name', how='left')

# Check the results
print(train_df[['city', 'x', 'y']].head())