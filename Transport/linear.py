import os
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
# Load and map the region data
cluster_data_path = r"C:\Users\Hp EliteBook\Documents\Python_Assignments\Transport\training_data\cluster_map\cluster_map"
cluster_data = pd.read_csv(cluster_data_path, sep='\t', header=None)
cluster_data.columns = ['region_hash', 'region_id']
region_mapping = dict(zip(cluster_data['region_hash'], cluster_data['region_id']))

# Initialize storage for collecting all data
all_data = []

# Process each file (assuming these files cover the first three weeks)
folder_path = r"C:\Users\Hp EliteBook\Documents\Python_Assignments\Transport\training_data\order_data"
files = sorted(os.listdir(folder_path))
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
        
    # Print file name to check
    print("\nProcessing file:", file_name)
        
    # Read order data from CSV file
    order_data = pd.read_csv(file_path, header=None, sep='\t')
        
    # Extract necessary columns by column indices
    start_region_hash_index = 3 # Assuming start_region_hash is in the fourth column
    driver_id_index = 1 # Assuming driver ID is in the second column
    timestamp_index = 6 # Assuming timestamp is in the seventh column
    order_data = order_data[[start_region_hash_index, driver_id_index, timestamp_index]]
        
    # Rename columns for clarity
    order_data.columns = ["start_region_hash", "driver_id", "timestamp"]

    # Convert timestamp to datetime and extract day and hour
    order_data['timestamp'] = pd.to_datetime(order_data['timestamp'])
    order_data['day'] = order_data['timestamp'].dt.date
    order_data['hour'] = order_data['timestamp'].dt.hour

    # Filter out rows with NULL driver IDs
    order_data = order_data[order_data['driver_id'] != 'NULL']

    # Group timestamps by start region hash, day, and hour using Pandas' groupby
    grouped = order_data.groupby(['start_region_hash', 'day', 'hour'])
    # Initialize dictionary to store grouped timestamps by start region hash, day, and hour
    grouped_timestamps_start_region_hash_day_hour = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Initialize dictionary to store driver counts by start region hash, day, and hour
    driver_counts_start_region_hash_day_hour = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # Initialize dictionary to store the gap for each start region hash, day, and hour
    gap_start_region_hash_day_hour = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    # Iterate over the groups and store timestamps
    for (start_region_hash, day, hour), group in grouped:
        # Append timestamp to the list of timestamps for the start region hash, day, and hour
        grouped_timestamps_start_region_hash_day_hour[start_region_hash][day][hour].extend(group['timestamp'])

        # Count unique driver IDs for the start region hash, day, and hour
        driver_counts_start_region_hash_day_hour[start_region_hash][day][hour] = group['driver_id'].nunique()

        # Calculate the gap for the start region hash, day, and hour
        gap_start_region_hash_day_hour[start_region_hash][day][hour] = len(group) - group['driver_id'].nunique()

# Now we have grouped timestamps, driver counts, and gaps based on start region hash, day, and hour
# Print timestamps, driver counts, and gaps for each start region hash, day, and hour
print("\nTimestamps, Driver Counts, and Gaps for each start Region Hash, Day, and Hour:")
for start_region_hash, days_data in grouped_timestamps_start_region_hash_day_hour.items():
    print("\nStart Region Hash:", start_region_hash)
    for day, hours_data in days_data.items():
        print("\nDay:", day)
        for hour in range(24):  # Assuming data for 24 hours
            if hour in hours_data:  # Check if data available for this hour
                print("Hour:", hour)
                timestamps = hours_data[hour]
                print("Number of orders:", len(timestamps))
                print("Timestamps:", timestamps[:20])  # Print first 4 timestamps for each hour
                print("Driver Count:", driver_counts_start_region_hash_day_hour[start_region_hash][day][hour])
                print("Gap:", gap_start_region_hash_day_hour[start_region_hash][day][hour])
            else:
                print(f"No data available for hour {hour}")
for file_name in files:
    file_path = os.path.join(folder_path, file_name)
    # Assuming order data CSVs have these columns: [order_id, driver_id, passenger_id, start_region_hash, destination_region_hash, price, timestamp]
    order_data = pd.read_csv(file_path, sep='\t', header=None)
    order_data.columns = ['order_id', 'driver_id', 'passenger_id', 'start_region_hash', 'destination_region_hash', 'price', 'timestamp']
    order_data['timestamp'] = pd.to_datetime(order_data['timestamp'])
    order_data['hour'] = order_data['timestamp'].dt.hour
    order_data['minute'] = order_data['timestamp'].dt.minute
    order_data['day'] = order_data['timestamp'].dt.day
    order_data['region_id'] = order_data['start_region_hash'].map(region_mapping)
    all_data.append(order_data[['region_id', 'hour', 'minute', 'day', 'driver_id']])

# Combine all data into a single DataFrame
data = pd.concat(all_data)

# Define demand (number of orders) and supply (number o answered orders)
data['demand'] = order_data['timestamp'].notna().astype(int)  # Each row is an order
data['supply'] = data['driver_id'].notna().astype(int)

# Aggregate data to get total demand and supply per region, hour, and minute
# Aggregate data to get total demand and unique supply per region, hour, and minute
grouped_data = data.groupby(['region_id', 'hour', 'minute', 'day']).agg({'demand': 'sum', 'driver_id': pd.Series.nunique}).reset_index()
grouped_data.rename(columns={'driver_id': 'supply'}, inplace=True)

# Calculate the gap
grouped_data['gap'] = grouped_data['demand'] - grouped_data['supply']

# Create dictionaries to store grouped timestamps, driver counts, and gaps based on start region hash and hour
grouped_timestamps_start_region_hash_hour = defaultdict(lambda: defaultdict(list))
driver_counts_start_region_hash_hour = defaultdict(lambda: defaultdict(int))
gap_start_region_hash_hour = defaultdict(lambda: defaultdict(float))

# Group timestamps, driver counts, and gaps based on start region hash and hour
# Group timestamps, driver counts, and gaps based on start region hash and hour
for index, row in grouped_data.iterrows():
    start_region_hash = row['region_id']
    hour = int(row['hour'])  # Convert hour to integer
    minute = int(row['minute'])  # Convert minute to integer
    timestamp = pd.Timestamp(year=2023, month=3, day=1, hour=hour, minute=minute) # Assuming the year, month, and day are fixed
    driver_count = row['supply']
    gap = row['gap']
    grouped_timestamps_start_region_hash_hour[start_region_hash][hour].append(timestamp)
    driver_counts_start_region_hash_hour[start_region_hash][hour] += driver_count
    gap_start_region_hash_hour[start_region_hash][hour] = gap


# Split data into features and target
X = grouped_data[['region_id', 'hour', 'minute', 'day']]
y = grouped_data['gap']

# Train-test split based on time criteria
X_train = X[(X['hour'] < 10) | (X['hour'] >= 14)]
X_test = X[(X['hour'] >= 10) & (X['hour'] < 14)]
y_train = y[y.index.isin(X_train.index)]
y_test = y[y.index.isin(X_test.index)]

# Regression model
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)



# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'MSE: {mse}')

# Example prediction for a specific region, hour, and minute
example_region = 4
example_hour = 15
example_minute = 00
example_day = 25 # Example day for prediction (from 22 to 35)
predicted_gap = model.predict([[example_region, example_hour, example_minute, example_day]])
print(f'Predicted gap for region {example_region} at hour {example_hour}, minute {example_minute}, and day {example_day}: {predicted_gap[0]/10}')

# Formatting the output
output = pd.DataFrame({
    'Region ID': X_test['region_id'],
    'Time slot': X_test['hour'].astype(str) + '-' + X_test['minute'].astype(str),
    'Prediction value': predictions
})
print(output)

# Limit the data to 200 values
predicted_gaps = predictions[:200]
time_slots = (X_test['hour'] + X_test['minute'] / 60)[:200]  # Convert minute to hour

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(predicted_gaps, time_slots, marker='o', linestyle='-')
plt.title('Predicted Gaps over Time Slots (First 200 Values)')
plt.xlabel('Time Slots (Hour)')
plt.ylabel('Predicted Gaps')
plt.grid(True)
plt.show()