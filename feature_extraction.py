import pandas as pd

# Load the smart meter data
df = pd.read_csv('smart_meter.csv')

# Convert the 'x_Timestamp' column to datetime to extract the date
df['x_Timestamp'] = pd.to_datetime(df['x_Timestamp'])

# Group by date (without time) and calculate the mean for each day
df['date'] = df['x_Timestamp'].dt.date  # Extract just the date part
daily_avg = df.groupby('date').agg({
    't_kWh': 'mean',
    'z_Avg Voltage (Volt)': 'mean',
    'z_Avg Current (Amp)': 'mean',
    'y_Freq (Hz)': 'mean',
    'meter': 'first'  # Keep the first value of 'meter' for each day
}).reset_index()

# Save the reduced dataset to a new CSV file
daily_avg.to_csv('smart_meter_reduced.csv', index=False)

print("Reduced dataset saved as 'smart_meter_reduced.csv'.")
