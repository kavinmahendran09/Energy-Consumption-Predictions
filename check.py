import pandas as pd
import numpy as np

# Load the smart meter data
df = pd.read_csv('smart_meter.csv')

# Convert the 'x_Timestamp' column to datetime and extract the date part
df['x_Timestamp'] = pd.to_datetime(df['x_Timestamp'])
df['date'] = df['x_Timestamp'].dt.date

# Create a full date range for 2020
full_date_range = pd.date_range(start='2020-01-01', end='2020-12-31')

# Group by date and calculate the mean for each day
daily_avg = df.groupby('date').agg({
    't_kWh': 'mean',
    'z_Avg Voltage (Volt)': 'mean',
    'z_Avg Current (Amp)': 'mean',
    'y_Freq (Hz)': 'mean',
    'meter': 'first'  # Keep the first value of 'meter' for each day
}).reset_index()

# Reindex to ensure all dates are present, even missing ones
daily_avg.set_index('date', inplace=True)
daily_avg = daily_avg.reindex(full_date_range.date, fill_value=np.nan)

# Impute missing values by using forward fill or mean imputation
daily_avg['t_kWh'].fillna(daily_avg['t_kWh'].mean(), inplace=True)
daily_avg['z_Avg Voltage (Volt)'].fillna(daily_avg['z_Avg Voltage (Volt)'].mean(), inplace=True)
daily_avg['z_Avg Current (Amp)'].fillna(daily_avg['z_Avg Current (Amp)'].mean(), inplace=True)
daily_avg['y_Freq (Hz)'].fillna(daily_avg['y_Freq (Hz)'].mean(), inplace=True)
daily_avg['meter'].fillna(method='ffill', inplace=True)  # Forward fill for 'meter'

# Save the imputed dataset to a new CSV file
daily_avg.reset_index().rename(columns={'index': 'date'}).to_csv('smart_meter_reduced_imputed.csv', index=False)

print("Imputed dataset saved as 'smart_meter_reduced_imputed.csv'.")
