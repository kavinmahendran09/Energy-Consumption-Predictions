import pandas as pd
import numpy as np

df = pd.read_csv('smart_meter.csv')

df['x_Timestamp'] = pd.to_datetime(df['x_Timestamp'])
df['date'] = df['x_Timestamp'].dt.date

full_date_range = pd.date_range(start='2020-01-01', end='2020-12-31')

daily_avg = df.groupby('date').agg({
    't_kWh': 'mean',
    'z_Avg Voltage (Volt)': 'mean',
    'z_Avg Current (Amp)': 'mean',
    'y_Freq (Hz)': 'mean',
    'meter': 'first'
}).reset_index()

daily_avg.set_index('date', inplace=True)
daily_avg = daily_avg.reindex(full_date_range.date, fill_value=np.nan)

daily_avg['t_kWh'].fillna(daily_avg['t_kWh'].mean(), inplace=True)
daily_avg['z_Avg Voltage (Volt)'].fillna(daily_avg['z_Avg Voltage (Volt)'].mean(), inplace=True)
daily_avg['z_Avg Current (Amp)'].fillna(daily_avg['z_Avg Current (Amp)'].mean(), inplace=True)
daily_avg['y_Freq (Hz)'].fillna(daily_avg['y_Freq (Hz)'].mean(), inplace=True)
daily_avg['meter'].fillna(method='ffill', inplace=True)

daily_avg.reset_index().rename(columns={'index': 'date'}).to_csv('smart_meter_reduced_imputed.csv', index=False)

print("Imputed dataset saved as 'smart_meter_reduced_imputed.csv'.")
