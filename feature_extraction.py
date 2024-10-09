import pandas as pd

# Load your dataset
smart_meter_df = pd.read_csv('smart_meter_reduced.csv')

# Ensure 'date' is in datetime format
smart_meter_df['date'] = pd.to_datetime(smart_meter_df['date'], errors='coerce')

# Generate a complete date range for the dataset
date_range = pd.date_range(start=smart_meter_df['date'].min(), end=smart_meter_df['date'].max(), freq='D')

# Find missing dates
missing_dates = date_range[~date_range.isin(smart_meter_df['date'])]

# Display missing dates
if len(missing_dates) > 0:
    print("Missing Dates:")
    for missing_date in missing_dates:
        print(missing_date.date())
else:
    print("No missing dates.")
