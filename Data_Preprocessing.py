import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load your datasets (replace with your actual file paths)
energy_consumption_df = pd.read_csv('parameters.csv')
smart_meter_df = pd.read_csv('smart_meter.csv')

# Convert datetime columns to the same type
energy_consumption_df['datetime'] = pd.to_datetime(energy_consumption_df['datetime'], errors='coerce')
smart_meter_df['date'] = pd.to_datetime(smart_meter_df['date'], errors='coerce')

# Merge the DataFrames
merged_df = pd.merge(energy_consumption_df, smart_meter_df, how='inner', left_on='datetime', right_on='date')

# Check if merged_df is empty before proceeding
if merged_df.empty:
    print("The merged DataFrame is empty. Please check your merge operation.")
else:
    # Extract month and day of the week from the datetime column
    merged_df['month'] = merged_df['datetime'].dt.month
    merged_df['day_of_week'] = merged_df['datetime'].dt.dayofweek  # Monday=0, Sunday=6
    merged_df['is_weekend'] = merged_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)  # 1 for weekend, 0 for weekday

    # Select relevant features and target variable
    X = merged_df[['temp', 'humidity', 'z_Avg Voltage (Volt)', 'z_Avg Current (Amp)', 'month', 'is_weekend']]  # Update as necessary
    y = merged_df['t_kWh']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models
    models = {
        'Random Forest': RandomForestRegressor(),
        'Linear Regression': LinearRegression(),
        'KNN Regression': KNeighborsRegressor()
    }

    # Store R² scores for each model
    r2_scores = {}

    # Define color schemes for each model
    colors = {
        'Random Forest': 'green',
        'Linear Regression': 'blue',
        'KNN Regression': 'orange'
    }

    plt.figure(figsize=(18, 5))

    # Train and evaluate each model
    for i, (name, model) in enumerate(models.items()):
        # Fit the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        r2_scores[name] = r2

        # Plot predictions vs actual values using distinct colors
        plt.subplot(1, 3, i + 1)
        plt.scatter(y_test, y_pred, color=colors[name], alpha=0.6, label=f'{name} Predictions')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Perfect Prediction Line')  # Diagonal line
        plt.title(f'{name} Model: Actual vs Predicted')
        plt.xlabel('Actual Consumption (t_kWh)')
        plt.ylabel('Predicted Consumption (t_kWh)')
        plt.legend()
        plt.grid()

    plt.tight_layout()
    plt.show()

    # Print accuracy (R² scores) of all models
    for name, score in r2_scores.items():
        print(f"{name} R² Accuracy Score: {score:.4f}")
