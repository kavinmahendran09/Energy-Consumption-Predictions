import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load datasets
energy_consumption_df = pd.read_csv('parameters.csv')
smart_meter_df = pd.read_csv('smart_meter_reduced.csv')

# Convert date columns to datetime format
energy_consumption_df['datetime'] = pd.to_datetime(energy_consumption_df['datetime'], errors='coerce')
smart_meter_df['date'] = pd.to_datetime(smart_meter_df['date'], errors='coerce')

# Merge the two datasets on the datetime and date columns
merged_df = pd.merge(energy_consumption_df, smart_meter_df, how='inner', left_on='datetime', right_on='date')

# Check if merged data is empty
if merged_df.empty:
    print("The merged DataFrame is empty. Please check your merge operation.")
else:
    # Feature selection
    X = merged_df[['temp', 'humidity', 'z_Avg Voltage (Volt)', 'z_Avg Current (Amp)']]
    y = merged_df['t_kWh']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features (important for regularized models like Lasso)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize models
    models = {
        'Random Forest': RandomForestRegressor(),
        'Linear Regression': LinearRegression(),
        'KNN Regression': KNeighborsRegressor(),
        'Lasso Regression': Lasso(alpha=0.1)  # Lasso model with regularization strength alpha
    }

    # Store R² scores
    r2_scores = {}
    # Store predictions for comparison plot
    all_predictions = {name: [] for name in models.keys()}

    plt.figure(figsize=(18, 6))  # Create a figure for subplots

    # Loop over each model to train, predict and plot
    for i, (name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)  # Train the model
        y_pred = model.predict(X_test)  # Predict using the test set

        # Store predictions for comparison plot
        all_predictions[name] = y_pred

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        r2_scores[name] = r2

        # Create subplots for each model
        plt.subplot(1, 5, i + 1)
        plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # Ideal prediction line
        plt.title(f'{name} Model: Actual vs Predicted')
        plt.xlabel('Actual Consumption (t_kWh)')
        plt.ylabel('Predicted Consumption (t_kWh)')
        plt.grid()

    # Create a final plot that compares all models
    plt.subplot(1, 5, 5)
    for name, y_pred in all_predictions.items():
        plt.scatter(y_test, y_pred, alpha=0.5, label=name)

    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # Ideal prediction line
    plt.title('Comparison of All Models')
    plt.xlabel('Actual Consumption (t_kWh)')
    plt.ylabel('Predicted Consumption (t_kWh)')
    plt.legend()
    plt.grid()

    # Adjust layout for better visualization
    plt.tight_layout()
    plt.show()

    # Print R² scores for all models
    for name, score in r2_scores.items():
        print(f"{name} R² Accuracy Score: {score:.4f}")

    # Print MAE for all models
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mae = mean_squared_error(y_test, y_pred) ** 0.5  # Using RMSE as an alternative to MAE
        print(f"{name} Mean Absolute Error (MAE): {mae:.4f}")
