import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

energy_consumption_df = pd.read_csv('parameters.csv')
smart_meter_df = pd.read_csv('smart_meter_reduced.csv')

energy_consumption_df['datetime'] = pd.to_datetime(energy_consumption_df['datetime'], errors='coerce')
smart_meter_df['date'] = pd.to_datetime(smart_meter_df['date'], errors='coerce')

merged_df = pd.merge(energy_consumption_df, smart_meter_df, how='inner', left_on='datetime', right_on='date')

if merged_df.empty:
    print("The merged DataFrame is empty. Please check your merge operation.")
else:
    X = merged_df[['temp', 'humidity', 'z_Avg Voltage (Volt)', 'z_Avg Current (Amp)']]
    y = merged_df['t_kWh']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Random Forest': RandomForestRegressor(),
        'Linear Regression': LinearRegression(),
        'KNN Regression': KNeighborsRegressor()
    }

    r2_scores = {}

    plt.figure(figsize=(18, 5))

    for i, (name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        r2_scores[name] = r2

        plt.subplot(1, 3, i + 1)
        plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
        plt.title(f'{name} Model: Actual vs Predicted')
        plt.xlabel('Actual Consumption (t_kWh)')
        plt.ylabel('Predicted Consumption (t_kWh)')
        plt.grid()

    plt.tight_layout()
    plt.show()

    for name, score in r2_scores.items():
        print(f"{name} R² Accuracy Score: {score:.4f}")
