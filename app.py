from flask import Flask, render_template, request
import pandas as pd
import matplotlib
import requests
from datetime import datetime
import io
import base64
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

API_KEY = '656df056e407fd93b840d048945a7bbf'

global_input_data = None

# Load datasets
energy_consumption_df = pd.read_csv('parameters.csv')
smart_meter_df = pd.read_csv('smart_meter_reduced.csv')

# Data preparation
energy_consumption_df['datetime'] = pd.to_datetime(energy_consumption_df['datetime'], errors='coerce')
smart_meter_df['date'] = pd.to_datetime(smart_meter_df['date'], errors='coerce')
merged_df = pd.merge(energy_consumption_df, smart_meter_df, how='inner', left_on='datetime', right_on='date')

# Create new features
merged_df['month'] = merged_df['datetime'].dt.month
merged_df['day_of_week'] = merged_df['datetime'].dt.dayofweek
merged_df['is_weekend'] = merged_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Features and target variable
X = merged_df[['temp', 'tempmax', 'tempmin', 'feelslike', 'humidity', 'precip', 'windspeed', 'month', 'is_weekend']]
y = merged_df['t_kWh']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standard scaling for SVR and KNN (feature scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'KNN Regression': KNeighborsRegressor(),
    'SVR': SVR(),
    'Lasso Regression': Lasso(alpha=0.1)
}

# Hyperparameter tuning for Random Forest, KNN, and SVR using GridSearchCV
param_grid_rf = {'n_estimators': [50, 100, 150], 'max_depth': [5, 10, 20], 'min_samples_split': [2, 5, 10]}
param_grid_knn = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
param_grid_svr = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}

# Initialize grid searches
grid_search_rf = GridSearchCV(RandomForestRegressor(), param_grid_rf, cv=5, n_jobs=-1)
grid_search_knn = GridSearchCV(KNeighborsRegressor(), param_grid_knn, cv=5, n_jobs=-1)
grid_search_svr = GridSearchCV(SVR(), param_grid_svr, cv=5, n_jobs=-1)

# Fit grid searches
grid_search_rf.fit(X_train, y_train)
grid_search_knn.fit(X_train_scaled, y_train)
grid_search_svr.fit(X_train_scaled, y_train)

# Get the best models after hyperparameter tuning
best_rf = grid_search_rf.best_estimator_
best_knn = grid_search_knn.best_estimator_
best_svr = grid_search_svr.best_estimator_

# Update models dictionary with best models
models['Random Forest'] = best_rf
models['KNN Regression'] = best_knn
models['SVR'] = best_svr

# Train all models
for model in models.values():
    model.fit(X_train, y_train)

# Function to fetch real-time weather data using OpenWeather API
def get_weather_data(api_key, city):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'appid': api_key,
        'units': 'imperial'
    }

    response = requests.get(base_url, params=params)
    weather_data = response.json()

    if response.status_code == 200:
        main = weather_data['main']
        wind = weather_data['wind']
        return {
            'temp': main['temp'],
            'tempmax': main['temp_max'],
            'tempmin': main['temp_min'],
            'feelslike': main['feels_like'],
            'humidity': main['humidity'],
            'precip': weather_data.get('rain', {}).get('1h', 0),
            'windspeed': wind['speed']
        }
    else:
        print(f"Error: {weather_data['message']}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict(temp=None, tempmax=None, tempmin=None, feelslike=None, humidity=None, precip=None, windspeed=None, date=None):
    global global_input_data

    if not temp or not humidity:
        date = request.form['date']
        temp = float(request.form['temp'])
        tempmax = float(request.form['tempmax'])
        tempmin = float(request.form['tempmin'])
        feelslike = float(request.form['feelslike'])
        humidity = float(request.form['humidity'])
        precip = float(request.form['precip'])
        windspeed = float(request.form['windspeed'])

    # Process date and prepare input features
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    month = date_obj.month
    is_weekend = 1 if date_obj.weekday() >= 5 else 0

    # Create feature set for prediction
    X_input = pd.DataFrame([[temp, tempmax, tempmin, feelslike, humidity, precip, windspeed, month, is_weekend]],
                           columns=['temp', 'tempmax', 'tempmin', 'feelslike', 'humidity', 'precip', 'windspeed', 'month', 'is_weekend'])

    # Store this input data in a global variable to reuse in /analysis
    global_input_data = X_input

    # Predict using each model and round to 3 decimal places
    predictions = {name: round(model.predict(X_input)[0], 3) for name, model in models.items()}

    # Calculate R² and MAE scores for the models and round to 3 decimal places
    r2_scores = {name: round(r2_score(y_test, model.predict(X_test)), 3) for name, model in models.items()}
    mae_scores = {name: round(mean_absolute_error(y_test, model.predict(X_test)), 3) for name, model in models.items()}

    # Initialize plot_urls dictionary
    plot_urls = {}
    for name, model in models.items():
        img = io.BytesIO()
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, model.predict(X_test), label=f'{name} Predictions')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal Prediction')
        plt.xlabel('Actual Consumption (kWh)')
        plt.ylabel('Predicted Consumption (kWh)')
        plt.title(f'{name} Prediction vs Actual')
        plt.legend()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_urls[name] = base64.b64encode(img.getvalue()).decode('utf8')

    # Combined model plot (for all models)
    img_all = io.BytesIO()
    plt.figure(figsize=(10, 6))
    for name, model in models.items():
        plt.scatter(y_test, model.predict(X_test), label=name, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal Prediction')
    plt.xlabel('Actual Consumption (kWh)')
    plt.ylabel('Predicted Consumption (kWh)')
    plt.title('All Models Prediction vs Actual')
    plt.legend()
    plt.savefig(img_all, format='png')
    plt.close()
    img_all.seek(0)
    plot_urls['Combined Model'] = base64.b64encode(img_all.getvalue()).decode('utf8')

    return render_template(
        'predictions.html', 
        predictions=predictions, 
        r2_scores=r2_scores, 
        mae_scores=mae_scores, 
        plot_urls=plot_urls
    )

@app.route('/analysis', methods=['GET'])
def analysis():
    if global_input_data is not None:
        # Predict using the models with the stored input data
        predictions = {name: round(model.predict(global_input_data)[0], 3) for name, model in models.items()}

        # Calculate R² and MAE scores for the models and round to 3 decimal places
        r2_scores = {name: round(r2_score(y_test, model.predict(X_test)), 3) for name, model in models.items()}
        mae_scores = {name: round(mean_absolute_error(y_test, model.predict(X_test)), 3) for name, model in models.items()}

        # Initialize plot_urls for the analysis
        plot_urls = {}
        for name, model in models.items():
            img = io.BytesIO()
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, model.predict(X_test), label=name)
            plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
            plt.xlabel('Actual Consumption (kWh)')
            plt.ylabel('Predicted Consumption (kWh)')
            plt.title(f'{name} Prediction vs Actual')
            plt.legend()
            plt.savefig(img, format='png')
            plt.close()
            img.seek(0)
            plot_urls[name] = base64.b64encode(img.getvalue()).decode('utf8')

        return render_template('results.html', 
                               predictions=predictions, 
                               r2_scores=r2_scores, 
                               mae_scores=mae_scores, 
                               plot_urls=plot_urls)
    
    return "No input data available for analysis."

@app.route('/real-time', methods=['POST'])
def real_time():
    today = datetime.today().strftime('%Y-%m-%d')

    city = "Bareilly"
    weather_data = get_weather_data(API_KEY, city)

    if weather_data:
        return predict(
            temp=weather_data['temp'],
            tempmax=weather_data['tempmax'],
            tempmin=weather_data['tempmin'],
            feelslike=weather_data['feelslike'],
            humidity=weather_data['humidity'],
            precip=weather_data['precip'],
            windspeed=weather_data['windspeed'],
            date=today
        )

    return "Error fetching weather data."

if __name__ == '__main__':
    app.run(debug=True)
