from flask import Flask, render_template, request, jsonify
import pandas as pd
import matplotlib
import requests  # Add this for API call
from datetime import datetime
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt

app = Flask(__name__)

API_KEY = '656df056e407fd93b840d048945a7bbf'  # Replace with your actual API key

# Load data
energy_consumption_df = pd.read_csv('parameters.csv')
smart_meter_df = pd.read_csv('smart_meter.csv')

# Data preparation
energy_consumption_df['datetime'] = pd.to_datetime(energy_consumption_df['datetime'], errors='coerce')
smart_meter_df['x_Timestamp'] = pd.to_datetime(smart_meter_df['x_Timestamp'], errors='coerce')
merged_df = pd.merge(energy_consumption_df, smart_meter_df, how='inner', left_on='datetime', right_on='x_Timestamp')
merged_df['month'] = merged_df['datetime'].dt.month
merged_df['day_of_week'] = merged_df['datetime'].dt.dayofweek
merged_df['is_weekend'] = merged_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Features and target variable
X = merged_df[['temp', 'humidity', 'z_Avg Voltage (Volt)', 'z_Avg Current (Amp)', 'month', 'is_weekend']]
y = merged_df['t_kWh']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Random Forest': RandomForestRegressor(),
    'Linear Regression': LinearRegression(),
    'KNN Regression': KNeighborsRegressor()
}

# Train the models
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
        return weather_data
    else:
        print(f"Error: {weather_data['message']}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict(temp=None, humidity=None, date=None):
    if not temp or not humidity:
        # Get user input from form
        date = request.form['date']
        temp = float(request.form['temp'])
        humidity = float(request.form['humidity'])
    
    # Process date and prepare input features
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    month = date_obj.month
    is_weekend = 1 if date_obj.weekday() >= 5 else 0

    # Create feature set for prediction
    X_input = pd.DataFrame([[temp, humidity, 230, 5, month, is_weekend]],
                           columns=['temp', 'humidity', 'z_Avg Voltage (Volt)', 'z_Avg Current (Amp)', 'month', 'is_weekend'])

    # Predict using each model
    predictions = {name: model.predict(X_input)[0] for name, model in models.items()}

    # Calculate R² scores for the models
    r2_scores = {name: r2_score(y_test, model.predict(X_test)) for name, model in models.items()}

    # Prepare plots
    plot_urls = {}
    for name, model in models.items():
        img = io.BytesIO()
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, model.predict(X_test), label=name)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        plt.xlabel('Actual Consumption')
        plt.ylabel('Predicted Consumption')
        plt.title(f'{name} Prediction vs Actual')
        plt.legend()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_urls[name] = base64.b64encode(img.getvalue()).decode('utf8')

    # New plot for all models in one graph
    img_all = io.BytesIO()
    plt.figure(figsize=(10, 6))
    for name, model in models.items():
        plt.scatter(y_test, model.predict(X_test), label=name, alpha=0.6)  # Use alpha for better visibility
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal Prediction')
    plt.xlabel('Actual Consumption')
    plt.ylabel('Predicted Consumption')
    plt.title('All Models Prediction vs Actual')
    plt.legend()
    plt.savefig(img_all, format='png')
    plt.close()
    img_all.seek(0)
    plot_urls['All Model'] = base64.b64encode(img_all.getvalue()).decode('utf8')

    return render_template('results.html', predictions=predictions, r2_scores=r2_scores, plot_urls=plot_urls)

@app.route('/real-time', methods=['POST'])
def real_time():
    # Get today's date
    today = datetime.today().strftime('%Y-%m-%d')

    # Fetch real-time weather data (replace 'City Name' with your desired city)
    city = "Bareilly"
    weather_data = get_weather_data(API_KEY, city)

    if weather_data:
        temp = weather_data['main']['temp']
        humidity = weather_data['main']['humidity']
        return predict(temp=temp, humidity=humidity, date=today)
    else:
        return "Error fetching real-time weather data"

if __name__ == '__main__':
    app.run(debug=True)
