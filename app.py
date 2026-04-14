from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests as http_requests
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

API_KEY = "c858e10f19d8ddef51725ae43cb42dd9"


def get_aqi_category(aqi):
    """Return AQI category string based on value."""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy (Sensitive)"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


@app.route("/")
def serve_index():
    return send_from_directory("static", "index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    city = data.get("city", "").strip()

    if not city:
        return jsonify({"error": "City name is required."}), 400

    # Step 1: Geocode the city
    try:
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={API_KEY}"
        geo_response = http_requests.get(geo_url, timeout=10)
        geo_response.raise_for_status()
        geo_data = geo_response.json()

        if not geo_data:
            return jsonify({"error": f"City '{city}' not found. Please try again."}), 404

        lat = geo_data[0]["lat"]
        lon = geo_data[0]["lon"]
        city_name = geo_data[0].get("name", city)
        country = geo_data[0].get("country", "")
    except Exception as e:
        return jsonify({"error": f"Failed to geocode city: {str(e)}"}), 500

    # Step 2: Fetch historical AQI data (11 days)
    aqi_history = []
    dates = []

    for i in range(10, -1, -1):
        date = datetime.now() - timedelta(days=i)
        timestamp = int(date.timestamp())

        try:
            url = (
                f"http://api.openweathermap.org/data/2.5/air_pollution/history"
                f"?lat={lat}&lon={lon}&start={timestamp}&end={timestamp + 86399}&appid={API_KEY}"
            )
            response = http_requests.get(url, timeout=10)
            resp_data = response.json()

            if "list" in resp_data and len(resp_data["list"]) > 0:
                aqi_components = [item["main"]["aqi"] for item in resp_data["list"]]
                avg_aqi = np.mean(aqi_components)
                aqi_scaled = (avg_aqi - 1) * 125

                aqi_history.append(float(aqi_scaled))
                dates.append(date.strftime("%Y-%m-%d"))
        except Exception:
            continue

    if len(aqi_history) < 5:
        return jsonify({"error": "Not enough historical data available for this city."}), 500

    # Step 3: Prepare training data
    window_size = 3
    X_train = []
    y_train = []

    for i in range(len(aqi_history) - window_size):
        X_train.append(aqi_history[i : i + window_size])
        y_train.append(aqi_history[i + window_size])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Step 4: Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X_tr, y_tr)

    y_pred_test = model.predict(X_te)
    mae = float(mean_absolute_error(y_te, y_pred_test))
    r2 = float(r2_score(y_te, y_pred_test))

    # Step 5: Predict next 5 days
    future_predictions = []
    current_window = list(aqi_history[-window_size:])

    for day_num in range(1, 6):
        pred = float(model.predict([current_window])[0])
        pred = max(0, pred)  # AQI can't be negative
        future_date = (datetime.now() + timedelta(days=day_num)).strftime("%Y-%m-%d")

        future_predictions.append(
            {
                "day": day_num,
                "date": future_date,
                "aqi": round(pred, 2),
                "category": get_aqi_category(pred),
            }
        )

        current_window.pop(0)
        current_window.append(pred)

    # Build history response
    history = []
    for d, a in zip(dates, aqi_history):
        history.append(
            {"date": d, "aqi": round(a, 2), "category": get_aqi_category(a)}
        )

    return jsonify(
        {
            "city": city_name,
            "country": country,
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "history": history,
            "predictions": future_predictions,
            "model_metrics": {"r2": round(r2, 4), "mae": round(mae, 4)},
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
