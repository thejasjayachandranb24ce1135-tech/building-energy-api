from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained energy model
model = joblib.load("building_energy_model_compressed.pkl")


@app.route("/")
def home():
    return "Building Energy Prediction API is running!"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    input_data = pd.DataFrame([{
        "floor_area_m2": data["floor_area_m2"],
        "hvac_load_w_m2": data["hvac_load_w_m2"],
        "lighting_load_w_m2": data["lighting_load_w_m2"],
        "equipment_load_w_m2": data["equipment_load_w_m2"],
        "outdoor_temperature_c": data["outdoor_temperature_c"],
        "occupancy_percent": data["occupancy_percent"],
        "operating_hours_per_day": data["operating_hours_per_day"],
        "humidity_percent": data["humidity_percent"],
        "day_of_week": data["day_of_week"],
        "is_weekend": data["is_weekend"]
    }])

    prediction = model.predict(input_data)[0]

    return jsonify({
        "predicted_energy_kwh_day": round(float(prediction), 2)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
