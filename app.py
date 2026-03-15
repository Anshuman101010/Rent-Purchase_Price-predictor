from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np

app = Flask(__name__)

# -------------------------------
# Load Trained Model
# -------------------------------

model = joblib.load("models/rent_model.pkl")

le_city = joblib.load("models/city_encoder.pkl")
le_area = joblib.load("models/area_encoder.pkl")
le_furnish = joblib.load("models/furnish_encoder.pkl")
le_location = joblib.load("models/location_encoder.pkl")


# -------------------------------
# Login Page
# -------------------------------

@app.route("/")
def login():
    return render_template("login.html")


# -------------------------------
# Signup Page
# -------------------------------

@app.route("/signup")
def signup():
    return render_template("signup.html")


# -------------------------------
# Home Page (Prediction Form)
# -------------------------------

@app.route("/home")
def home():
    return render_template("index.html")


# -------------------------------
# Prediction Route
# -------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        bhk = int(request.form["bhk"])
        size = int(request.form["size"])
        bathroom = int(request.form["bathroom"])

        city_name = request.form["city"]
        locality = request.form["locality"]

        area_type_name = request.form["area_type"]
        furnish_name = request.form["furnish"]

        # Combine city + locality
        location_name = city_name + "_" + locality

        # Encode categorical values
        city = le_city.transform([city_name])[0]
        area_type = le_area.transform([area_type_name])[0]
        furnish = le_furnish.transform([furnish_name])[0]
        location = le_location.transform([location_name])[0]

        # Prepare input for model
        features = np.array([[bhk, size, bathroom, city, area_type, furnish, location]])

        # Predict rent
        rent = model.predict(features)[0]

        # Estimate house price
        price = rent * 200

        return render_template(
            "result.html",
            rent=round(rent, 2),
            price=round(price, 2)
        )

    except Exception as e:
        return f"Error during prediction: {str(e)}"


# -------------------------------
# Run Flask App
# -------------------------------

if __name__ == "__main__":
    app.run(debug=True)