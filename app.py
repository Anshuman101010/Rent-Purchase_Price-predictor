from flask import Flask, render_template, request, redirect, url_for, jsonify
import os, json, datetime, uuid
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

@app.route("/my_predictions")
def my_predictions():
    return render_template("predictions.html")
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

        # Long-term rent calculation
        years = 15
        total_rent = rent * 12 * years

        # Calculate percentage difference
        difference = abs(price - total_rent)
        percent = (difference / price) * 100

        # --- Server-side backup: persist prediction with generated id ---
        try:
            record = {
                "id": str(uuid.uuid4()),
                "email": request.form.get('email') or 'unknown',
                "rent": float(rent),
                "price": float(price),
                "pred_area": int(size),
                "years": years,
                "total_rent": float(total_rent),
                "percent": float(round(percent,2)),
                "timestamp": datetime.datetime.utcnow().isoformat() + 'Z'
            }
            storage_path = os.path.join(os.path.dirname(__file__), 'models', 'predictions.json')
            if not os.path.exists(storage_path):
                with open(storage_path, 'w', encoding='utf-8') as f:
                    json.dump([record], f, ensure_ascii=False, indent=2)
            else:
                with open(storage_path, 'r+', encoding='utf-8') as f:
                    try:
                        arr = json.load(f)
                        if not isinstance(arr, list):
                            arr = []
                    except Exception:
                        arr = []
                    arr.append(record)
                    f.seek(0)
                    json.dump(arr, f, ensure_ascii=False, indent=2)
                    f.truncate()
        except Exception:
            pass

        return render_template(
            "predict.html",
            rent=round(rent, 2),
            price=round(price, 2),
            total_rent=round(total_rent, 2),
            years=years,
            percent=round(percent, 2),
            pred_area=size,
            areas=[500, 800, 1000, 1200, 1500],
            prices=[10000, 15000, 20000, 25000, 30000]
        )

    except Exception as e:
        return f"Error during prediction: {str(e)}"
    

# -------------------------------
# Run Flask App (moved to bottom)
# -------------------------------


@app.route('/save_prediction', methods=['POST'])
def save_prediction():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status":"error","message":"no json"}),400
        record = {
            "id": data.get('id') or str(uuid.uuid4()),
            "email": data.get('email') or 'unknown',
            "rent": float(data.get('rent',0)),
            "price": float(data.get('price',0)),
            "pred_area": int(data.get('pred_area',0)),
            "timestamp": data.get('timestamp') or datetime.datetime.utcnow().isoformat() + 'Z'
        }
        storage_path = os.path.join(os.path.dirname(__file__), 'models', 'predictions.json')
        if not os.path.exists(storage_path):
            with open(storage_path, 'w', encoding='utf-8') as f:
                json.dump([record], f, ensure_ascii=False, indent=2)
        else:
            with open(storage_path, 'r+', encoding='utf-8') as f:
                try:
                    arr = json.load(f)
                    if not isinstance(arr, list):
                        arr = []
                except Exception:
                    arr = []
                arr.append(record)
                f.seek(0)
                json.dump(arr, f, ensure_ascii=False, indent=2)
                f.truncate()
        return jsonify({"status":"ok","id":record['id']})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}),500


@app.route('/predictions_backup', methods=['GET'])
def predictions_backup():
    storage_path = os.path.join(os.path.dirname(__file__), 'models', 'predictions.json')
    if not os.path.exists(storage_path):
        return jsonify([])
    try:
        with open(storage_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                data = []
    except Exception:
        data = []
    return jsonify(data)


@app.route('/prediction_view')
def prediction_view():
    rent = float(request.args.get('rent', 0) or 0)
    price = float(request.args.get('price', 0) or 0)
    pred_area = int(request.args.get('pred_area', 0) or 0)
    years = int(request.args.get('years', 15) or 15)
    total_rent = round(rent * 12 * years, 2)
    percent = round((abs(price - total_rent) / price) * 100, 2) if price else 0

    return render_template(
        'prediction_view.html',
        rent=rent,
        price=price,
        pred_area=pred_area,
        years=years,
        total_rent=total_rent,
        percent=percent
    )


if __name__ == "__main__":
    app.run(debug=True)