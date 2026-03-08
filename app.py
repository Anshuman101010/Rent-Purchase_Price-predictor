from flask import Flask,render_template,request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("models/rent_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():

    bhk = int(request.form["bhk"])
    size = int(request.form["size"])
    bathroom = int(request.form["bathroom"])
    city = int(request.form["city"])
    area = int(request.form["area"])
    furnish = int(request.form["furnish"])

    features = np.array([[bhk,size,bathroom,city,area,furnish]])

    rent = model.predict(features)[0]

    price = rent * 200

    return render_template(
        "result.html",
        rent = round(rent,2),
        price = round(price,2)
    )

if __name__=="__main__":
    app.run(debug=True)