import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("dataset/House_Rent_Dataset.csv")

# Combine city + locality
df["Location"] = df["City"] + "_" + df["Area Locality"]

# Encoders
le_city = LabelEncoder()
le_area = LabelEncoder()
le_furnish = LabelEncoder()
le_location = LabelEncoder()

df["City"] = le_city.fit_transform(df["City"])
df["Area Type"] = le_area.fit_transform(df["Area Type"])
df["Furnishing Status"] = le_furnish.fit_transform(df["Furnishing Status"])
df["Location"] = le_location.fit_transform(df["Location"])

# Features
X = df[
[
"BHK",
"Size",
"Bathroom",
"City",
"Area Type",
"Furnishing Status",
"Location"
]
]

# Target
y = df["Rent"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(n_estimators=200, random_state=42)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/rent_model.pkl")

# Save encoders
joblib.dump(le_city, "models/city_encoder.pkl")
joblib.dump(le_area, "models/area_encoder.pkl")
joblib.dump(le_furnish, "models/furnish_encoder.pkl")
joblib.dump(le_location, "models/location_encoder.pkl")

print("Model trained successfully")