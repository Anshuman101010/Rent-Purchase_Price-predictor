import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# load dataset
df = pd.read_csv("dataset/House_Rent_Dataset.csv")

# encode categorical features
le_city = LabelEncoder()
le_area = LabelEncoder()
le_furnish = LabelEncoder()

df["City"] = le_city.fit_transform(df["City"])
df["Area Type"] = le_area.fit_transform(df["Area Type"])
df["Furnishing Status"] = le_furnish.fit_transform(df["Furnishing Status"])

# select features
X = df[["BHK","Size","Bathroom","City","Area Type","Furnishing Status"]]

y = df["Rent"]

# split data
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# train model
model = RandomForestRegressor(n_estimators=200)
model.fit(X_train,y_train)

# save model
joblib.dump(model,"models/rent_model.pkl")

print("Model trained successfully")