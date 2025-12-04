
#================================================================================
#----------PREPROCESSING, MODEL LOADING, TRAINING & TESTING----------------------
#================================================================================

#importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Load data
df = pd.read_csv(r"C:\Users\Kuria\Desktop\Jackfruit Problem\train.csv")

# simple cleaning = fill empty categorical with "None" and numeric with 0
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna("None")
    else:
        df[col] = df[col].fillna(0)

# Separate features and target
X = df.drop(columns=["SalePrice", "Id"])
y = df["SalePrice"]

# Identify categorical & numeric columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# preprocess: One-hot encode categoricals
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

#binding the model and preprocessor
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# predict using test set
y_pred = model.predict(X_test)

# print evaluation metric/criteria
print("R² Score:", r2_score(y_test, y_pred))

#saving our model for later use
import joblib
joblib.dump(model, r"C:\Users\Kuria\Desktop\Jackfruit Problem\price_prediction.pkl")
print("\nModel saved as price_prediction.pkl")


#===============================================================================
#---------------IF INPUT IS TO BE TAKEN THROUGH PYTHON TERMINAL-----------------
#===============================================================================
'''   #defining a template row for storing input
template = X.head(1).copy()  # empty template row
template.loc[0] = X.mean(numeric_only=True)  # fill numeric with mean
for col in categorical_cols:
    template.loc[0, col] = "None"

# taking input
print("\nEnter house details:\n")

GrLivArea = float(input("Enter Ground Living Area: "))
OverallQual = int(input("Enter Overall Quality (1-10): "))
GarageCars = int(input("Enter number of garage cars: "))
TotalBsmtSF = float(input("Enter Total Basement SF: "))
FullBath = int(input("Enter number of full bathrooms: "))
YearBuilt = int(input("Enter year built: "))
Neighborhood = input("Enter neighborhood: ")
HouseStyle = input("Enter house style: ")

# putting into template
template.loc[0, 'GrLivArea'] = GrLivArea
template.loc[0, 'OverallQual'] = OverallQual
template.loc[0, 'GarageCars'] = GarageCars
template.loc[0, 'TotalBsmtSF'] = TotalBsmtSF
template.loc[0, 'FullBath'] = FullBath
template.loc[0, 'YearBuilt'] = YearBuilt
template.loc[0, 'Neighborhood'] = Neighborhood
template.loc[0, 'HouseStyle'] = HouseStyle

#final prediction for user
prediction = model.predict(template)[0]
print("\nPredicted House Price:", prediction)      '''