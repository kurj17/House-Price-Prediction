import wx
import joblib
import pandas as pd

#loading stuff
MODEL_PATH = r"C:\Users\Kuria\Desktop\House_Price_Prediction(Jackfruit_Problem)\price_prediction.pkl"  
TRAIN_CSV_PATH = r"C:\Users\Kuria\Desktop\House_Price_Prediction(Jackfruit_Problem)\train.csv"         

model = joblib.load(MODEL_PATH)
csv = pd.read_csv(TRAIN_CSV_PATH)

# Columns for user input
INPUT_COLS = [
    ("GrLivArea",   "Ground Living Area"),
    ("OverallQual", "Overall Quality (1-10)"),
    ("GarageCars",  "Garage Cars"),
    ("TotalBsmtSF", "Basement Area (SF)"),
    ("FullBath",    "Full Bathrooms"),
    ("YearBuilt",   "Year Built"),
    ("Neighborhood","Neighborhood"),
    ("HouseStyle",  "House Style"),
]

# Categorical columns in the dataset
categorical_cols = csv.select_dtypes(include=["object"]).columns

#making template row
def make_template_row():
    # Start with first row (to get correct structure)
    template = csv.head(1).copy()

    # Fill numeric columns with mean
    template.loc[0] = csv.mean(numeric_only=True)

    # Fill categoricals with "None"
    for col in categorical_cols:
        if col in template.columns:
            template.loc[0, col] = "None"

    return template


#starting the gui
app = wx.App()
frame = wx.Frame(None, title="House Price Predictor", size=(450, 400))
panel = wx.Panel(frame)

text_boxes = {}

#creating the labels and text boxes
y = 20
for col_name, label_text in INPUT_COLS:
    wx.StaticText(panel, label=label_text + ":", pos=(20, y))
    tb = wx.TextCtrl(panel, pos=(200, y), size=(200, 25))
    text_boxes[col_name] = tb
    y += 40


#main interface code 
def on_predict(event):
    try:
        df = make_template_row()

        for col_name, label_text in INPUT_COLS:
            value = text_boxes[col_name].GetValue().strip()

            # If user leaves it empty, keep template's default (mean / "None")
            if value == "":
                continue

            # Handle numeric vs categorical
            if col_name in ["Neighborhood", "HouseStyle"]:
                # Categorical – keep as string
                df.loc[0, col_name] = value
            else:
                # Numeric – try to convert to float/int
                try:
                    if "." in value:
                        df.loc[0, col_name] = float(value)
                    else:
                        df.loc[0, col_name] = int(value)
                except ValueError:
                    wx.MessageBox(f"Invalid number for {label_text}", "Input Error")
                    return

        # Make prediction
        pred = model.predict(df)[0]
        wx.MessageBox(f"Predicted Price: ₹{pred}", "Prediction")

    except Exception as e:
        wx.MessageBox("Error: " + str(e), "Error")


#prediction
btn = wx.Button(panel, label="Predict", pos=(200, y + 10))
btn.Bind(wx.EVT_BUTTON, on_predict)

frame.Show()
app.MainLoop()
