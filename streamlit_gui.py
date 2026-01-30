import streamlit as st
import pandas as pd
import joblib
import streamlit.components.v1 as components

# -------------------------------
# PAGE CONFIG (BETTER UI)
# -------------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>

body {
    background-color: #f8f9fa;
}

.big-font {
    font-size:32px !important;
    font-weight:700;
}

.stButton>button {
    background-color:#4CAF50;
    color:black;   
    padding:10px 28px;
    font-size:18px;
    border-radius:10px;
}

.pred-card {
    padding:20px;
    background-color:white;
    border-radius:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.15);
    text-align:center;
    margin-top:20px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL + TRAINING CSV
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load(r"C:\Users\Kuria\OneDrive\Documents\IO Project\house_price_prediction.pkl")

@st.cache_data
def load_training_csv():
    df = pd.read_csv(r"C:\Users\Kuria\OneDrive\Documents\IO Project\train.csv")
    df = df.drop(columns=["SalePrice", "Id"])
    return df

model = load_model()
train_df = load_training_csv()

# Template row with dtypes
template = pd.DataFrame({col: [None] for col in train_df.columns})
for col in template.columns:
    if train_df[col].dtype == "object":
        template[col] = "None"
    else:
        template[col] = 0

# -------------------------------
# SIDEBAR NAVIGATION
# -------------------------------
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Predict Price", "📊 Tableau Dashboard"])

# -------------------------------
# PAGE 1: PREDICTION
# -------------------------------
if page == "🏠 Predict Price":

    st.markdown("<div class='big-font'>🏡 House Price Prediction</div>", unsafe_allow_html=True)
    st.write("Fill in the details below to estimate the property price.")

    # Two-column UI layout
    col1, col2 = st.columns(2)

    with col1:
        GrLivArea = st.number_input("Above-ground living area (sq ft)", min_value=100, max_value=10000, value=1500)
        OverallQual = st.slider("Overall Quality (1–10)", 1, 10, 5)
        GarageCars = st.slider("Garage Capacity (cars)", 0, 5, 2)
        YearBuilt = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000)

    with col2:
        TotalBsmtSF = st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=5000, value=900)
        FullBath = st.slider("Full Bathrooms", 0, 5, 2)
        Neighborhood = st.text_input("Neighborhood (e.g., NAmes, CollgCr)")
        HouseStyle = st.text_input("House Style (e.g., 1Story, 2Story)")

    predict_btn = st.button("🔮 Predict Price")

    if predict_btn:
        # Insert user input
        template.loc[0, 'GrLivArea'] = GrLivArea
        template.loc[0, 'OverallQual'] = OverallQual
        template.loc[0, 'GarageCars'] = GarageCars
        template.loc[0, 'TotalBsmtSF'] = TotalBsmtSF
        template.loc[0, 'FullBath'] = FullBath
        template.loc[0, 'YearBuilt'] = YearBuilt
        template.loc[0, 'Neighborhood'] = Neighborhood
        template.loc[0, 'HouseStyle'] = HouseStyle

        # Predict
        predicted_price = model.predict(template)[0]

        st.markdown(
            f"""
            <div class='pred-card'>
                <h2 style='color:#1E90FF;'>Predicted House Price</h2>
                <h1 style='color:#FFD700;'>${predicted_price:,.2f}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

# -------------------------------
# PAGE 2: TABLEAU DASHBOARD
# -------------------------------
elif page == "📊 Tableau Dashboard":

    st.markdown("<div class='big-font'>📊 House Market Tableau Dashboard</div>", unsafe_allow_html=True)
    st.write("This dashboard helps visualize real estate trends & patterns.")

    # **WORKING URL**
    tableau_url = "https://public.tableau.com/views/House_Price_17630990066190/HousePriceGraphs?:embed=yes&:showVizHome=no"

    components.iframe(tableau_url, width=1400, height=900, scrolling=True)