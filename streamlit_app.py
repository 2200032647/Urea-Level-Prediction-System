import streamlit as st
import pandas as pd
import pickle
import numpy as np

MODEL_PATH = "models/urea_model.pkl"

@st.cache_resource
def load_model(path=MODEL_PATH):
    with open(path,'rb') as f:
        data = pickle.load(f)
    return data

st.title("Urea Level Prediction — Demo")
st.write("Enter soil, weather and crop parameters to get a predicted urea application level (kg/ha).")

with st.form("input_form"):
    soil_ph = st.number_input("Soil pH", 4.5, 8.5, 6.5, step=0.1)
    soil_moisture = st.number_input("Soil moisture (%)", 0.0, 100.0, 20.0, step=0.1)
    temp_c = st.number_input("Temperature (°C)", -10.0, 50.0, 25.0, step=0.1)
    rainfall_mm = st.number_input("Rainfall (mm)", 0.0, 500.0, 50.0, step=1.0)
    nitrogen = st.number_input("Nitrogen (mg/kg)", 0.0, 200.0, 20.0, step=0.1)
    phosphorus = st.number_input("Phosphorus (mg/kg)", 0.0, 200.0, 10.0, step=0.1)
    potassium = st.number_input("Potassium (mg/kg)", 0.0, 1000.0, 150.0, step=1.0)
    organic_matter = st.number_input("Organic matter (%)", 0.0, 30.0, 3.0, step=0.1)
    previous_urea = st.number_input("Previous urea applied (kg/ha)", 0.0, 1000.0, 0.0, step=1.0)
    crop_type = st.selectbox("Crop type", ("rice","maize","wheat","cotton"))
    submitted = st.form_submit_button("Predict")

if submitted:
    df = pd.DataFrame([{
        "soil_ph": soil_ph,
        "soil_moisture": soil_moisture,
        "temp_c": temp_c,
        "rainfall_mm": rainfall_mm,
        "nitrogen": nitrogen,
        "phosphorus": phosphorus,
        "potassium": potassium,
        "organic_matter": organic_matter,
        "previous_urea": previous_urea,
        "crop_type": ["rice","maize","wheat","cotton"].index(crop_type)
    }])
    data = load_model()
    model = data['model']
    cols = data['columns']
    # one-hot and align
    df2 = pd.get_dummies(df, columns=['crop_type'], prefix='crop')
    for c in cols:
        if c not in df2.columns:
            df2[c] = 0
    df2 = df2[cols]
    pred = model.predict(df2)[0]
    st.success(f"Predicted urea application: {pred:.2f} kg/ha")
