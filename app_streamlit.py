import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

model  = joblib.load(Path(__file__).parent/"models"/"model.pkl")

st.title("ASG 05 MD - Ferdi Tan Leonardy - Spaceship Titanic Model Deployment")

with st.form("input_form"):
    st.write("### Data Penumpang")
    col1, col2 = st.columns(2)
    
    with col1:
        hp = st.selectbox("Home Planet", ["Earth", "Europa", "Mars"])
        cs = st.selectbox("CryoSleep", [False, True])
        dest = st.selectbox("Destination", ["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e"])
        age = st.number_input("Age", 0, 100, 25)
        vip = st.selectbox("VIP", [False, True])
        rs = st.number_input("Room Service", 0.0, 10000.0, 0.0)
        fc = st.number_input("Food Court", 0.0, 10000.0, 0.0)

    with col2:
        sm = st.number_input("Shopping Mall", 0.0, 10000.0, 0.0)
        spa = st.number_input("Spa", 0.0, 10000.0, 0.0)
        vrd = st.number_input("VR Deck", 0.0, 10000.0, 0.0)
        group = st.number_input("Group ID", 0, 9999, 1000)
        deck = st.selectbox("Deck", ["A", "B", "C", "D", "E", "F", "G", "T"])
        side = st.selectbox("Side (P/S)", ["P", "S"])
    
    submitted = st.form_submit_button("Predict Transported Status")

if submitted:
    input_data = pd.DataFrame([{
        'HomePlanet': hp,
        'CryoSleep': cs,
        'Destination': dest,
        'Age': age,
        'VIP': vip,
        'RoomService': rs,
        'FoodCourt': fc,
        'ShoppingMall': sm,
        'Spa': spa,
        'VRDeck': vrd,
        'Group': group,
        'Deck': deck,
        'Side': side
    }])
    try:
        prediction = model.predict(input_data)
        res = "Transported" if prediction[0] == 1 else "Not Transported"
        
        if prediction[0] == 1:
            st.success(f"Hasil Prediksi: **{res}**")
        else:
            st.error(f"Hasil Prediksi: **{res}**")
            
    except Exception as e:
        st.error(f"Error dalam prediksi: {e}")
        st.info("Pastikan urutan dan nama kolom di Pipeline sama dengan di DataFrame ini.")