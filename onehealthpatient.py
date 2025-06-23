import streamlit as st
import pandas as pd
import numpy as np
import random

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="🩷 U Health - My Labor Companion",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- STYLING ---
st.markdown("""
    <style>
        .main {background-color: #fff0f5;}
        .stMetric {font-size: 22px;}
        .stButton>button {
            background-color: #ff69b4;
            color: white;
            font-weight: bold;
            border-radius: 12px;
        }
    </style>
""", unsafe_allow_html=True)

# --- TITLE ---
st.title("🩷 U Health - My Labor Companion")
st.markdown("""
Welcome! We're here to support you through your labor journey by keeping an eye on your body signals.
Don't worry — you're not alone 💪👶
""")

# --- SIMULATE LIVE VITALS ---
with st.spinner("Getting your latest readings..."):
    vitals = {
        "Oxytocin (mU/mL)": round(random.uniform(1.5, 7.5), 2),
        "Skin Temp (°C)": round(random.uniform(35.8, 37.3), 1),
        "Blood Loss (ml)": random.randint(50, 800),
        "Contraction Intensity": random.randint(1, 5),
        "Contractions per Min": round(random.uniform(1.0, 4.5), 1),
    }

# --- DISPLAY VITALS ---
st.header("📊 Your Current Body Signals")
col1, col2, col3 = st.columns(3)
col4, col5 = st.columns(2)

col1.metric("💗 Oxytocin", f"{vitals['Oxytocin (mU/mL)']} mU/mL")
col2.metric("🌡️ Skin Temp", f"{vitals['Skin Temp (°C)']} °C")
col3.metric("🩸 Blood Loss", f"{vitals['Blood Loss (ml)']} ml")
col4.metric("🧘 Contraction Intensity", vitals['Contraction Intensity'])
col5.metric("⏱️ Contractions/Min", vitals['Contractions per Min'])

# --- INTERPRETATION ---
st.subheader("🤖 What This Means")
if vitals['Blood Loss (ml)'] > 600 or vitals['Contraction Intensity'] >= 4:
    st.error("You're experiencing higher than usual contractions or bleeding. Stay calm — your medical team is being alerted.")
elif vitals['Contractions per Min'] > 3.5:
    st.warning("Your contractions are becoming more frequent. Try deep breaths and call for a midwife if needed.")
else:
    st.success("You're doing great! Your vitals are within safe ranges. Keep breathing slowly and stay hydrated 💧")

# --- PAST LOGS SIMULATION ---
st.subheader("🕒 Past 30 Minutes Summary")
past_data = pd.DataFrame({
    "Time": [f"-{i*5} min" for i in range(6)],
    "Contractions/min": np.clip(np.random.normal(loc=vitals['Contractions per Min'], scale=0.5, size=6), 1, 5),
    "Blood Loss (ml)": np.clip(np.random.normal(loc=vitals['Blood Loss (ml)'], scale=50, size=6), 0, 1200),
})
past_data = past_data[::-1]  # reverse for most recent at top

st.line_chart(past_data.set_index("Time"))

# --- EDUCATION & REASSURANCE ---
st.subheader("💬 Tips for You")
st.markdown("""
- Drink water regularly. 💧
- Ask your caregiver if you need support — you're allowed to!
- If you feel dizzy or in too much pain, don’t wait — alert someone 🛎️
""")

# --- FOOTER ---
st.markdown("---")
st.markdown("Built with ❤️ by U Health to support safer motherhood.")