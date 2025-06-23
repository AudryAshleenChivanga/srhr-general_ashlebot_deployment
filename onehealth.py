import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="🩺 U Health - Live PPH Risk Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- STYLING ---
st.markdown("""
    <style>
        .main {background-color: #ffe6f0;}
        .stButton>button {
            background-color: #ff69b4;
            color: white;
            font-weight: bold;
        }
        .stTabs [role="tab"] {
            background-color: #ffc0cb;
            color: black;
        }
    </style>
""", unsafe_allow_html=True)

# --- APP TITLE ---
st.title("🩺 U Health - Live Postpartum Hemorrhage Risk Monitor")
st.markdown("""
Real-time PPH risk monitoring system based on maternal health indicators collected live during birth.
Each record is linked to an anonymized Patient ID.
""")

# --- SIDEBAR: Simulate or Upload Data ---
st.sidebar.header("Live Data Options")
simulate_data = st.sidebar.checkbox("Simulate Live Data", value=True)
uploaded_file = st.sidebar.file_uploader("Upload Patient Dataset (CSV)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
elif simulate_data:
    import random
    data = pd.DataFrame([{
        "patient_id": f"PID{1000+i}",
        "oxytocin_level": round(random.uniform(1.0, 8.0), 2),
        "skin_temperature": round(random.uniform(35.5, 37.5), 1),
        "blood_loss_ml": random.randint(100, 1600),
        "uterine_tone": round(random.uniform(0.5, 4.5), 1),
        "uterine_contractions": random.randint(1, 5),
        "contractions_per_min": round(random.uniform(1.0, 4.0), 1),
        "blood_group": random.choice(["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
    } for i in range(8)])
else:
    data = pd.DataFrame()

# --- PROCESSING ---
if not data.empty:
    st.sidebar.markdown(f"*Live Data Loaded:* {data.shape[0]} patients")
    encoded_data = data.copy()
    encoded_data["blood_group"] = LabelEncoder().fit_transform(data["blood_group"])

    features = ["oxytocin_level", "skin_temperature", "blood_loss_ml", "uterine_tone", "uterine_contractions", "contractions_per_min", "blood_group"]

    if "pph_risk" not in data.columns:
        encoded_data["pph_risk"] = [random.choice(["Low Risk", "Moderate Risk", "High Risk"]) for _ in range(len(data))]

    target = "pph_risk"
    le = LabelEncoder()
    y = le.fit_transform(encoded_data[target])
    X = encoded_data[features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Patient Stream", "📋 Intake Form", "🤖 Live Risk Monitor", "📈 Feature Importance"])

    with tab1:
        st.subheader("Live Patient Data Feed")
        st.dataframe(data.set_index("patient_id"))

    with tab2:
        st.subheader("New Patient Intake Form")
        with st.form("patient_form"):
            patient_id = st.text_input("Patient ID", value="PID9999")
            oxytocin = st.slider("Oxytocin Level (mU/mL)", 0.0, 10.0, 4.0)
            skin_temp = st.slider("Skin Temperature (°C)", 34.0, 39.0, 36.8)
            blood_loss = st.slider("Estimated Blood Loss (ml)", 0, 2000, 500)
            uterine_tone = st.slider("Uterine Tone (arbitrary units)", 0.0, 5.0, 2.5)
            uterine_contractions = st.slider("Uterine Contractions (intensity count)", 0, 10, 3)
            contractions_per_min = st.slider("Contractions Per Minute", 0.0, 5.0, 2.5)
            blood_group = st.selectbox("Blood Group", options=["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
            submit = st.form_submit_button("Submit")

        if submit:
            new_patient = pd.DataFrame([{ 
                "patient_id": patient_id,
                "oxytocin_level": oxytocin,
                "skin_temperature": skin_temp,
                "blood_loss_ml": blood_loss,
                "uterine_tone": uterine_tone,
                "uterine_contractions": uterine_contractions,
                "contractions_per_min": contractions_per_min,
                "blood_group": blood_group
            }])
            st.success(f"Patient {patient_id} data submitted successfully.")
            data = pd.concat([data, new_patient], ignore_index=True)

    with tab3:
        st.subheader("Real-time Risk Prediction")
        for idx, row in data.iterrows():
            input_row = pd.DataFrame([{**row}])
            input_row["blood_group"] = LabelEncoder().fit(data["blood_group"]).transform([row["blood_group"]])[0]
            input_row = input_row[features]
            pred = model.predict(input_row)[0]
            prob = model.predict_proba(input_row)[0]
            label = le.inverse_transform([pred])[0]

            with st.expander(f"Patient {row['patient_id']} - Risk: {label}"):
                st.write("**Vitals:**")
                st.json(row.to_dict())
                st.write("**Model Confidence:**")
                st.progress(float(max(prob)))
                if label == "High Risk":
                    st.warning("🛑 Immediate action needed! Administer emergency oxytocin, monitor contractions, and prepare for transfusion.")
                elif label == "Moderate Risk":
                    st.info("⚠️ Monitor contractions and tone. May need assistance to stabilize uterine activity.")
                else:
                    st.success("✅ Stable. Continue routine monitoring.")

    with tab4:
        st.subheader("Feature Importance")
        importances = model.feature_importances_
        imp_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)
        st.bar_chart(imp_df.set_index("Feature"))

else:
    st.warning("Please simulate or upload live data to begin monitoring.")

# --- FOOTER ---
st.markdown("---")
st.markdown("© 2025 U Health — Built with ❤ for safer motherhood.")