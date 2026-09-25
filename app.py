import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os

st.set_page_config(page_title="Burnout Risk Predictor", layout="centered")

st.markdown(
    """
    <style>
    [data-testid="stNumberInputStepDown"] { display: none; }
    [data-testid="stNumberInputStepUp"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("early burnout risk prediction", anchor = False)

@st.cache_resource
def load_resources():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    rf_model = joblib.load(os.path.join(BASE_DIR, 'after_tuning.pkl'))
    test_data = joblib.load(os.path.join(BASE_DIR, 'X_test.pkl'))
    return rf_model, test_data

try:
    model, X_test = load_resources()
    feature_columns = X_test.columns.tolist()
except FileNotFoundError as e:
    st.error(f"System Error: Could not find model files. {e}")
    st.stop()

st.subheader("Data Input", anchor=False)
st.write("provide your information below")

# INTERFACE
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 65, 30)
    gender_input = st.selectbox("Gender", options=["Male", "Female", "Other"])
    emp_status_input = st.selectbox("Employment Status", options=["Full-Time", "Part-Time", "Freelance", "Unemployed"])
    work_hours = st.number_input("Work Hours Per Week", min_value=0, max_value=120, value=40)
    
    screen_time = st.slider("Screen Time Hours (Per Day)", 0, 24, 6)
    sleep_hours = st.slider("Sleep Hours (Per Day)", 0, 24, 7)
    sleep_quality = st.slider("Sleep Quality (1 = Poor, 10 = Excellent)", 1, 10, 5)

with col2:
    physical_activity = st.number_input("Physical Activity Hours (Per Week)", min_value=0.0, value=3.0)
    meditation_mins = st.number_input("Meditation Minutes (Per Day)", min_value=0.0, value=10.0)
    coffee_cups = st.number_input("Coffee Cups Per Day", min_value=0, value=2)
    stress_level = st.slider("Stress Level (1 = Low, 10 = High)", 1, 10, 5)
    occupation_input = st.selectbox("Occupation Category", options=["IT", "Healthcare", "Education", "Business", "Other"])
    education_input = st.selectbox("Education Level", options=["High School", "Bachelor", "Master", "PhD"])
    chronic_stress_input = st.selectbox("Experiencing Chronic Stress?", options=["No", "Yes"])

if st.button("Predict"):
    
    # BACKGROUND DATA TRANSFORMATION
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    emp_map = {"Full-Time": 0, "Part-Time": 1, "Freelance": 2, "Unemployed": 3}
    occ_map = {"IT": 0, "Healthcare": 1, "Education": 2, "Business": 3, "Other": 4}
    edu_map = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
    
    chronic_no = 1 if chronic_stress_input == "No" else 0
    chronic_yes = 1 if chronic_stress_input == "Yes" else 0

    wk_hours_boxcox = np.log1p(work_hours)
    screen_boxcox = np.log1p(screen_time)
    sleep_boxcox = np.log1p(sleep_hours)
    phys_log = np.log1p(physical_activity)
    med_log = np.log1p(meditation_mins)
    coffee_log = np.log1p(coffee_cups)

    processed_data = {
        'Age': age,
        'Gender': gender_map.get(gender_input, 0),
        'Employment_Status': emp_map.get(emp_status_input, 0),
        'Work_Hours_Per_Week': work_hours,
        'Screen_Time_Hours': screen_time,
        'Sleep_Hours': sleep_hours,
        'Sleep_Quality': sleep_quality,
        'Physical_Activity_Hours': physical_activity,
        'Meditation_Minutes': meditation_mins,
        'Coffee_Cups_Per_Day': coffee_cups,
        'Stress_Level': stress_level,
        'Occupation': occ_map.get(occupation_input, 0),
        'Education_Level': edu_map.get(education_input, 0),
        'Chronic_OHE_No': chronic_no,
        'Chronic_OHE_Yes': chronic_yes,
        'Work_Hours_Per_Week_BoxCox': wk_hours_boxcox,
        'Screen_Time_Hours_BoxCox': screen_boxcox,
        'Sleep_Hours_BoxCox': sleep_boxcox,
        'Physical_Activity_Hours_Log1p': phys_log,
        'Meditation_Minutes_Log1p': med_log,
        'Coffee_Cups_Per_Day_Log1p': coffee_log
    }

    input_df = pd.DataFrame([processed_data])[feature_columns]

    # PREDICTION & SHAP (Personalized Advice Only)
    prediction = model.predict(input_df)[0]
    
    risk_mapping = {
        0: "Low Risk (healthy)",
        1: "Moderate Risk (risky)",
        2: "High Risk (likely to be burnout)"
    }
    risk_label = risk_mapping.get(prediction, "Unknown Class")
    
    st.subheader("Result", anchor=False)
    st.write(f"**Assigned class:** {prediction} - {risk_label}")
    
    st.subheader("Personalized action plan", anchor=False)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    
    if isinstance(shap_values, list):
        instance_shap = shap_values[prediction][0]
    elif len(np.array(shap_values).shape) == 3:
        instance_shap = np.array(shap_values)[0, :, prediction]
    else:
        instance_shap = shap_values[0]
        
    feature_impact = dict(zip(feature_columns, instance_shap))
    
    top_pushers = sorted([item for item in feature_impact.items() if item[1] > 0], 
                         key=lambda x: x[1], reverse=True)
    
    advice_map = { # every advice for each factor here
        'Work_Hours_Per_Week_BoxCox': "decreasing your weekly work hours",
        'Work_Hours_Per_Week': "decreasing your weekly work hours",
        'Sleep_Hours_BoxCox': "increasing your daily sleep duration",
        'Sleep_Hours': "increasing your daily sleep duration",
        'Coffee_Cups_Per_Day_Log1p': "reducing your daily caffeine intake",
        'Coffee_Cups_Per_Day': "reducing your daily caffeine intake",
        'Screen_Time_Hours_BoxCox': "limiting your daily screen time",
        'Screen_Time_Hours': "limiting your daily screen time",
        'Physical_Activity_Hours_Log1p': "engaging in more physical activity",
        'Physical_Activity_Hours': "engaging in more physical activity",
        'Meditation_Minutes_Log1p': "incorporating more mindfulness or meditation",
        'Meditation_Minutes': "incorporating more mindfulness or meditation",
        'Stress_Level': "adopting effective stress-management techniques",
        'Sleep_Quality': "improving your overall sleep habits",
        'Chronic_OHE_Yes': "managing your chronic stress factors"
    }

    if prediction > 0:
        st.write("Based on the data analysis, your current risk level is primarily driven by specific lifestyle factors. To mitigate this risk, consider the following personalized adjustments:")
        
        advice_count = 0
        for feature, impact in top_pushers:
            if feature in advice_map and advice_count < 3:
                st.write(f"- Focus on {advice_map[feature]}.")
                advice_count += 1
                
        if advice_count == 0:
            st.write("Please consult a healthcare professional for a detailed lifestyle assessment.")
    else:
        st.write("Your current habits appear well-balanced. Maintain your routine to sustain this low risk level.")