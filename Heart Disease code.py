# Import Libraries
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Heart Disease Prediction App",
    page_icon="❤️",
    layout="wide"

)
# Load model
with open("best_model.pkl", "rb") as f:
    best_model = pickle.load(f)

# Try loading the scaler, if it exists
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except : 
    scaler = None

def preprocess_and_predict(features):

# Convert input features to DataFrame
    input_df = pd.DataFrame([features])

# Get required columns for the model
    required_columns = best_model.feature_names_in_

# Add missing columns
    for col in required_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # or any default value you want to assign

# Arrange columns
    input_df = input_df[required_columns]

# Apply scaling if scaler is exists
    if scaler is not None:
        input_df = scaler.transform(input_df)

# Make prediction
    prediction = best_model.predict(input_df)
    probability = best_model.predict_proba(input_df)[:, 1]  # Probability of class 1
    return prediction[0], probability[0]



# ---------------Streamlit UI------------------
# Video section
try:
    video_file = open("heart video.mp4", "rb")
    video_bytes = video_file.read()
    st.video(video_bytes, format="video/mp4", autoplay=True, loop=True, muted=True)
except FileNotFoundError:
    st.warning("Video file not found.")

# Header Design
st.markdown("<h1 style='text-align: center; color: red;'>Heart Disease Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: black;'>Enter the details below to predict the likelihood of heart disease.</p>", unsafe_allow_html=True)
st.markdown("--------")

# Align input fields in two columns
col1, col2 = st.columns(2, gap="large")
with col1:
    st.subheader("📋 General & Clinical Metrics")
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    
    sex_input = st.selectbox("Sex", ["Male", "Female"])
    sex = 1 if sex_input == "Male" else 0
    
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3], 
                      help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic")
    
    trestbps = st.number_input("Resting Blood Pressure (in mm Hg)", 80, 200, 120)
    chol = st.number_input("Serum Cholesterol (in mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [0, 1])

with col2:
    st.subheader("🩺 Test Results & Reports")
    restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
    oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 6.0, 1.0, step=0.1)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

st.markdown("---")


# ------------------Teacher's Note:--------------------------------------- 
# Set page title
#st.title("Heart Disease Prediction App")
#age = st.number_input("Age", min_value=1, max_value=120, value=30)
#OR
#age = st.number_input("Age",1,120,50)
#sex = st.selectbox("Sex", ["Male", "Female"])
#sex = 1 if sex == "Male" else 0
#cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
#trestbps = st.number_input("Resting Blood Pressure",80, 200, 120)
#OR
#trestbps = st.number_input("Resting Blood Pressure (in mm Hg)", min_value=80, max_value=200, value=120)
#chol = st.number_input("Serum Cholesterol (in mg/dl)", 100, 600, 200)
#fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
#thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
#exang = st.selectbox("Exercise Induced Angina", [0, 1])
#oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest", 0.0, 6.0, 1.0, step=0.1)
#slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
#ca = st.selectbox("Number of Major Vessels (0-3) Colored by Fluoroscopy", [0, 1, 2, 3,4])
#thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

# input features as a dictionary
features = {
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
}

#------------------Prediction Button------------------
if st.button("Predict"):
    prediction, probability = preprocess_and_predict(features)
    if prediction == 1:
        st.error(f"⚠️ The model predicts that the patient is likely to have heart disease with a probability of {probability:.2f}.")
    else:
        st.success(f"✅ The model predicts that the patient is unlikely to have heart disease with a probability of {1 - probability:.2f}.")


