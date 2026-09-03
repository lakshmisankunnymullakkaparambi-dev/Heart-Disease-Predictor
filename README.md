# ❤️ Heart Disease Prediction Web Application

An interactive, end-to-end Machine Learning web application developed using **Streamlit** and **Python**. The application leverages a pre-trained classification model to predict the likelihood of heart disease in patients based on specific clinical and diagnostic metrics.

## 🔗 Live Deployment
🚀 **[Launch Live Application](https://heart-disease-predictor-mbcakblruzkpkmxrc8tkcv.streamlit.app/)**


## ✨ Key Features
- **Machine Learning Pipeline:** Integrates a pre-trained robust classification model (`best_model.pkl`) with dynamic feature scaling (`scaler.pkl`).
- **Modern User Interface:** Built with a highly responsive, user-friendly dual-column layout separating general demographics from advanced laboratory metrics.
- **Dynamic UX Elements:** Features custom CSS styling, looping background/header medical video animations, and real-time processing indicators.
- **Automated Preprocessing:** Features built-in data handling to structuralize input arrays, map categorical flags (e.g., gender encoding), and handle runtime data formats dynamically.

---

## 🔬 Clinical Parameters Covered
- **Demographics:** Age, Sex
- **Symptoms:** Chest Pain Type (`cp`)
- **Vitals:** Resting Blood Pressure (`trestbps`), Serum Cholesterol (`chol`), Fasting Blood Sugar (`fbs`)
- **Diagnostic Reports:** Resting ECG (`restecg`), Max Heart Rate (`thalach`), Exercise-Induced Angina (`exang`), ST Depression (`oldpeak`), ST Slope (`slope`), Major Vessels (`ca`), Thalassemia (`thal`)

---

## 🛠️ Technology Stack
- **Frontend Framework:** Streamlit
- **Data Engineering & Preprocessing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, Pickle
- **Version Control & Hosting:** Git, GitHub, Streamlit Community Cloud

---

## 📂 Project Architecture
```text
├── .ipynb_checkpoints/        # Notebook checkpoints
├── Heart Disease code.py      # Main Streamlit application source script
├── basic.ipynb                # Exploratory Data Analysis & Model Training Notebook
├── best_model.pkl             # Serialized pre-trained ML model binary
├── scaler.pkl                 # Serialized StandardScaler object binary
├── heart disease dataset.csv  # Base clinical training dataset
├── heart video.mp4            # Loopable medical header video asset
└── requirements.txt           # Cloud deployment environment dependencies
```
