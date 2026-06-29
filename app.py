import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("salary_model.pkl")

st.set_page_config(page_title="Employee Salary Prediction")

st.title("💼 Employee Salary Prediction")
st.write("Predict employee salary using Machine Learning")

# User Inputs
age = st.number_input("Age", min_value=18, max_value=70, value=25)

gender = st.selectbox("Gender", ["Male", "Female"])

education = st.selectbox(
    "Education Level",
    ["Bachelor", "Master", "PhD"]
)

experience = st.number_input(
    "Years of Experience",
    min_value=0,
    max_value=40,
    value=2
)

job = st.selectbox(
    "Job Title",
    [
        "Intern",
        "Junior Developer",
        "Software Engineer",
        "Data Analyst",
        "Business Analyst",
        "Data Scientist",
        "ML Engineer",
        "Project Manager",
        "Manager",
        "Director"
    ]
)

# Encoding dictionaries
gender_map = {
    "Male": 1,
    "Female": 0
}

education_map = {
    "Bachelor": 0,
    "Master": 1,
    "PhD": 2
}

job_map = {
    "Intern": 0,
    "Junior Developer": 1,
    "Software Engineer": 2,
    "Data Analyst": 3,
    "Business Analyst": 4,
    "Data Scientist": 5,
    "ML Engineer": 6,
    "Project Manager": 7,
    "Manager": 8,
    "Director": 9
}

if st.button("Predict Salary"):

    data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender_map[gender]],
        "Education Level": [education_map[education]],
        "Years of Experience": [experience],
        "Job Title": [job_map[job]]
    })

    prediction = model.predict(data)

    st.success(f"Predicted Salary: ₹ {prediction[0]:,.2f}")
