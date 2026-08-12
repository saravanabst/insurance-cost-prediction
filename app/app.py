import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# --------------------------------------------------
# Load Model
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Insurance Premium Predictor",
    layout="wide"
)


# --------------------------------------------------
# Application Title
# --------------------------------------------------

st.title("Insurance Premium Predictor")

st.markdown(
    """
    ### Machine Learning Based Insurance Cost Estimation

    Enter the customer's demographic and health information
    to estimate the insurance premium using a trained
    **Random Forest Regression model**.
    """
)

st.divider()

# --------------------------------------------------
# Customer Inputs
# --------------------------------------------------

st.header("Customer Information")

st.write(
    "Enter the customer's demographic and health information "
    "below."
)

col1, col2 = st.columns(2)


# --------------------------------------------------
# Column 1
# --------------------------------------------------

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=40,
        step=1
    )

    diabetes = st.selectbox(
        "Diabetes",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    blood_pressure = st.selectbox(
        "Blood Pressure Problems",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    transplants = st.selectbox(
        "Any Transplants",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    chronic_disease = st.selectbox(
        "Any Chronic Diseases",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )


# --------------------------------------------------
# Column 2
# --------------------------------------------------

with col2:

    height = st.number_input(
        "Height (cm)",
        min_value=100.0,
        max_value=220.0,
        value=170.0,
        step=1.0
    )

    weight = st.number_input(
        "Weight (kg)",
        min_value=30.0,
        max_value=200.0,
        value=70.0,
        step=1.0
    )

    allergies = st.selectbox(
        "Known Allergies",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    cancer_history = st.selectbox(
        "History of Cancer in Family",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    major_surgeries = st.number_input(
        "Number of Major Surgeries",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )


# --------------------------------------------------
# BMI Calculation
# --------------------------------------------------

height_m = height / 100

bmi = weight / (height_m ** 2)

st.info(f"Calculated BMI: {bmi:.2f}")


# --------------------------------------------------
# Prepare Model Input
# --------------------------------------------------

input_data = pd.DataFrame({
    "Age": [age],
    "Diabetes": [diabetes],
    "BloodPressureProblems": [blood_pressure],
    "AnyTransplants": [transplants],
    "AnyChronicDiseases": [chronic_disease],
    "Height": [height],
    "Weight": [weight],
    "KnownAllergies": [allergies],
    "HistoryOfCancerInFamily": [cancer_history],
    "NumberOfMajorSurgeries": [major_surgeries],
    "BMI": [bmi]
})


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button(
    "Predict Insurance Premium",
    type="primary",
    use_container_width=True
):

    # -------------------------------
    # Input Validation
    # -------------------------------

    if age < 18 or age > 100:
        st.error("Please enter an age between 18 and 100.")
        st.stop()

    if height <= 0:
        st.error("Height must be greater than zero.")
        st.stop()

    if weight <= 0:
        st.error("Weight must be greater than zero.")
        st.stop()

    if major_surgeries < 0 or major_surgeries > 10:
        st.error(
            "Number of major surgeries must be between 0 and 10."
        )
        st.stop()

    # -------------------------------
    # BMI Validation
    # -------------------------------

    if bmi <= 0:
        st.error(
            "BMI calculation resulted in an invalid value."
        )
        st.stop()

    if bmi < 10 or bmi > 80:
        st.warning(
            "The calculated BMI is outside the expected range. "
            "Please check the height and weight values."
        )

    # -------------------------------
    # Prediction
    # -------------------------------

    prediction = model.predict(input_data)

    predicted_premium = prediction[0]

    # -------------------------------
    # Prediction Result
    # -------------------------------

    st.divider()

    st.subheader("Prediction Result")

    st.metric(
        label="Estimated Insurance Premium",
        value=f"{predicted_premium:,.0f}"
    )

    st.info(
        "This estimate is generated by the trained "
        "Random Forest model using the demographic "
        "and health information provided."
    )

st.divider()

st.subheader("Model Information")

model_col1, model_col2, model_col3 = st.columns(3)

with model_col1:
    st.write("**Model**")
    st.write("Random Forest Regression")

with model_col2:
    st.write("**Target**")
    st.write("PremiumPrice")

with model_col3:
    st.write("**Input Features**")
    st.write("11")

st.divider()

st.subheader("How the Prediction Works")

st.write(
    """
    1. Customer demographic and health information is entered.
    2. BMI is calculated from height and weight.
    3. The engineered features are passed to the trained model.
    4. The Random Forest model estimates the insurance premium.
    """
)

st.divider()

st.caption(
    "Disclaimer: This prediction is for educational and analytical "
    "purposes only and should not be considered an actual insurance quote."
)



