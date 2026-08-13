import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ==================================================
# LOAD MODEL
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"

model = joblib.load(MODEL_PATH)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Insurance Premium Predictor",
    layout="centered"
)


# ==================================================
# APPLICATION HEADER
# ==================================================

st.title("Insurance Premium Predictor")

st.markdown(
    "**Machine Learning Based Insurance Cost Estimation**"
)

st.caption(
    "Estimate insurance premiums using demographic and "
    "health information with a trained Random Forest "
    "Regression model."
)

st.divider()


# ==================================================
# CUSTOMER INFORMATION
# ==================================================

st.subheader("Customer Information")

st.caption(
    "Enter the customer's demographic and health information below."
)

st.markdown("---")


# ==================================================
# TWO INPUT COLUMNS
# ==================================================

col1, col2 = st.columns(2, gap="large")


# ==================================================
# LEFT CARD — PERSONAL INFORMATION
# ==================================================

with col1:

    st.markdown("### Personal Information")

    st.caption(
        "Basic demographic information"
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=40,
        step=1
    )

    height = st.number_input(
        "Height (cm)",
        min_value=50.0,
        max_value=250.0,
        value=170.0,
        step=1.0
    )

    weight = st.number_input(
        "Weight (kg)",
        min_value=20.0,
        max_value=250.0,
        value=70.0,
        step=1.0
    )

    diabetes = st.selectbox(
        "Diabetes",
        ["No", "Yes"]
    )

    blood_pressure = st.selectbox(
        "Blood Pressure Problems",
        ["No", "Yes"]
    )


# ==================================================
# RIGHT CARD — HEALTH INFORMATION
# ==================================================

with col2:

    st.markdown("### Health & Medical History")

    st.caption(
        "Medical conditions and risk indicators"
    )

    transplants = st.selectbox(
        "Any Transplants",
        ["No", "Yes"]
    )

    chronic_disease = st.selectbox(
        "Any Chronic Diseases",
        ["No", "Yes"]
    )

    allergies = st.selectbox(
        "Known Allergies",
        ["No", "Yes"]
    )

    cancer_history = st.selectbox(
        "History of Cancer in Family",
        ["No", "Yes"]
    )

    major_surgeries = st.number_input(
        "Number of Major Surgeries",
        min_value=0,
        max_value=3,
        value=0,
        step=1
    )


# ==================================================
# BMI CALCULATION
# ==================================================

height_m = height / 100

bmi = weight / (height_m ** 2)

st.info(
    f"Calculated BMI: **{bmi:.2f}**"
)


# ==================================================
# CONVERT YES / NO TO 1 / 0
# ==================================================

diabetes_value = 1 if diabetes == "Yes" else 0

blood_pressure_value = (
    1 if blood_pressure == "Yes" else 0
)

transplants_value = (
    1 if transplants == "Yes" else 0
)

chronic_disease_value = (
    1 if chronic_disease == "Yes" else 0
)

allergies_value = (
    1 if allergies == "Yes" else 0
)

cancer_history_value = (
    1 if cancer_history == "Yes" else 0
)


# ==================================================
# PREPARE MODEL INPUT
# ==================================================

input_data = pd.DataFrame({

    "Age": [age],

    "Diabetes": [diabetes_value],

    "BloodPressureProblems": [
        blood_pressure_value
    ],

    "AnyTransplants": [
        transplants_value
    ],

    "AnyChronicDiseases": [
        chronic_disease_value
    ],

    "Height": [height],

    "Weight": [weight],

    "KnownAllergies": [
        allergies_value
    ],

    "HistoryOfCancerInFamily": [
        cancer_history_value
    ],

    "NumberOfMajorSurgeries": [
        major_surgeries
    ],

    "BMI": [bmi]
})


# ==================================================
# PREDICTION BUTTON
# ==================================================

st.markdown("###")

predict_clicked = st.button(
    "Predict Insurance Premium",
    type="primary",
    use_container_width=True
)


# ==================================================
# PREDICTION PROCESS
# ==================================================

if predict_clicked:

    # --------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------

    if age < 18 or age > 100:

        st.error(
            "Please enter an age between 18 and 100."
        )

        st.stop()


    if height <= 0:

        st.error(
            "Height must be greater than zero."
        )

        st.stop()


    if weight <= 0:

        st.error(
            "Weight must be greater than zero."
        )

        st.stop()


    if major_surgeries < 0 or major_surgeries > 3:

        st.error(
            "Number of major surgeries must be "
            "between 0 and 3."
        )

        st.stop()


    # --------------------------------------------------
    # BMI VALIDATION
    # --------------------------------------------------

    if bmi <= 0:

        st.error(
            "BMI calculation resulted in an invalid value."
        )

        st.stop()


    if bmi < 10 or bmi > 80:

        st.warning(
            "The calculated BMI is outside the expected "
            "range. Please check the height and weight values."
        )


    # --------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------

    prediction = model.predict(input_data)

    predicted_premium = prediction[0]


    # ==================================================
    # PREDICTION RESULT
    # ==================================================

    st.divider()

    st.markdown(
        """
        <h2 style="text-align: center;">
            Prediction Result
        </h2>
        """,
        unsafe_allow_html=True
    )


    result_col1, result_col2 = st.columns(
        2,
        gap="large"
    )


    # --------------------------------------------------
    # PREMIUM
    # --------------------------------------------------

    with result_col1:

        st.metric(
            label="Estimated Insurance Premium",
            value=f"₹{predicted_premium:,.0f}"
        )


    # --------------------------------------------------
    # BMI
    # --------------------------------------------------

    with result_col2:

        st.metric(
            label="Calculated BMI",
            value=f"{bmi:.2f}"
        )


    # --------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------

    st.info(
        "This estimate is generated by the trained "
        "Random Forest model using the demographic and "
        "health information provided. "
        "It is for demonstration purposes and should "
        "not be considered an actual insurance quotation."
    )


# ==================================================
# MODEL INFORMATION
# ==================================================

st.divider()

st.subheader("Model Information")

st.caption(
    "Details of the machine learning model used for prediction."
)

model_info = {
    "Model": "Random Forest Regression",
    "Prediction Target": "PremiumPrice",
    "Number of Input Features": "11"
}

for label, value in model_info.items():

    info_col1, info_col2 = st.columns([1, 2])

    with info_col1:
        st.markdown(f"**{label}**")

    with info_col2:
        st.write(value)


# ==================================================
# HOW THE PREDICTION WORKS
# ==================================================

st.divider()

st.markdown(
    """
    <h2 style="text-align: center;">
        How the Prediction Works
    </h2>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    1. **Customer information is entered** using
       demographic and health inputs.

    2. **BMI is calculated automatically** from
       height and weight.

    3. The information is converted into the
       **11 model input features**.

    4. The trained **Random Forest model** predicts
       the estimated insurance premium.
    """
)


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Insurance Premium Predictor | "
    "Machine Learning Demonstration"
)
