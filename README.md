# Insurance Cost Prediction

## Machine Learning-Based Insurance Premium Estimation

An end-to-end machine learning project that predicts estimated insurance premiums using customer demographic, physical, and health-related information.

The project combines exploratory data analysis, feature engineering, machine learning, model interpretation, business insights, interactive Tableau dashboards, and a deployed Streamlit web application.

## Live Demo
https://premiumcal.streamlit.app/

## Project Dashboard
https://public.tableau.com/app/profile/saravanakumar.kannan/viz/InsurancePremiumPredictionDashboard_17865594658270/Dashboard1

# Tableau Dashboards
The project includes three interactive Tableau Public dashboards covering exploratory analysis, health and risk factors, and key analytical insights.

## Dashboard 1 — EDA Overview
[EDA Overview](tableau/dashboard1_eda_overview.png)

Provides an overview of the dataset and major insurance premium patterns.

Key analysis includes:
- Premium distribution
- Age analysis
- Customer characteristics
- Premium by age group
- Key EDA metrics

## Dashboard 2 — Health & Risk Analysis

[Health & Risk Analysis](tableau/dashboard2_health_risk.png)

Focuses on the relationship between health conditions, risk factors, and insurance premiums.

Key analysis includes:
- Chronic disease impact
- Health condition comparisons
- Risk-factor analysis
- Premium differences across health conditions

## Dashboard 3 — EDA Insights

[EDA Insights](tableau/dashboard3_eda_insights.png)

Provides deeper analytical insights into relationships between customer characteristics and insurance premiums.

Key analysis includes:
- Correlation analysis
- BMI versus PremiumPrice
- Age versus PremiumPrice
- Health-factor analysis
- Interactive filters
- Key analytical observations

### View the Interactive Tableau Dashboard
https://public.tableau.com/app/profile/saravanakumar.kannan/viz/InsurancePremiumPredictionDashboard_17865594658270/Dashboard1

## GitHub Repository
https://github.com/saravanabst/insurance-cost-prediction

## Business Problem
Insurance premiums can vary depending on factors such as age, medical conditions, physical characteristics, previous surgeries, and other health-related indicators.

The objective of this project is to develop a machine learning model that estimates the expected insurance premium for an individual based on these characteristics.

The final model is deployed as an interactive web-based calculator so that users can enter customer information and receive a real-time premium estimate.

## Project Objectives
- Understand the factors associated with insurance premiums
- Perform exploratory data analysis
- Identify important demographic and health-related factors
- Engineer useful features such as BMI
- Train a Random Forest regression model
- Evaluate model performance
- Interpret model predictions
- Generate business insights
- Build interactive Tableau dashboards
- Develop a Streamlit prediction application
- Deploy the application to the cloud
- Maintain the project using Git and GitHub

## Project Workflow
Raw Data
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Interpretation
   ↓
Business Insights
   ↓
Random Forest Model
   ↓
Streamlit Application
   ↓
Cloud Deployment

## Technologies Used
### Programming & Data Analysis
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Joblib

### Machine Learning
Random Forest Regression
Cross Validation
Feature Importance
Permutation Importance
Model Evaluation
Prediction Reliability Analysis

### Visualization & Business Intelligence
Tableau Public
Matplotlib
Streamlit

### Development & Deployment
Jupyter Notebook
Git
GitHub
Streamlit Community Cloud

## Dataset
The dataset contains customer demographic, physical, and health-related information used to estimate insurance premiums.
 _____________________________________________________________________________
| Variable                | Description                                       |
| ----------------------- | ------------------------------------------------- |
| Age                     | Customer age                                      |
| Diabetes                | Diabetes indicator                                |
| BloodPressureProblems   | Blood pressure problem indicator                  |
| AnyTransplants          | Whether the customer has had a transplant         |
| AnyChronicDiseases      | Chronic disease indicator                         |
| Height                  | Customer height in centimetres                    |
| Weight                  | Customer weight in kilograms                      |
| KnownAllergies          | Known allergy indicator                           |
| HistoryOfCancerInFamily | Family history of cancer indicator                |
| NumberOfMajorSurgeries  | Number of major surgeries                         |
| BMI                     | Body Mass Index calculated from height and weight |
| PremiumPrice            | Insurance premium used as the prediction target   |
|_____________________________________________________________________________|

## Data Preparation
The dataset was examined and prepared before machine learning.

The preparation process included:
Reviewing the dataset structure
Checking data types
Checking missing values
Checking duplicate records
Reviewing numerical variables
Reviewing categorical/binary variables
Checking distributions
Identifying potential outliers
Preparing variables for machine learning

## Exploratory Data Analysis
Exploratory Data Analysis was performed to understand the characteristics of the dataset and identify relationships between customer characteristics and insurance premiums.

The analysis included:
Distribution analysis
Customer demographic analysis
Health-condition analysis
Age-group analysis
BMI analysis
Premium distribution
Correlation analysis
Premium versus health-factor analysis
Premium versus age analysis
Premium versus BMI analysis
Chronic disease impact
Health-condition comparisons

## Feature Engineering
A new feature, BMI (Body Mass Index), was created using height and weight.

The formula used was:
BMI = Weight / (Height*Height)

where height is converted from centimetres to metres.

BMI was included as an additional model feature to capture information about body composition that may not be represented by height and weight independently.

## Machine Learning Model
The final machine learning model is a:

Random Forest Regression Model

Random Forest was selected because it can capture nonlinear relationships and interactions between demographic and health-related variables.

The model was trained to predict: PremiumPrice

## Model Features
The final model uses 11 input features:
Age
Diabetes
BloodPressureProblems
AnyTransplants
AnyChronicDiseases
Height
Weight
KnownAllergies
HistoryOfCancerInFamily
NumberOfMajorSurgeries
BMI

## Target Variable
PremiumPrice

## Model Evaluation
The Random Forest model was evaluated using multiple techniques rather than relying on a single metric.

The evaluation included:
R² Score
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
Cross-validation
Prediction reliability analysis

## Model Interpretability
Model interpretation was performed to understand which variables contributed most strongly to the predictions.

## Feature Importance
The Random Forest feature importance analysis was used to identify features that contributed strongly to the model's predictions.

## Permutation Importance
Permutation importance was also performed.

This evaluates how model performance changes when individual features are randomly shuffled.Using both approaches provides a broader understanding of the model's important predictors.

## Correlation Analysis
Correlation analysis was used to understand linear relationships between numerical variables and PremiumPrice.

One of the stronger relationships observed in the analysis was:
Age → PremiumPrice
Correlation ≈ 0.698

This indicates a relatively strong positive linear relationship between age and premium in the dataset.Other variables showed weaker or moderate relationships.

However, correlation was not used as the only method for determining feature importance because machine learning models can capture nonlinear relationships that correlation does not capture.

## Key Business Insights
The analysis identified several important patterns.
1. Age
Age showed one of the strongest positive relationships with PremiumPrice. Older customers generally tended to have higher estimated insurance premiums.

2. Number of Major Surgeries
The number of major surgeries showed a positive relationship with PremiumPrice. Customers with a greater history of major surgeries tended to have higher estimated premiums in the dataset.

3. Health Conditions showed varying relationships with insurance premiums.
Health-related variables such as:
Diabetes
Blood pressure problems
Chronic diseases
Transplant history
Known allergies
Family history of cancer

4. BMI
BMI was engineered from height and weight and included as a model feature. The analysis explored whether differences in BMI were associated with differences in estimated insurance premiums.

5. Age and Health Factors
The analysis showed that insurance premium estimation is influenced by multiple factors rather than a single variable.

Therefore, the Random Forest model considers the combined effect of demographic, physical, and health-related characteristics.

## Tableau Dashboards

The project includes three Tableau Public dashboards.
Dashboard 1 — EDA Overview
Provides an overview of the dataset and major insurance premium patterns.

Key analysis includes:
Premium distribution
Age analysis
Customer characteristics
Premium by age group
Key EDA metrics
Dashboard 2 — Health & Risk Analysis

Focuses on the relationship between health conditions and insurance premiums.

Key analysis includes:
Chronic disease impact
Health condition comparisons
Risk-factor analysis
Premium differences across health conditions
Dashboard 3 — EDA Insights

Provides deeper analytical insights using:
Correlation analysis
BMI versus PremiumPrice
Age versus PremiumPrice
Health-factor analysis
Interactive filters
Key analytical observations
Tableau Public

https://public.tableau.com/app/profile/saravanakumar.kannan/viz/InsurancePremiumPredictionDashboard_17865594658270/Dashboard1

## Streamlit Web Application
The trained Random Forest model was integrated into a Streamlit web application.

The application allows users to enter:
Personal Information
Age
Height
Weight
Health Information
Diabetes
Blood Pressure Problems
Transplants
Chronic Diseases
Known Allergies
Family History of Cancer
Number of Major Surgeries

BMI is calculated automatically from height and weight. The application then sends the 11 required features to the trained Random Forest model and displays the estimated insurance premium.

## Application Features
The Streamlit application includes:
User-friendly input interface
Demographic inputs
Health-condition inputs
Automatic BMI calculation
Input validation
Binary Yes/No conversion
Real-time prediction
Estimated insurance premium
Model information
Explanation of the prediction process
Responsive web interface

## Deployment
The Streamlit application is deployed using Streamlit Community Cloud.

### Deployment Workflow
Local Development
       ↓
Git
       ↓
GitHub
       ↓
Streamlit Community Cloud
       ↓
Dependency Installation
       ↓
Model Loading
       ↓
Streamlit Application
       ↓
Public URL

## Live Application
https://premiumcal.streamlit.app/

## Installation
Clone the repository: git clone https://github.com/saravanabst/insurance-cost-prediction.git
Move into the project directory: cd insurance-cost-prediction
Create a virtual environment: python -m venv .venv
Activate the virtual environment on Windows: .venv\Scripts\activate
Install the required packages: pip install -r requirements.txt

## Run the Application Locally
Start the Streamlit application: streamlit run app/app.py
The application will open in the browser.

## Saved Machine Learning Model
The trained model is saved as: models/random_forest_model.pkl
The Streamlit application loads this model using Joblib.

## Model Development Notebook
The main model evaluation and interpretation work was performed in: 
notebooks/Model_Evaluation_Interpretability.ipynb

The notebook includes:
1. Import Libraries
2. Load Engineered Dataset
3. Prepare Data
4. Train Random Forest Model
5. Cross Validation
6. Feature Importance
7. Permutation Importance
8. Prediction Reliability
9. Business Insights
10. Save Final Model
11. Sensitivity Analysis-Age

## Reproducibility
The project uses:
requirements.txt for Python dependencies
Git/GitHub for version control
A saved Random Forest model for deployment
A structured project directory
Streamlit Community Cloud for deployment

This allows the project to be reproduced and deployed from the source repository.

## Limitations
This project has several limitations:
The model is trained on the available dataset and may not generalize to all populations.
The dataset may not contain all factors used by real insurance companies.
Machine learning predictions are estimates and not actual insurance quotations.
Random Forest predictions are not guaranteed to increase monotonically with individual features.
Predictions for values outside the range observed in the training data may be less reliable.
Correlation does not imply causation.
The model is intended for educational and analytical purposes.

## Future Improvements
Potential future improvements include:
Hyperparameter optimization
Comparison with additional regression algorithms
Explainable AI using SHAP
Prediction confidence or uncertainty estimates
Model monitoring
Automated model retraining
Improved input validation
Containerized deployment using Docker
CI/CD pipeline
API deployment using Flask or FastAPI
Integration with a production database
More comprehensive insurance datasets

## Project Highlights
This project demonstrates an end-to-end data science workflow:
Data
 ↓
Analysis
 ↓
Feature Engineering
 ↓
Machine Learning
 ↓
Evaluation
 ↓
Interpretability
 ↓
Business Insights
 ↓
Visualization
 ↓
Web Application
 ↓
Cloud Deployment


