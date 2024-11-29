import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# Updated sample data including all relevant features
background_data = np.array([
    [25, 130, 80, 15.0, 98.0, 86, 0, 0, 0, 1],
    [35, 140, 90, 13.0, 98.0, 70, 0, 0, 0, 0],
    [29, 90, 70, 8.0, 100.0, 80, 0, 0, 0, 0],
    [30, 140, 85, 7.0, 98.0, 70, 0, 0, 0, 0],
    [35, 120, 60, 6.1, 98.0, 76, 0, 1, 0, 0]
])

# Load the pre-trained models
risk_level_model = joblib.load('risk_level_model.pkl')
labor_type_model = joblib.load('labor_type_model.pkl')

# Initialize SHAP explainers
risk_explainer = shap.TreeExplainer(risk_level_model)  # Use TreeExplainer for Random Forest
labor_explainer = shap.LinearExplainer(labor_type_model, background_data)  # Use LinearExplainer for Logistic Regression

# Function to predict Risk Level
def predict_risk_level(input_data):
    prediction = risk_level_model.predict([input_data])
    risk_mapping = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
    return risk_mapping[prediction[0]]

# Function to predict Labor Type
def predict_labor_type(input_data):
    prediction = labor_type_model.predict([input_data])
    labor_mapping = {0: 'Braxton Hicks', 1: 'True Labor'}
    return labor_mapping[prediction[0]]

# Function to generate SHAP visualizations
def generate_shap_plot(explainer, input_data, feature_names, model_type='tree'):
    # Ensure input_data is in the correct shape (2D array: 1 row, multiple columns)
    input_data = np.array([input_data])  # Convert input_data into a 2D array (1 row, multiple features)

    try:
        # Generate SHAP values
        if model_type == 'tree':
            # For tree-based models (Random Forest)
            shap_values = explainer.shap_values(input_data)

            # If it's a multi-class problem (list of SHAP values for each class), select the most probable class' SHAP values
            if isinstance(shap_values, list):
                # Choose the SHAP values for the class of interest (e.g., the one predicted by the model)
                predicted_class = risk_level_model.predict(input_data)[0]  # Get the predicted class index
                shap_values_class = shap_values[predicted_class]
            else:
                # If it's not a list, directly use the shap_values (in case of binary classification or single-class output)
                shap_values_class = shap_values

        else:
            # For non-tree-based models (Logistic Regression), shap_values is directly a single array
            shap_values_class = explainer.shap_values(input_data)

        # Create a SHAP summary plot for feature contributions
        shap.summary_plot(
            shap_values_class,
            features=input_data,
            feature_names=feature_names,
            plot_type="bar",  # Bar plot for feature importance
            show=False  # Don't display it immediately
        )

        # Save the plot to a file
        plt.tight_layout()
        plt.savefig('shap_contributions.png')
        plt.close()

    except Exception as e:
        st.error(f"Error generating SHAP plot: {e}")
        # Fallback to a basic feature importance plot if SHAP fails
        plt.figure(figsize=(10, 6))
        if hasattr(risk_level_model, 'feature_importances_'):
            importances = risk_level_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.title("Feature Importances")
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            plt.savefig('shap_contributions.png')
            plt.close()


# Streamlit UI
st.title("Maternal Health Risk Prediction App")
st.write("Predict the Risk Level and Labor Type based on health parameters with visual explanations.")

# Input fields for user data
age = st.number_input("Age", min_value=10, max_value=70, value=30)
systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=50, max_value=200, value=120)
diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=30, max_value=120, value=80)
bs = st.number_input("Blood Sugar Level (mmol/L)", min_value=2.0, max_value=20.0, value=7.0)
body_temp = st.number_input("Body Temperature (°F)", min_value=95.0, max_value=105.0, value=98.6)
heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=70)

# Additional inputs based on the features used in training the model
hypertension = st.number_input("Hypertension (0 or 1)", min_value=0, max_value=1, value=0)
blood_sugar_issue = st.number_input("Blood Sugar Issue (0 or 1)", min_value=0, max_value=1, value=0)
abnormal_body_temp = st.number_input("Abnormal Body Temperature (0 or 1)", min_value=0, max_value=1, value=0)
high_heart_rate = st.number_input("High Heart Rate (0 or 1)", min_value=0, max_value=1, value=0)

# Collect input data
input_data = [
    age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate,
    hypertension, blood_sugar_issue, abnormal_body_temp, high_heart_rate
]

# List of feature names corresponding to the trained model's input features
feature_names = [
    "Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate",
    "Hypertension", "Blood Sugar Issue", "Abnormal Body Temperature", "High Heart Rate"
]

# Prediction button
if st.button("Predict"):
    # Predictions
    risk_level = predict_risk_level(input_data)
    labor_type = predict_labor_type(input_data)

    # Display Predictions
    st.subheader(f"Predicted Risk Level: {risk_level}")
    st.subheader(f"Predicted Labor Type: {labor_type}")

    # Only attempt SHAP visualization if SHAP is available
    try:
        # Generate SHAP plots for Risk Level (Tree-based model)
        st.write("### Feature Contribution for Risk Level Prediction")
        generate_shap_plot(risk_explainer, input_data, feature_names, model_type='tree')
        st.image('shap_contributions.png')

        # Generate SHAP plots for Labor Type (Non-tree-based model)
        st.write("### Feature Contribution for Labor Type Prediction")
        generate_shap_plot(labor_explainer, input_data, feature_names, model_type='non-tree')
        st.image('shap_contributions.png')

    except Exception as e:
        st.error(f"Error in SHAP visualization: {e}")
        st.write("Falling back to standard feature importance visualization")

