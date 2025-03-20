import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pickle  # For saving and loading the model

# Load dataset
data = pd.read_csv("test_Y3wMUE5_7gLdaTN.csv")

# Data Preprocessing
# Handle missing values
data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])
data['Married'] = data['Married'].fillna(data['Married'].mode()[0])
data['Dependents'] = data['Dependents'].fillna(data['Dependents'].mode()[0])
data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].median())
data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].median())
data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode()[0])

# Encode categorical variables
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
encoder = LabelEncoder()
for col in categorical_columns:
    data[col] = encoder.fit_transform(data[col])

# Define features and target variable
X = data.drop(['Loan_ID', 'Credit_History'], axis=1)
y = data['Credit_History']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2]
}

# Initialize RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, 
                           cv=3, scoring='accuracy', verbose=1, n_jobs=-1)

# Fit the GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Use the best model
best_rf_model = grid_search.best_estimator_

# Evaluate the best model
y_train_pred = best_rf_model.predict(X_train)
y_test_pred = best_rf_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred) * 100
test_accuracy = accuracy_score(y_test, y_test_pred) * 100

# Save the model and scaler
filename = 'loan_model.pkl'
pickle.dump((best_rf_model, scaler), open(filename, 'wb'))

#################### Streamlit App ####################

# Function to load the model and scaler
def load_model():
    with open('loan_model.pkl', 'rb') as file:
        model, scaler = pickle.load(file)
    return model, scaler

# Load the model and scaler
model, scaler = load_model()

# Custom CSS for styling
st.markdown(
    """
    <style>
        body {
            background-color: #f0f2f6;
            font-family: sans-serif;
        }
        .sidebar .sidebar-content {
            background-color: #3498db;
            color: white;
        }
        .stButton > button {
            color: #ffffff;
            background-color: #27ae60;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
        }
        .stButton > button:hover {
            background-color: #219653;
        }
        .reportview-container .main .block-container {
            max-width: 80%;
            padding-top: 50px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit App
st.title("Loan Eligibility Prediction App")

st.sidebar.header("Enter Applicant Details:")

# Input fields with units
gender = st.sidebar.selectbox("Gender", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
married = st.sidebar.selectbox("Married", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
dependents = st.sidebar.selectbox("Dependents", options=[0, 1, 2, 3], format_func=lambda x: str(x) + "+" if x == 3 else str(x))
education = st.sidebar.selectbox("Education", options=[1, 0], format_func=lambda x: "Graduate" if x == 1 else "Not Graduate")
self_employed = st.sidebar.selectbox("Self Employed", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
applicant_income = st.sidebar.number_input("Applicant Income (Annual)", value=50000, step=1000)
coapplicant_income = st.sidebar.number_input("Coapplicant Income (Annual)", value=20000, step=1000)
loan_amount = st.sidebar.number_input("Loan Amount (in USD)", value=15000, step=1000)
loan_amount_term = st.sidebar.number_input("Loan Amount Term (in Months)", value=360, step=30)
property_area = st.sidebar.selectbox("Property Area", options=[2, 1, 0], format_func=lambda x: "Urban" if x == 2 else ("Semiurban" if x == 1 else "Rural"))

# Prediction Section
st.header("Prediction:")
if st.button("Predict Eligibility"):
    user_data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Property_Area': property_area
    }

    user_df = pd.DataFrame([user_data])
    user_scaled = scaler.transform(user_df)

    prediction = model.predict(user_scaled)[0]
    if prediction == 1:
        st.success("The applicant is eligible for a loan.")
    else:
        st.error("The applicant is not eligible for a loan.")

# Model Performance Section
st.header("Model Performance:")
st.write(f"Training Accuracy: {train_accuracy:.2f}%")
st.write(f"Test Accuracy: {test_accuracy:.2f}%")

# ROC Curve Section
st.header("ROC Curve:")
y_test_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='red', linestyle='--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC) Curve')
ax.legend()
st.pyplot(fig)

# Feature Importance Section
st.header("Feature Importance:")
feature_importances = model.feature_importances_

fig, ax = plt.subplots()
sns.barplot(x=feature_importances, y=X.columns, ax=ax)
ax.set_title('Feature Importance')
st.pyplot(fig)
