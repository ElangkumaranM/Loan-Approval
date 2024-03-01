import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load your dataset (replace 'your_dataset.csv' with your actual CSV file)
file_path = "convert.csv"
df = pd.read_csv(file_path)

# Include only the relevant features
selected_features = ['age', 'loan_amount', 'annual_income', 'cibil_score', 'loan_status']
df = df[selected_features]

# Encode categorical features
label_encoder = LabelEncoder()
df['loan_status'] = label_encoder.fit_transform(df['loan_status'])

# Split the data into features and target
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Train a Random Forest classifier
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Streamlit App
st.title("Loan Approval Prediction App")

# Sidebar for user input
st.sidebar.header("User Input Features")

def user_input_features():
    age = st.sidebar.text_input('Age', int(df['age'].mean()))
    loan_amount = st.sidebar.text_input('Loan Amount', int(df['loan_amount'].mean()))
    annual_income = st.sidebar.text_input('Annual Income', int(df['annual_income'].mean()))
    cibil_score = st.sidebar.text_input('CIBIL Score', int(df['cibil_score'].mean()))

    data = {
        'age': int(age),
        'loan_amount': int(loan_amount),
        'annual_income': int(annual_income),
        'cibil_score': int(cibil_score)
    }

    features = pd.DataFrame(data, index=[0])
    return features

# Display user input
user_input = user_input_features()
st.subheader('User Input:')
st.write(user_input)

# Predict function
def predict_loan_approval(user_input):
    prediction = model.predict(user_input)
    return prediction

# Predict button
if st.sidebar.button('Predict'):
    prediction = predict_loan_approval(user_input)
    st.subheader('Prediction:')
    st.write("Loan Approved" if prediction[0] == 1 else "Loan Not Approved")
