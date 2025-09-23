# -------------------------
# 1. Import Libraries
# -------------------------
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# -------------------------
# 2. Load Dataset
# -------------------------
df = pd.read_csv("train_dataset.csv")
df = df.ffill()  # Fill missing values

# -------------------------
# 3. Encode Categorical Variables
# -------------------------
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# -------------------------
# 4. Features & Target
# -------------------------
X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']

# -------------------------
# 5. Train-Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
# 6. Train Model
# -------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -------------------------
# 7. Prediction Function
# -------------------------
def predict_loan(input_data):
    input_df = pd.DataFrame([input_data], columns=X.columns)
    prediction = model.predict(input_df)
    return "✅ Approved" if prediction[0]==1 else "❌ Not Approved"

# -------------------------
# 8. Prepare Dashboard Data (Decode Categorical)
# -------------------------
df_display = df.copy()
for col in categorical_cols:
    df_display[col] = df_display[col].map(lambda x: label_encoders[col].inverse_transform([x])[0])

# -------------------------
# 9. Prepare Mappings for Inputs
# -------------------------
categorical_mappings = {}
for col in categorical_cols:
    categorical_mappings[col] = {label: i for i, label in enumerate(label_encoders[col].classes_)}

# -------------------------
# 10. Streamlit UI
# -------------------------
st.set_page_config(page_title="Loan Prediction App", layout="wide")
st.title("💰 Loan Prediction Web App")

st.sidebar.header("Navigation")
option = st.sidebar.selectbox("Choose Option", ["Single Prediction", "Batch Prediction", "Dashboard"])

# -------------------------
# 11. Single Loan Prediction
# -------------------------
# --- Single Loan Prediction ---
# --- Single Loan Prediction ---
st.header("Single Loan Prediction")
st.write("Enter details of the applicant:")

user_data = {}
gender_choice = st.radio("Gender", ["Male", "Female"])
user_data["Gender"] = 1 if gender_choice == "Male" else 0

married_choice = st.radio("Married", ["Yes", "No"])
user_data["Married"] = 1 if married_choice == "Yes" else 0

dependents_choice = st.selectbox("Dependents", ["0", "1", "2", "3+"])
user_data["Dependents"] = 3 if dependents_choice == "3+" else int(dependents_choice)

# --- Education (Graduate/Not Graduate) ---
edu_choice = st.selectbox("Education", ["Graduate", "Not Graduate"])
user_data["Education"] = 1 if edu_choice == "Graduate" else 0

# --- Self_Employed (Yes/No) ---
se_choice = st.selectbox("Self Employed", ["Yes", "No"])
user_data["Self_Employed"] = 1 if se_choice == "Yes" else 0

# --- ApplicantIncome ---
user_data["ApplicantIncome"] = st.number_input("Applicant Income", value=float(X_train["ApplicantIncome"].median()))

# --- CoapplicantIncome ---
user_data["CoapplicantIncome"] = st.number_input("Coapplicant Income", value=float(X_train["CoapplicantIncome"].median()))

# --- LoanAmount ---
user_data["LoanAmount"] = st.number_input("Loan Amount", value=float(X_train["LoanAmount"].median()))

# --- Loan_Amount_Term ---
user_data["Loan_Amount_Term"] = st.number_input("Loan Amount Term", value=float(X_train["Loan_Amount_Term"].median()))

# --- Credit_History ---
ch_choice = st.selectbox("Credit History", ["Yes", "No"])
user_data["Credit_History"] = 1 if ch_choice == "Yes" else 0

# --- Property_Area ---
prop_choice = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
# map them into 0,1,2
area_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
user_data["Property_Area"] = area_map[prop_choice]





if st.button("Predict Single Loan"):
    result = predict_loan(user_data)
    if result == "✅ Loan Approved":
        st.success(result)
    else:
        st.error(result)



# -------------------------
# 12. Batch Prediction
# -------------------------
elif option == "Batch Prediction":
    st.header("Batch Loan Prediction")
    st.write("Upload a CSV file with the same columns as the training dataset.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        # Fill missing columns with default values
        for col in X.columns:
            if col not in batch_df.columns:
                if col in categorical_cols:
                    # Fill categorical missing with most frequent value
                    batch_df[col] = X_train[col].mode()[0]
                else:
                    # Fill numeric missing with median
                    batch_df[col] = X_train[col].median()

        # Encode categorical columns if needed
        for col in categorical_cols:
            if col in batch_df.columns:
                batch_df[col] = batch_df[col].map(lambda x: label_encoders[col].transform([x])[0] 
                                                  if x in label_encoders[col].classes_ else 0)

        # Predict
        predictions = model.predict(batch_df[X.columns])
        batch_df['Loan_Prediction'] = ["✅ Approved" if p==1 else "❌ Not Approved" for p in predictions]
        st.dataframe(batch_df)
        st.download_button("Download Predictions", batch_df.to_csv(index=False), "predictions.csv")


# -------------------------
# 13. Dashboard
# -------------------------
elif option =="Dashboard":
    st.header("Loan Approval Dashboard")

    # Overall Approval Pie Chart
    st.subheader("Overall Loan Approval %")
    approved_count = (y==1).sum()
    rejected_count = (y==0).sum()
    fig1, ax1 = plt.subplots()
    ax1.pie([approved_count, rejected_count], labels=["Approved","Rejected"], autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

    # Feature-wise Approval Charts
    for col in ['Gender', 'Married', 'Education', 'Property_Area', 'Credit_History']:
        if col in df_display.columns:
            st.subheader(f"Approval by {col}")
            group = df_display.groupby(col)['Loan_Status'].mean()
            fig, ax = plt.subplots()
            group.plot(kind='bar', ax=ax, color='lightblue')
            ax.set_ylabel("Approval Rate")
            st.pyplot(fig)
