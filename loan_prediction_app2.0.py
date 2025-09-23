# -------------------------
# 1. Import Libraries
# -------------------------
import streamlit as st
import pandas as pd
import os
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# -------------------------
# 2. Page config
# -------------------------
st.set_page_config(page_title="Loan Prediction App", layout="wide")

# -------------------------
# 3. Load Training Data
# -------------------------
train_csv_path = "loan_dataset_1000.csv"  # Your dataset
if not os.path.exists(train_csv_path):
    st.error(f"{train_csv_path} not found! Place CSV in the same folder.")
else:
    train_data = pd.read_csv(train_csv_path)

# Encode categorical features
le_gender = LabelEncoder()
train_data['Gender'] = le_gender.fit_transform(train_data['Gender'])
le_emp = LabelEncoder()
train_data['Employment_Status'] = le_emp.fit_transform(train_data['Employment_Status'])
le_loan_type = LabelEncoder()
train_data['Loan_Type'] = le_loan_type.fit_transform(train_data['Loan_Type'])

# Features & Target
X = train_data.drop(columns=["Loan_Status"])
y = train_data["Loan_Status"]

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# -------------------------
# 4. Storage for applicant data
# -------------------------
if not os.path.exists("applications.csv"):
    df_apps = pd.DataFrame(columns=[
        "ApplicantID","Name","Email","Mobile","Address","Gender","Income",
        "Credit_Score","Employment_Status","Loan_Type","Loan_Amount",
        "Loan_Term","Documents","AI_Probability","Admin_Decision","Admin_Comments"
    ])
    df_apps.to_csv("applications.csv", index=False)
else:
    df_apps = pd.read_csv("applications.csv")

# -------------------------
# 5. Helper Functions
# -------------------------
def save_uploaded_files(files, applicant_id, loan_type):
    saved_files = []
    folder = f"uploads/{loan_type}_{applicant_id}"
    os.makedirs(folder, exist_ok=True)
    for f in files:
        path = os.path.join(folder, f.name)
        with open(path, "wb") as file:
            file.write(f.getbuffer())
        saved_files.append(path)
    return saved_files

def predict_loan(input_data):
    # ✅ Fix: Align features with training order
    expected_features = list(X.columns)
    input_df = pd.DataFrame([input_data])[expected_features]
    prob = model.predict_proba(input_df)[0][1]
    return prob

# -------------------------
# 6. Navigation Sidebar
# -------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Applicant","Admin"])

# -------------------------
# 7. Applicant Page
# -------------------------
if page == "Applicant":
    st.title("💰 Loan Application")

    st.subheader("Applicant Details")
    name = st.text_input("Full Name")
    email = st.text_input("Email")
    mobile = st.text_input("Mobile Number")
    address = st.text_area("Address")
    gender = st.radio("Gender", ["Male","Female"])
    income = st.number_input("Monthly Income", min_value=0)
    credit_score = st.number_input("Credit Score", min_value=0, max_value=1000)
    employment_status = st.radio("Employment Status", ["Employed","Unemployed"])

    loan_docs_required = {
        "Car": ["Car Purchase Invoice", "ID Proof", "Income Proof"],
        "Personal": ["ID Proof", "Income Proof", "Bank Statement"],
        "Home": ["Property Documents", "ID Proof", "Income Proof", "Salary Slip"],
        "Education": ["Admission Certificate", "Fee Structure", "ID Proof"],
        "Business": ["Business License", "Bank Statement", "Income Proof"],
        "Gold": ["Gold Valuation Certificate", "ID Proof"],
        "Agriculture": ["Land Ownership Document", "Income Proof", "ID Proof"],
        "Medical": ["Medical Bills", "Prescription", "ID Proof"],
        "Travel": ["Travel Plan/Invoice", "ID Proof", "Bank Statement"],
        "Marriage": ["Invitation Card / Proof", "ID Proof", "Income Proof"]
    }

    loan_type = st.selectbox("Loan Type", list(loan_docs_required.keys()))
    st.write(f"📄 Required Documents for {loan_type} Loan:")
    for doc in loan_docs_required[loan_type]:
        st.write(f"- {doc}")
    documents = st.file_uploader("Upload Required Documents", accept_multiple_files=True)

    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_term = st.number_input("Loan Term (Months)", min_value=1)

    if st.button("Submit Application"):
        applicant_id = len(df_apps) + 1
        saved_docs = save_uploaded_files(documents, applicant_id, loan_type)
        input_model = {
            "Gender": le_gender.transform([gender])[0],
            "Employment_Status": le_emp.transform([employment_status])[0],
            "Loan_Type": le_loan_type.transform([loan_type])[0],
            "Income": income,
            "Credit_Score": credit_score,
            "Loan_Amount": loan_amount,
            "Loan_Term": loan_term
        }
        ai_prob = predict_loan(input_model)
        new_app = pd.DataFrame([{
            "ApplicantID": applicant_id,
            "Name": name,
            "Email": email,
            "Mobile": mobile,
            "Address": address,
            "Gender": gender,
            "Income": income,
            "Credit_Score": credit_score,
            "Employment_Status": employment_status,
            "Loan_Type": loan_type,
            "Loan_Amount": loan_amount,
            "Loan_Term": loan_term,
            "Documents": saved_docs,
            "AI_Probability": ai_prob,
            "Admin_Decision": "Pending",
            "Admin_Comments": ""
        }])
        df_apps = pd.concat([df_apps, new_app], ignore_index=True)
        df_apps.to_csv("applications.csv", index=False)
        st.success(f"Application submitted! AI Prediction Probability: {ai_prob*100:.2f}%")

# -------------------------
# 8. Admin Page
# -------------------------
elif page == "Admin":
    st.title("🧾 Admin Dashboard")
    if "admin_authenticated" not in st.session_state:
        st.session_state["admin_authenticated"] = False

    if not st.session_state["admin_authenticated"]:
        password = st.text_input("Enter Admin Password", type="password")
        if st.button("Login"):
            if password == "admin123":  # Change if needed
                st.session_state["admin_authenticated"] = True
                st.success("Access Granted")
            else:
                st.error("Wrong password!")
    else:
        pending_apps = df_apps[df_apps["Admin_Decision"]=="Pending"]
        if pending_apps.empty:
            st.info("No pending applications.")
        for i, row in pending_apps.iterrows():
            st.markdown(f"### Applicant ID: {row['ApplicantID']}")
            st.text(f"Name: {row['Name']}")
            st.text(f"Email: {row['Email']}")
            st.text(f"Mobile: {row['Mobile']}")
            st.text(f"Loan Type: {row['Loan_Type']}")
            st.text(f"AI Predicted Approval Probability: {row['AI_Probability']*100:.2f}%")
            st.text("Documents Uploaded:")
            for f in eval(row['Documents']):
                st.text(f"- {f}")
            decision = st.selectbox(f"Decision for Applicant {row['ApplicantID']}", ["Pending","Approved","Rejected"], key=i)
            comments = st.text_input(f"Comments for Applicant {row['ApplicantID']}", key=f"c{i}")
            if st.button(f"Submit Decision for Applicant {row['ApplicantID']}", key=f"btn{i}"):
                df_apps.at[i, "Admin_Decision"] = decision
                df_apps.at[i, "Admin_Comments"] = comments
                df_apps.to_csv("applications.csv", index=False)
                st.success(f"Decision for Applicant {row['ApplicantID']} saved!")
            
