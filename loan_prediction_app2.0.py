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
    # Stop the app if the training data is missing
    st.stop()
else:
    train_data = pd.read_csv(train_csv_path)

# Encode categorical features for the model
le_gender = LabelEncoder()
train_data['Gender'] = le_gender.fit_transform(train_data['Gender'])
le_emp = LabelEncoder()
train_data['Employment_Status'] = le_emp.fit_transform(train_data['Employment_Status'])
le_loan_type = LabelEncoder()
train_data['Loan_Type'] = le_loan_type.fit_transform(train_data['Loan_Type'])

# Features & Target for the model
X = train_data.drop(columns=["Loan_Status"])
y = train_data["Loan_Status"]

# Train the RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# -------------------------
# 4. Storage for applicant data
# -------------------------
# Ensure the applications.csv file exists for data persistence
if not os.path.exists("applications.csv"):
    df_apps = pd.DataFrame(columns=[
        "ApplicantID", "Name", "Email", "Mobile", "Address", "Gender", "Income",
        "Credit_Score", "Employment_Status", "Loan_Type", "Loan_Amount",
        "Loan_Term", "Documents", "AI_Probability", "Admin_Decision", "Admin_Comments"
    ])
    df_apps.to_csv("applications.csv", index=False)
else:
    # Read the existing data on each app rerun to ensure it's up to date
    df_apps = pd.read_csv("applications.csv")

# -------------------------
# 5. Helper Functions
# -------------------------
def save_uploaded_files(files, applicant_id, loan_type):
    """Saves uploaded documents to a unique folder for each application."""
    saved_files = []
    # Create a unique folder path for the application's documents
    folder = f"uploads/{loan_type}_{applicant_id}"
    os.makedirs(folder, exist_ok=True)
    for f in files:
        path = os.path.join(folder, f.name)
        with open(path, "wb") as file:
            file.write(f.getbuffer())
        saved_files.append(path)
    return saved_files

def predict_loan(input_data):
    """Predicts loan approval probability using the trained model."""
    # Align features with the order used during training
    expected_features = list(X.columns)
    input_df = pd.DataFrame([input_data])[expected_features]
    prob = model.predict_proba(input_df)[0][1]
    return prob

# -------------------------
# 6. Navigation Sidebar
# -------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Applicant", "Admin"])

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
    gender = st.radio("Gender", ["Male", "Female"])
    income = st.number_input("Monthly Income", min_value=0)
    credit_score = st.number_input("Credit Score", min_value=0, max_value=1000)
    employment_status = st.radio("Employment Status", ["Employed", "Unemployed"])

    # Dictionary of required documents for each loan type
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
        # Save documents and get a list of paths
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
        
        # Create a new row for the application
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
            "Documents": str(saved_docs), # Store list as string
            "AI_Probability": ai_prob,
            "Admin_Decision": "Pending",
            "Admin_Comments": ""
        }])
        
        # Append the new application and save to CSV
        df_apps = pd.concat([df_apps, new_app], ignore_index=True)
        df_apps.to_csv("applications.csv", index=False)
        st.success(f"Application submitted! Your AI-predicted probability is: {ai_prob*100:.2f}%")

# -------------------------
# 8. Admin Page
# -------------------------
elif page == "Admin":
    st.title("🧾 Admin Dashboard")
    # Simple password-based authentication for the admin page
    if "admin_authenticated" not in st.session_state:
        st.session_state["admin_authenticated"] = False

    if not st.session_state["admin_authenticated"]:
        password = st.text_input("Enter Admin Password", type="password")
        if st.button("Login"):
            if password == "admin123":
                st.session_state["admin_authenticated"] = True
                st.success("Access Granted")
                st.rerun()
            else:
                st.error("Wrong password!")
    else:
        if df_apps.empty:
            st.info("No applications have been submitted yet.")
        else:
            # 📊 Statistical Analysis Section
            st.subheader("Statistical Analysis")
            col1, col2, col3 = st.columns(3)

            # Display key metrics
            total_apps = len(df_apps)
            col1.metric("Total Applications", total_apps)
            
            pending_count = (df_apps['Admin_Decision'] == 'Pending').sum()
            col2.metric("Pending for Review", pending_count)

            approved_count = (df_apps['Admin_Decision'] == 'Approved').sum()
            approval_rate = (approved_count / total_apps) * 100 if total_apps > 0 else 0
            col3.metric("Overall Approval Rate", f"{approval_rate:.2f}%")

            st.markdown("---")

            # Chart 1: Distribution of Loan Types
            st.subheader("Distribution by Loan Type")
            loan_type_counts = df_apps['Loan_Type'].value_counts()
            st.bar_chart(loan_type_counts)

            # Chart 2: Distribution of Admin Decisions
            st.subheader("Distribution of Admin Decisions")
            decision_counts = df_apps['Admin_Decision'].value_counts()
            st.bar_chart(decision_counts)

            st.markdown("---")
            
            # 📋 Review and Decision Section
            st.subheader("All Applications")
            # Display all applications in a dataframe
            st.dataframe(df_apps)

            st.subheader("Review and Decision")
            # Allow admin to select a specific application to review
            app_ids = df_apps['ApplicantID'].tolist()
            selected_id = st.selectbox("Select Applicant to Review:", app_ids)
            
            # Get the row for the selected application
            selected_app = df_apps[df_apps['ApplicantID'] == selected_id].iloc[0]

            st.markdown(f"### Applicant ID: {selected_app['ApplicantID']}")
            st.text(f"Name: {selected_app['Name']}")
            st.text(f"Email: {selected_app['Email']}")
            st.text(f"Mobile: {selected_app['Mobile']}")
            st.text(f"Loan Type: {selected_app['Loan_Type']}")
            st.text(f"AI Predicted Approval Probability: {selected_app['AI_Probability']*100:.2f}%")
            st.text(f"Current Decision: {selected_app['Admin_Decision']}")
            
            st.text("Documents Uploaded:")
            # Use a try-except block to handle cases where 'Documents' column is empty or invalid
            try:
                # Use eval() to convert the string representation of a list into a list
                docs_list = eval(selected_app['Documents'])
                if docs_list:
                    for f_path in docs_list:
                        # Create a download button for each document
                        with open(f_path, "rb") as file:
                            st.download_button(
                                label=f"Download {os.path.basename(f_path)}",
                                data=file,
                                file_name=os.path.basename(f_path),
                                mime="application/octet-stream"
                            )
                else:
                    st.info("No documents were uploaded for this application.")
            except (SyntaxError, FileNotFoundError):
                st.warning("No documents found or a format error occurred.")

            # Admin can make or change a decision
            current_decision = selected_app['Admin_Decision']
            decision = st.selectbox(
                "Update Decision:", 
                ["Pending", "Approved", "Rejected"], 
                index=["Pending", "Approved", "Rejected"].index(current_decision)
            )
            comments = st.text_area("Update Comments:", value=selected_app['Admin_Comments'])

            if st.button("Submit Decision"):
                # Find the row index of the selected application
                idx = df_apps[df_apps['ApplicantID'] == selected_id].index[0]
                df_apps.at[idx, "Admin_Decision"] = decision
                df_apps.at[idx, "Admin_Comments"] = comments
                df_apps.to_csv("applications.csv", index=False)
                st.success(f"Decision for Applicant {selected_id} updated!")
                st.rerun()
