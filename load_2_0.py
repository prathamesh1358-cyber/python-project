# -------------------------
# 1. Import Libraries
# -------------------------
import streamlit as st
import pandas as pd
import os
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# -------------------------
# 2. Email Config
# -------------------------
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "prathambuissmca7@gmail.com"        # <-- Replace with your Gmail
SENDER_PASSWORD = "kufdiickouvliyal"        # <-- Replace with App Password

def send_email(applicant_email, applicant_name, otp=None, decision=None, admin_comments=""):
    try:
        msg = MIMEMultipart("alternative")
        msg["From"] = SENDER_EMAIL
        msg["To"] = applicant_email

        if decision:
            msg["Subject"] = f"Loan Application Status: {decision}"
            html_body = f"""
            <html>
            <body>
                <p>Dear {applicant_name},</p>
                <p>Your loan application has been <b>{decision}</b>.</p>
                <p><b>Admin Comments:</b> {admin_comments if admin_comments else 'No additional comments provided.'}</p>
                <p>Thank you for using our Loan Management System!</p>
                <br>
                <p><i>Loan Management Team</i></p>
            </body>
            </html>
            """
        elif otp:
            msg["Subject"] = "Your OTP for Loan Application"
            html_body = f"""
            <html>
            <body>
                <p>Dear {applicant_name},</p>
                <p>Your <b>One-Time Password (OTP)</b> for loan application verification is:</p>
                <h2 style='color:blue;'>{otp}</h2>
                <p>Please enter this OTP in the application to proceed.</p>
                <br>
                <p><i>Loan Management Team</i></p>
            </body>
            </html>
            """
        else:
            msg["Subject"] = "Loan Application Submitted Successfully"
            html_body = f"""
            <html>
            <body>
                <p>Dear {applicant_name},</p>
                <p>Your loan application has been submitted successfully.</p>
                <p>We will review your application and notify you once a decision is made.</p>
                <br>
                <p><i>Loan Management Team</i></p>
            </body>
            </html>
            """

        part = MIMEText(html_body, "html")
        msg.attach(part)

        context = ssl.create_default_context()
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls(context=context)
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, applicant_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"Email sending failed: {e}")
        return False

# -------------------------
# 3. Page config
# -------------------------
st.set_page_config(page_title="Loan Prediction App", layout="wide")

# -------------------------
# 4. Load Training Data
# -------------------------
train_csv_path = "loan_dataset_1000.csv"  # Your dataset
if not os.path.exists(train_csv_path):
    st.error(f"{train_csv_path} not found! Place CSV in the same folder.")
    st.stop()
else:
    train_data = pd.read_csv(train_csv_path)

le_gender = LabelEncoder()
train_data['Gender'] = le_gender.fit_transform(train_data['Gender'])
le_emp = LabelEncoder()
train_data['Employment_Status'] = le_emp.fit_transform(train_data['Employment_Status'])
le_loan_type = LabelEncoder()
train_data['Loan_Type'] = le_loan_type.fit_transform(train_data['Loan_Type'])

X = train_data.drop(columns=["Loan_Status"])
y = train_data["Loan_Status"]

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# -------------------------
# 5. Storage for applicant data
# -------------------------
if not os.path.exists("applications.csv"):
    df_apps = pd.DataFrame(columns=[
        "ApplicantID", "Name", "Email", "Mobile", "Address", "Gender", "Income",
        "Credit_Score", "Employment_Status", "Loan_Type", "Loan_Amount",
        "Loan_Term", "Documents", "AI_Probability", "Admin_Decision", "Admin_Comments"
    ])
    df_apps.to_csv("applications.csv", index=False)
else:
    df_apps = pd.read_csv("applications.csv")

# -------------------------
# 6. Helper Functions
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
    expected_features = list(X.columns)
    input_df = pd.DataFrame([input_data])[expected_features]
    prob = model.predict_proba(input_df)[0][1]
    return prob

# -------------------------
# 7. Navigation Sidebar
# -------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Applicant", "Admin"])

# -------------------------
# 8. Applicant Page
# -------------------------
# -------------------------
# 8. Applicant Page (Updated for OTP First)
# -------------------------
if page == "Applicant":
    st.title("💰 Loan Application")

    # Step 1: Enter basic info for OTP
    st.subheader("Step 1: Verify Email")
    name = st.text_input("Full Name")
    email = st.text_input("Email")

    if st.button("Send OTP"):
        if not name or not email:
            st.error("Please enter both name and email to receive OTP.")
        else:
            otp = random.randint(100000, 999999)
            st.session_state["otp"] = otp
            st.session_state["otp_verified"] = False
            st.session_state["applicant_name"] = name
            st.session_state["applicant_email"] = email
            send_email(email, name, otp=otp)
            st.success(f"OTP sent to {email}. Please check your email.")

    # Step 2: Verify OTP
    if "otp" in st.session_state:
        user_otp = st.text_input("Enter OTP sent to your email")
        if st.button("Verify OTP"):
            if str(user_otp) == str(st.session_state["otp"]):
                st.session_state["otp_verified"] = True
                st.success("OTP Verified! You can now fill the application form.")
            else:
                st.error("OTP incorrect. Please try again.")

    # Step 3: Show application form only after OTP verified
    if st.session_state.get("otp_verified", False):
        st.subheader("Step 2: Fill Loan Application Form")

        mobile = st.text_input("Mobile Number")
        address = st.text_area("Address")
        gender = st.radio("Gender", ["Male", "Female"])
        income = st.number_input("Monthly Income", min_value=0)
        credit_score = st.number_input("Credit Score", min_value=0, max_value=1000)
        employment_status = st.radio("Employment Status", ["Employed", "Unemployed"])

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
                "Documents": str(saved_docs),
                "AI_Probability": ai_prob,
                "Admin_Decision": "Pending",
                "Admin_Comments": ""
            }])
            df_apps = pd.concat([df_apps, new_app], ignore_index=True)
            df_apps.to_csv("applications.csv", index=False)
            send_email(email, name)  # Confirmation email
            st.success(f"Application submitted! AI Probability: {ai_prob*100:.2f}%")


# -------------------------
# 9. Admin Page
# -------------------------
elif page == "Admin":
    st.title("🧾 Admin Dashboard")
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
            st.info("No applications submitted yet.")
        else:
            st.subheader("All Applications")
            st.dataframe(df_apps)

            app_ids = df_apps['ApplicantID'].tolist()
            selected_id = st.selectbox("Select Applicant to Review:", app_ids)
            selected_app = df_apps[df_apps['ApplicantID'] == selected_id].iloc[0]

            st.text(f"Name: {selected_app['Name']}")
            st.text(f"Email: {selected_app['Email']}")
            st.text(f"Loan Type: {selected_app['Loan_Type']}")
            st.text(f"AI Probability: {selected_app['AI_Probability']*100:.2f}%")
            st.text(f"Current Decision: {selected_app['Admin_Decision']}")

            try:
                docs_list = eval(selected_app['Documents'])
                if docs_list:
                    for f_path in docs_list:
                        with open(f_path, "rb") as file:
                            st.download_button(
                                label=f"Download {os.path.basename(f_path)}",
                                data=file,
                                file_name=os.path.basename(f_path),
                                mime="application/octet-stream"
                            )
            except:
                st.warning("No documents found.")

            current_decision = selected_app['Admin_Decision']
            decision = st.selectbox(
                "Update Decision:", 
                ["Pending", "Approved", "Rejected"], 
                index=["Pending", "Approved", "Rejected"].index(current_decision)
            )
            comments = st.text_area("Update Comments:", value=selected_app['Admin_Comments'])

            if st.button("Submit Decision"):
                idx = df_apps[df_apps['ApplicantID'] == selected_id].index[0]
                df_apps.at[idx, "Admin_Decision"] = decision
                df_apps.at[idx, "Admin_Comments"] = comments
                df_apps.to_csv("applications.csv", index=False)
                st.success(f"Decision for Applicant {selected_id} updated!")
                send_email(
                    applicant_email=selected_app['Email'],
                    applicant_name=selected_app['Name'],
                    decision=decision,
                    admin_comments=comments
                )
                st.rerun()
