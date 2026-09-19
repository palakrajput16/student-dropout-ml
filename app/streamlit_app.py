
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Student Outcome Predictor",
    page_icon=None,
    layout="wide"
)

# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    * {
        font-family: "Times New Roman", Times, serif !important;
    }

    html, body, [class*="css"] {
        font-family: "Times New Roman", Times, serif !important;
    }

    .stApp {
        background-color: #ffffff;
    }

    h1, h2, h3, h4, h5, h6, p, label, div, span {
        font-family: "Times New Roman", Times, serif !important;
    }

    h1 {
        font-size: 38px !important;
        font-weight: 600 !important;
    }

    h2 {
        font-size: 28px !important;
        font-weight: 600 !important;
    }

    h3 {
        font-size: 22px !important;
        font-weight: 600 !important;
    }

    .subtitle {
        font-size: 18px;
        line-height: 1.6;
        margin-bottom: 24px;
    }

    .result-box {
        padding: 22px;
        border: 1px solid #d9d9d9;
        border-radius: 8px;
        background-color: #fafafa;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .result-label {
        font-size: 17px;
        margin-bottom: 6px;
    }

    .result-value {
        font-size: 30px;
        font-weight: 600;
    }

    .note {
        font-size: 15px;
        line-height: 1.6;
        color: #555555;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid #dddddd;
    }

    button, input, textarea, select {
        font-family: "Times New Roman", Times, serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Data
# ---------------------------------------------------------

@st.cache_data
def load_data():
    enrollment = pd.read_csv("data/processed/enrollment_only.csv")
    semester1 = pd.read_csv("data/processed/semester1.csv")
    return enrollment, semester1


enrollment_df, semester1_df = load_data()

# ---------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------

categorical_cols = [
    "Marital Status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International"
]

numerical_cols = [
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Unemployment rate",
    "Inflation rate",
    "GDP"
]

semester1_numerical_cols = numerical_cols + [
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)"
]


def make_preprocessor(numerical_features):
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_cols
            ),
            (
                "num",
                StandardScaler(),
                numerical_features
            )
        ]
    )


# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

@st.cache_resource
def train_models():
    X_enrollment = enrollment_df.drop(columns=["Target"])
    y_enrollment = enrollment_df["Target"]

    X_semester1 = semester1_df.drop(columns=["Target"])
    y_semester1 = semester1_df["Target"]

    enrollment_model = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(numerical_cols)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=20,
                    min_samples_leaf=1,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    semester1_model = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(semester1_numerical_cols)),
            (
                "model",
                LogisticRegression(
                    C=10.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42
                )
            )
        ]
    )

    enrollment_model.fit(X_enrollment, y_enrollment)
    semester1_model.fit(X_semester1, y_semester1)

    return enrollment_model, semester1_model


enrollment_model, semester1_model = train_models()

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("Student Outcome Predictor")

st.markdown(
    """
    <div class="note">
    <b>Data source:</b> UCI Machine Learning Repository, "Predict Students' Dropout
    and Academic Success" dataset. The dataset contains 4,424 student records from
    a higher education institution in Portugal.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    A machine learning application for predicting a student's recorded academic
    outcome using information available at different stages of their university journey.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="note">
    This application is a project demonstration. Model predictions are statistical
    outputs and should not be treated as definitive statements about an individual student.
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.header("Prediction Stage")

stage = st.sidebar.radio(
    "Choose the information available:",
    [
        "Enrollment-only",
        "After Semester 1"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div class="note">
    <b>Important:</b> This model was trained on students from a higher education
    institution in Portugal. It can be used as a demonstration with students from
    other countries, but its performance has not been validated across countries
    or universities.
    </div>
    <br>
    <div class="note">
    <b>Enrollment-only</b><br>
    Uses information available at enrollment.<br><br>

    <b>After Semester 1</b><br>
    Adds first-semester academic information.
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Human-readable UCI code mappings
# ---------------------------------------------------------

MARITAL_STATUS = {
    1: "Single",
    2: "Married",
    3: "Widower",
    4: "Divorced",
    5: "Facto union",
    6: "Legally separated",
}

APPLICATION_MODE = {
    1: "1st phase - general contingent",
    2: "Ordinance No. 612/93",
    5: "1st phase - special contingent (Azores)",
    7: "Holders of other higher courses",
    10: "Ordinance No. 854-B/99",
    15: "International student (bachelor)",
    16: "1st phase - special contingent (Madeira)",
    17: "2nd phase - general contingent",
    18: "3rd phase - general contingent",
    26: "Ordinance No. 533-A/99, item b2 (Different Plan)",
    27: "Ordinance No. 533-A/99, item b3 (Other Institution)",
    39: "Over 23 years old",
    42: "Transfer",
    43: "Change of course",
    44: "Technological specialization diploma holders",
    51: "Change of institution/course",
    53: "Short cycle diploma holders",
    57: "Change of institution/course (International)",
}

COURSE = {
    33: "Biofuel Production Technologies",
    171: "Animation and Multimedia Design",
    8014: "Social Service (evening attendance)",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management (evening attendance)",
}

PREVIOUS_QUALIFICATION = {
    1: "Secondary education",
    2: "Higher education - bachelor's degree",
    3: "Higher education - degree",
    4: "Higher education - master's degree",
    5: "Higher education - doctorate",
    6: "Frequency of higher education",
    9: "12th year - not completed",
    10: "11th year - not completed",
    12: "Other 11th year",
    14: "10th year",
    15: "10th year - not completed",
    19: "Basic education - 3rd cycle",
    38: "Basic education - 2nd cycle",
    39: "Technological specialization course",
    40: "Higher education degree - 1st cycle",
    42: "Professional higher technical course",
    43: "Higher education master's - 2nd cycle",
}

NATIONALITY = {
    1: "Portuguese",
    2: "German",
    6: "Spanish",
    11: "Italian",
    13: "Dutch",
    14: "English",
    17: "Lithuanian",
    21: "Angolan",
    22: "Cape Verdean",
    24: "Guinean",
    25: "Mozambican",
    26: "Santomean",
    32: "Turkish",
    41: "Brazilian",
    62: "Romanian",
    100: "Moldova (Republic of)",
    101: "Mexican",
    103: "Ukrainian",
    105: "Russian",
    108: "Cuban",
    109: "Colombian",
}

MOTHER_QUALIFICATION = {
    1: "Secondary education - 12th year or equivalent",
    2: "Higher education - bachelor's degree",
    3: "Higher education - degree",
    4: "Higher education - master's",
    5: "Higher education - doctorate",
    6: "Frequency of higher education",
    9: "12th year - not completed",
    10: "11th year - not completed",
    11: "7th year (old)",
    12: "Other - 11th year",
    14: "10th year",
    18: "General commerce course",
    19: "Basic education - 3rd cycle",
    22: "Technical-professional course",
    26: "7th year of schooling",
    27: "2nd cycle of general high school course",
    29: "9th year - not completed",
    30: "8th year of schooling",
    34: "Unknown",
    35: "Cannot read or write",
    36: "Can read without 4th year of schooling",
    37: "Basic education - 1st cycle",
    38: "Basic education - 2nd cycle",
    39: "Technological specialization course",
    40: "Higher education - degree (1st cycle)",
    41: "Specialized higher studies course",
    42: "Professional higher technical course",
    43: "Higher education - master (2nd cycle)",
    44: "Higher education - doctorate (3rd cycle)",
}

FATHER_QUALIFICATION = {
    1: "Secondary education - 12th year or equivalent",
    2: "Higher education - bachelor's degree",
    3: "Higher education - degree",
    4: "Higher education - master's",
    5: "Higher education - doctorate",
    6: "Frequency of higher education",
    9: "12th year - not completed",
    10: "11th year - not completed",
    11: "7th year (old)",
    12: "Other - 11th year",
    13: "2nd year complementary high school course",
    14: "10th year",
    18: "General commerce course",
    19: "Basic education - 3rd cycle",
    20: "Complementary high school course",
    22: "Technical-professional course",
    25: "Complementary high school course - not concluded",
    26: "7th year of schooling",
    27: "2nd cycle of general high school course",
    29: "9th year - not completed",
    30: "8th year of schooling",
    31: "General course of administration and commerce",
    33: "Supplementary accounting and administration",
    34: "Unknown",
    35: "Cannot read or write",
    36: "Can read without 4th year of schooling",
    37: "Basic education - 1st cycle",
    38: "Basic education - 2nd cycle",
    39: "Technological specialization course",
    40: "Higher education - degree (1st cycle)",
    41: "Specialized higher studies course",
    42: "Professional higher technical course",
    43: "Higher education - master (2nd cycle)",
    44: "Higher education - doctorate (3rd cycle)",
}

# The UCI documentation contains additional code categories for some fields.
# Where a complete human-readable mapping is not included in this app, the
# original numeric code is retained rather than guessing at a label.

def code_options(mapping, values):
    return [(mapping.get(int(v), f"Code {int(v)}"), int(v)) for v in values]


def select_code(label, mapping, values, default_value, help_text=None):
    options = code_options(mapping, values)
    labels = [x[0] for x in options]
    default_index = next(i for i, (_, code) in enumerate(options) if code == default_value)
    selected_label = st.selectbox(label, labels, index=default_index, help=help_text)
    return dict(options)[selected_label]


def binary_choice(label, true_label="Yes", false_label="No", default=False, help_text=None):
    choice = st.selectbox(label, [false_label, true_label], index=1 if default else 0, help=help_text)
    return 1 if choice == true_label else 0


# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------

st.header("Student Information")

st.markdown(
    """
    <div class="note">
    The input fields follow the definitions and scales used by the original UCI
    dataset. The app uses human-readable choices where the UCI documentation
    provides a clear code mapping. Numeric values are retained for continuous
    variables such as grades, age, rates, and academic counts.
    </div>
    """,
    unsafe_allow_html=True
)

with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        marital_status = select_code(
            "Marital status", MARITAL_STATUS, sorted(MARITAL_STATUS), 1
        )

        application_mode = select_code(
            "Application mode", APPLICATION_MODE, sorted(APPLICATION_MODE), 1,
            "The student's application/admission route as coded by the UCI dataset."
        )

        application_order = st.selectbox(
            "Application order",
            list(range(0, 10)),
            index=1,
            format_func=lambda x: f"{x + 1}{'st' if x + 1 == 1 else 'nd' if x + 1 == 2 else 'rd' if x + 1 == 3 else 'th'} choice",
            help="The UCI dataset codes the first application choice as 0 and the last as 9."
        )

        course = select_code(
            "Course", COURSE, sorted(COURSE), 9500,
            "Course selected by the student. The labels follow the UCI dataset documentation."
        )

        attendance = st.selectbox(
            "Attendance",
            ["Daytime", "Evening"],
            help="The UCI dataset codes daytime attendance as 1 and evening attendance as 0."
        )
        attendance = 1 if attendance == "Daytime" else 0

        previous_qualification = select_code(
            "Previous qualification", PREVIOUS_QUALIFICATION,
            sorted(PREVIOUS_QUALIFICATION), 1
        )

        nationality = select_code(
            "Nationality", NATIONALITY, sorted(NATIONALITY), 1,
            "These are the nationality categories represented in the original UCI dataset. The model has not been validated for nationalities outside these categories."
        )

        previous_qualification_grade = st.number_input(
            "Previous qualification grade (0-200)",
            min_value=0.0,
            max_value=200.0,
            value=130.0,
            step=0.1,
            help="Grade recorded for the student's previous qualification. The UCI dataset uses a 0 to 200 scale."
        )

    with col2:
        admission_grade = st.number_input(
            "Admission grade (0-200)",
            min_value=0.0,
            max_value=200.0,
            value=125.0,
            step=0.1,
            help="Admission grade recorded in the source dataset. It uses a 0 to 200 scale. For example, 125 means a score of 125 on that scale, not 125 percent."
        )

        age = st.number_input(
            "Age at enrollment (years)",
            min_value=15,
            max_value=80,
            value=20,
            step=1,
            help="Age of the student when they enrolled, measured in years."
        )

        displaced = binary_choice(
            "Displaced", default=False,
            help_text="Whether the student is classified as displaced in the UCI dataset."
        )

        special_needs = binary_choice(
            "Educational special needs", default=False,
            help_text="Whether the student is classified as having special educational needs in the source dataset."
        )

        debtor = binary_choice(
            "Debtor", default=False,
            help_text="Whether the student is classified as a debtor in the source dataset."
        )

        tuition = binary_choice(
            "Tuition fees up to date", true_label="Yes", false_label="No", default=True,
            help_text="Whether the student's tuition fees were up to date at the recorded point in the dataset."
        )

        gender_label = st.selectbox(
            "Gender",
            ["Female", "Male"],
            help="The source dataset records Gender as a binary variable. It does not provide an 'Other' category."
        )
        gender = 0 if gender_label == "Female" else 1

    with col3:
        scholarship = binary_choice(
            "Scholarship holder", default=False,
            help_text="Whether the student is recorded as a scholarship holder."
        )

        international = binary_choice(
            "International student", default=False,
            help_text="Whether the student is recorded as an international student in the source dataset."
        )

        mother_qualification = select_code(
            "Mother's qualification", MOTHER_QUALIFICATION,
            sorted(MOTHER_QUALIFICATION), 1
        )

        father_qualification = select_code(
            "Father's qualification", FATHER_QUALIFICATION,
            sorted(FATHER_QUALIFICATION), 1
        )

        mother_occupation = st.number_input(
            "Mother's occupation code",
            min_value=0,
            value=1,
            step=1,
            help="Occupation is represented by a categorical code in the original UCI dataset."
        )

        father_occupation = st.number_input(
            "Father's occupation code",
            min_value=0,
            value=1,
            step=1,
            help="Occupation is represented by a categorical code in the original UCI dataset."
        )

        unemployment_rate = st.number_input(
            "Unemployment rate (%)",
            min_value=0.0,
            value=11.6,
            step=0.1,
            help="Unemployment rate recorded in the dataset, expressed as a percentage."
        )

        inflation_rate = st.number_input(
            "Inflation rate (%)",
            value=1.2,
            step=0.1,
            help="Inflation rate recorded in the dataset, expressed as a percentage."
        )

        gdp = st.number_input(
            "GDP growth rate (%)",
            value=0.0,
            step=0.1,
            help="GDP growth rate recorded in the dataset, expressed as a percentage."
        )

    semester_values = {}

    if stage == "After Semester 1":

        st.subheader("First-Semester Academic Information")

        st.markdown(
            """
            <div class="note">
            Unit fields are counts of curricular units. Grade is on the source
            dataset's 0 to 20 scale. These variables are available only when the
            prediction stage is set to After Semester 1.
            </div>
            """,
            unsafe_allow_html=True
        )

        s1_col1, s1_col2, s1_col3 = st.columns(3)

        with s1_col1:
            semester_values["Curricular units 1st sem (credited)"] = st.number_input(
                "1st semester credited units (count)",
                min_value=0.0, value=0.0, step=1.0,
                help="Number of curricular units credited during the first semester."
            )

            semester_values["Curricular units 1st sem (enrolled)"] = st.number_input(
                "1st semester enrolled units (count)",
                min_value=0.0, value=6.0, step=1.0,
                help="Number of curricular units in which the student was enrolled during the first semester."
            )

        with s1_col2:
            semester_values["Curricular units 1st sem (evaluations)"] = st.number_input(
                "1st semester evaluations (count)",
                min_value=0.0, value=8.0, step=1.0,
                help="Number of curricular units with an evaluation recorded during the first semester."
            )

            semester_values["Curricular units 1st sem (approved)"] = st.number_input(
                "1st semester approved units (count)",
                min_value=0.0, value=5.0, step=1.0,
                help="Number of curricular units approved during the first semester."
            )

        with s1_col3:
            semester_values["Curricular units 1st sem (grade)"] = st.number_input(
                "1st semester grade (0-20)",
                min_value=0.0, max_value=20.0, value=12.0, step=0.1,
                help="Average first-semester grade recorded in the source dataset. It uses a 0 to 20 scale."
            )

            semester_values["Curricular units 1st sem (without evaluations)"] = st.number_input(
                "1st semester units without evaluations (count)",
                min_value=0.0, value=0.0, step=1.0,
                help="Number of first-semester curricular units without a recorded evaluation."
            )

    submitted = st.form_submit_button("Predict Outcome")

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if submitted:

    input_data = {
        "Marital Status": marital_status,
        "Application mode": application_mode,
        "Application order": application_order,
        "Course": course,
        "Daytime/evening attendance": attendance,
        "Previous qualification": previous_qualification,
        "Nacionality": nationality,
        "Previous qualification (grade)": previous_qualification_grade,
        "Mother's qualification": mother_qualification,
        "Father's qualification": father_qualification,
        "Mother's occupation": mother_occupation,
        "Father's occupation": father_occupation,
        "Admission grade": admission_grade,
        "Displaced": displaced,
        "Educational special needs": special_needs,
        "Debtor": debtor,
        "Tuition fees up to date": tuition,
        "Gender": gender,
        "Scholarship holder": scholarship,
        "Age at enrollment": age,
        "International": international,
        "Unemployment rate": unemployment_rate,
        "Inflation rate": inflation_rate,
        "GDP": gdp
    }

    input_data.update(semester_values)

    input_df = pd.DataFrame([input_data])

    if stage == "Enrollment-only":
        model = enrollment_model
        feature_columns = enrollment_df.drop(columns=["Target"]).columns
    else:
        model = semester1_model
        feature_columns = semester1_df.drop(columns=["Target"]).columns

    input_df = input_df[feature_columns]

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    class_names = model.named_steps["model"].classes_

    probability_df = pd.DataFrame({
        "Outcome": class_names,
        "Probability": probabilities
    })

    st.header("Prediction")

    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-label">Predicted recorded outcome</div>
            <div class="result-value">{prediction}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Model probabilities")

    display_df = probability_df.copy()
    display_df["Probability"] = (
        display_df["Probability"] * 100
    ).round(2).astype(str) + "%"

    st.table(display_df)

    st.markdown(
        """
        <div class="note">
        The probabilities represent the model's estimated class probabilities
        for the supplied input. They should not be interpreted as a student's
        personal certainty of dropping out, remaining enrolled, or graduating.
        </div>
        """,
        unsafe_allow_html=True
    )
