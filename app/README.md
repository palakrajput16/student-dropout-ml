# Student Dropout Prediction

A machine learning project that predicts whether a student is likely to **Dropout**, remain **Enrolled**, or **Graduate**.

The project focuses on an important question:

> **How does the information available at different stages of a student's academic journey affect the prediction of their final outcome?**

Instead of using only one prediction setup, I compare two stages:

1. **Enrollment-only:** information available when the student enters the institution.
2. **After Semester 1:** enrollment information plus first-semester academic performance.

This lets me study how much predictive information becomes available after the first semester while avoiding the use of second-semester information for an earlier prediction.

---

## Project Overview

The target variable has three classes:

- `Dropout`
- `Enrolled`
- `Graduate`

I use Logistic Regression, Decision Tree, and Random Forest models and evaluate them using accuracy, precision, recall, F1-score, Macro F1, and confusion matrices.

The project also includes hyperparameter tuning and feature interpretation.

### Main question

**Can student outcome be predicted using information available at enrollment, and how much does prediction improve when first-semester performance is added?**

---

## Dataset

The dataset is:

**Predict Students' Dropout and Academic Success**

Source: **UCI Machine Learning Repository**

Dataset ID: 697  
DOI: `10.24432/C5MC89`

The dataset contains:

- 4,424 student records
- 36 predictor variables
- 1 target variable
- 3 outcome classes
- No missing values
- No duplicate records

The data comes from a higher education institution in **Portugal**.

The original dataset includes information available at enrollment, first-semester performance, and second-semester performance.

I use the dataset according to its original three-class classification problem.

---

## Target Distribution

| Outcome | Count | Percentage |
|---|---:|---:|
| Graduate | 2,209 | 49.93% |
| Dropout | 1,421 | 32.12% |
| Enrolled | 794 | 17.95% |

The classes are therefore not perfectly balanced.

Because of this, I do not rely on accuracy alone. Macro F1 and class-level recall are particularly useful for understanding model performance across all three outcomes.

---

## Exploratory Data Analysis

The EDA examined:

- Target distribution
- Age
- Gender
- Scholarship status
- Tuition fee status
- Course
- Admission grade
- Previous qualification grade
- First-semester academic performance
- Second-semester academic performance
- Debtor status
- Displacement status
- Educational special needs
- International status
- Unemployment rate
- Inflation
- GDP
- Correlations
- Categorical associations using chi-square tests

### Some observations

Scholarship and tuition status showed substantial differences in observed outcome distributions.

For example, among students without scholarships:

- Dropout: 38.71%
- Enrolled: 19.97%
- Graduate: 41.32%

Among scholarship holders:

- Dropout: 12.19%
- Enrolled: 11.83%
- Graduate: 75.98%

Tuition status also showed a strong association with the target outcome.

These findings describe **associations in the dataset**. They should not be interpreted as evidence that a particular factor causes a student's outcome.

Small groups also need caution. For example, some courses have very few observations, so their observed dropout percentages can be unstable.

---

## Prediction Stages

I created two separate feature sets.

### 1. Enrollment-only

This feature set contains information that would be available around the time of enrollment.

It excludes first-semester and second-semester academic information.

```text
Enrollment information
        |
        v
   ML classifier
        |
        v
Dropout / Enrolled / Graduate
```

### 2. After Semester 1

This feature set contains enrollment information plus first-semester academic performance.

Second-semester information is excluded.

```text
Enrollment information
        +
First-semester performance
        |
        v
   ML classifier
        |
        v
Dropout / Enrolled / Graduate
```

This separation is important because using later academic information to make an earlier prediction would introduce information leakage.

---

## Preprocessing

The preprocessing pipeline includes:

- Train/test split
- Stratification by target class
- Categorical feature encoding
- Numerical feature scaling where required
- Model-specific preprocessing
- Class balancing during model tuning

The train/test split uses 80% of the data for training and 20% for the held-out test set.

The training set contains 3,539 students and the test set contains 885 students.

---

## Models

I compared three classification algorithms:

### Logistic Regression

A linear classification model that provides useful coefficient-based interpretation.

### Decision Tree

A tree-based model that can represent nonlinear decision rules.

### Random Forest

An ensemble of decision trees that can capture nonlinear relationships and interactions between variables.

---

## Baseline Results

The initial baseline results were:

| Prediction Stage | Model | Accuracy | Macro F1 |
|---|---|---:|---:|
| Enrollment-only | Logistic Regression | 62.37% | 0.5060 |
| Enrollment-only | Decision Tree | 52.20% | 0.4666 |
| Enrollment-only | Random Forest | 62.15% | 0.4848 |
| Semester 1 | Logistic Regression | 73.79% | 0.6465 |
| Semester 1 | Decision Tree | 66.10% | 0.5999 |
| Semester 1 | Random Forest | 71.86% | 0.5925 |

The Semester 1 models generally perform better than the enrollment-only models, showing that first-semester academic information provides additional predictive information.

---

## Hyperparameter Tuning

I used cross-validation and hyperparameter search with **Macro F1** as the main tuning objective.

Class weighting was selected as part of the tuning process for all six model-stage combinations.

The tuned results were:

| Prediction Stage | Model | Baseline Macro F1 | Tuned Macro F1 | Change |
|---|---|---:|---:|---:|
| Enrollment-only | Logistic Regression | 0.5060 | 0.5359 | +0.0300 |
| Enrollment-only | Decision Tree | 0.4666 | 0.5033 | +0.0368 |
| Enrollment-only | Random Forest | 0.4848 | 0.5432 | +0.0585 |
| Semester 1 | Logistic Regression | 0.6465 | 0.6732 | +0.0268 |
| Semester 1 | Decision Tree | 0.5999 | 0.5905 | -0.0094 |
| Semester 1 | Random Forest | 0.5925 | 0.6521 | +0.0597 |

Tuning did not improve every model. This is useful because it shows that hyperparameter tuning is not guaranteed to improve a model.

---

## Final Model Comparison

For the final comparison, I used the tuned models selected during the project evaluation.

| Prediction Stage | Model | Accuracy | Macro F1 |
|---|---|---:|---:|
| Enrollment-only | Logistic Regression | 56.27% | 0.5359 |
| Enrollment-only | Decision Tree | 51.07% | 0.5033 |
| Enrollment-only | Random Forest | 58.64% | 0.5432 |
| Semester 1 | Logistic Regression | 71.07% | 0.6732 |
| Semester 1 | Decision Tree | 60.56% | 0.5905 |
| Semester 1 | Random Forest | 70.96% | 0.6521 |

The difference between enrollment-only and Semester 1 Macro F1 was:

| Model | Change in Macro F1 |
|---|---:|
| Logistic Regression | +0.1373 |
| Decision Tree | +0.0871 |
| Random Forest | +0.1089 |

The results show how much predictive performance changes when first-semester academic information becomes available.

---

## Final Semester 1 Logistic Regression

The tuned Semester 1 Logistic Regression model had the following held-out test performance:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Dropout | 0.8143 | 0.6796 | 0.7409 |
| Enrolled | 0.3975 | 0.5975 | 0.4774 |
| Graduate | 0.8337 | 0.7715 | 0.8014 |

The `Enrolled` class is more difficult for the model to identify than the other two classes.

This is one reason Macro F1 is useful: it prevents the overall result from being dominated by the larger or easier classes.

---

## Cross-Validation vs Held-Out Test

For the two stronger Semester 1 tuned models:

| Model | CV Macro F1 | Test Macro F1 |
|---|---:|---:|
| Logistic Regression | 0.6770 | 0.6732 |
| Random Forest | 0.6856 | 0.6521 |

The held-out test performance is reasonably close to cross-validation performance for Logistic Regression, while the Random Forest shows a larger difference.

---

## Feature Interpretation

I also examined which variables were influential to the trained models.

For the enrollment-only Random Forest, important variables included:

- Tuition fee status
- Scholarship status
- Age at enrollment
- Admission grade
- Previous qualification grade

After adding first-semester information, academic variables became much more prominent.

Important Semester 1 Random Forest variables included:

- First-semester approved units
- First-semester grade
- First-semester evaluations

The model also used variables such as course, parental education or occupation variables, tuition status, admission grade, and application information.

The Logistic Regression coefficients also showed substantial contributions from first-semester academic variables.

### Important interpretation rule

Feature importance and model coefficients describe how the model uses variables.

They do **not** prove that a variable causes dropout or graduation.

For example, if a feature is important to the model, that does not mean changing that feature would necessarily change a student's outcome.

---

## Interactive Streamlit App

The project also includes a Streamlit application that allows a user to enter student information and receive a predicted outcome.

The app supports two prediction stages:

### Enrollment-only

Uses the tuned Random Forest configuration from the project evaluation.

Reference held-out performance:

- Accuracy: 58.64%
- Macro F1: 0.5432

### After Semester 1

Uses the tuned Logistic Regression configuration from the project evaluation.

Reference held-out performance:

- Accuracy: 71.07%
- Macro F1: 0.6732

The application refits the selected model on the complete processed dataset before making interactive predictions.

The app also displays model probabilities for the three possible outcomes.

---

## About the Input Values

The Streamlit interface uses human-readable choices wherever the UCI documentation provides a clear category mapping.

Examples include:

- Female / Male
- Daytime / Evening
- Course names
- Previous qualification
- Nationality
- Application mode
- Marital status
- Scholarship status
- Tuition fee status

Continuous variables retain their original dataset scales.

For example:

### Admission grade

The original dataset uses a **0 to 200** scale.

Therefore:

```text
Admission grade = 125
```

means a value of 125 on the dataset's 0 to 200 admission-grade scale.

Similarly, first-semester grades use the original dataset's **0 to 20** scale.

The app includes help text explaining the units and scales rather than expecting the user to understand the raw dataset codes.

---

## Can Students From Other Countries Use the App?

The underlying dataset comes from a higher education institution in Portugal.

The application can technically accept inputs representing students from outside Portugal, but the model's performance has **not been validated across different countries or universities**.

The nationality categories are also based on the categories represented in the original dataset.

Therefore, the application should be treated as a machine learning demonstration rather than a universally validated student-risk system.

Testing and validating the model on data from other institutions and countries would be necessary before making claims about its generalization.

---

## Limitations

### 1. Single-institution dataset

The dataset comes from one higher education institution in Portugal.

This limits how confidently the results can be generalized to other universities.

### 2. Generalization across countries

The model has not been validated on students from other countries.

Differences in education systems, admission practices, grading systems, financial conditions, and student populations could affect performance.

### 3. Association is not causation

The model identifies statistical patterns.

It does not establish causal relationships.

### 4. Small subgroups

Some categories contain relatively few students.

Percentages calculated from very small groups should therefore be interpreted cautiously.

### 5. Model probabilities are not automatically calibrated risk

The probabilities shown by the application are outputs from the fitted classifier.

They should not automatically be interpreted as precise individual probabilities of dropout, enrollment, or graduation.

### 6. Prediction is not a final judgment

A model prediction should not be treated as a definitive statement about an individual student's future.

---

## What I Learned

Through this project, I worked through a complete machine learning workflow:

### Data Analytics

- Loading datasets
- Data quality checks
- Missing-value analysis
- Duplicate detection
- Grouped analysis
- Cross-tabulation
- Correlation analysis
- Statistical association testing
- Data visualization

### Machine Learning

- Multiclass classification
- Train/test splitting
- Categorical encoding
- Feature preprocessing
- Logistic Regression
- Decision Trees
- Random Forest
- Class weighting
- Cross-validation
- Hyperparameter tuning

### Model Evaluation

- Accuracy
- Precision
- Recall
- F1-score
- Macro F1
- Confusion matrices
- Class imbalance
- Cross-validation vs held-out evaluation

### Model Interpretation

- Feature importance
- Logistic Regression coefficients
- Understanding model behavior
- Separating association from causation

---

## Project Structure

```text
student-droupout-ml/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── enrollment_only.csv
│       ├── semester1.csv
│       └── final_model_comparison.csv
│
├── notebooks/
│   ├── 01_data_loading_and_cleaning.ipynb
│   ├── 01_student_dropout_eda.ipynb
│   ├── 02_student_dropout_preprocessing.ipynb
│   ├── 03_student_dropout_baseline_models.ipynb
│   ├── 04_student_dropout_model_evaluation_tuning.ipynb
│   ├── 05_student_dropout_feature_interpretation.ipynb
│   └── 06_student_dropout_final_evaluation.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## How to Run the Project

Clone the repository and move into the project directory.

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

To run the notebooks:

```powershell
jupyter notebook
```

To run the Streamlit application:

```powershell
streamlit run app/streamlit_app.py
```

The application expects the processed datasets to be available in:

```text
data/processed/enrollment_only.csv
data/processed/semester1.csv
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn
- Jupyter Notebook
- Streamlit
- Git
- GitHub

---

## Future Work

Possible extensions include:

- External validation using data from another university
- Validation using data from another country
- Probability calibration
- More detailed error analysis
- Additional classification models
- Explainable AI methods such as SHAP
- Temporal validation
- Fairness analysis across demographic groups
- A larger interactive dashboard
- Model monitoring after deployment

---

## Dataset Citation

The dataset used in this project is:

> Realinho, V., Machado, J., Oliveira, J. L., &amp; Santos, R. (2021). Predict Students' Dropout and Academic Success. UCI Machine Learning Repository. DOI: 10.24432/C5MC89.

Dataset license: **CC BY 4.0**

---

## Final Note

This project is primarily a learning and portfolio project focused on understanding the complete machine learning workflow from raw data to an interactive prediction application.

The predictions should be interpreted as model outputs, not as definitive judgments about individual students.
