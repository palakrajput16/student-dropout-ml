# Student Dropout Prediction

A machine learning project exploring whether a student's eventual academic outcome can be predicted from information available at different stages of their university journey.


This is one of my first end-to-end machine learning projects. I wanted to do more than train a model and report an accuracy score, so I worked through the project from data cleaning and exploratory analysis to model tuning, feature interpretation, and final evaluation.

The project uses the **UCI Predict Students' Dropout and Academic Success** dataset, which contains records for 4,424 students.

---
## Live Demo

[**Try the Streamlit App**](https://student-dropout-prediction-ml0.streamlit.app/)

The interactive application allows users to enter student information and receive a predicted outcome with the model's probability distribution.

---

## The question I wanted to explore

A simple question motivated the project:

> **How much can we predict about a student's eventual outcome using the information available at different points in time?**

The target has three possible outcomes:

- **Dropout**
- **Enrolled**
- **Graduate**

An important part of the project is that I did not want to use information from the future when making an earlier prediction.

Because of that, I created two prediction stages:

### 1. Enrollment-only

Only information that could be available when a student enrolls is used.

### 2. Semester 1

Enrollment information is combined with first-semester academic information.

This lets me compare what prediction looks like early in a student's journey with what changes once their first semester has happened.

---

## Dataset

The dataset comes from the UCI Machine Learning Repository:

**Predict Students' Dropout and Academic Success**

- Students: **4,424**
- Predictors: **36**
- Target: **Target**
- Target classes: Dropout, Enrolled, Graduate
- Missing values: **0**
- Duplicate rows: **0**

The target distribution is:

| Outcome | Students | Percentage |
|---|---:|---:|
| Graduate | 2,209 | 49.93% |
| Dropout | 1,421 | 32.12% |
| Enrolled | 794 | 17.95% |

The classes are therefore not evenly distributed. Because of this, I use **Macro F1** alongside accuracy instead of relying on accuracy alone.

### Source

[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

DOI: [10.24432/C5MC89](https://doi.org/10.24432/C5MC89)

---

## What I did

I broke the project into several stages.

### 1. Data loading and cleaning

I first checked the structure and quality of the dataset.

The dataset had:

- 4,424 rows
- 37 columns including the target
- no missing values
- no duplicate rows

I also examined the data types, unique values, target distribution, and numerical summaries.

### 2. Exploratory Data Analysis

I looked at how different variables were related to the three outcomes.

Some of the patterns I found were:

- Tuition fee status showed a strong association with the observed outcome.
- Scholarship status also showed a noticeable difference in outcome distributions.
- Students recorded as debtors had a higher observed dropout rate than students who were not recorded as debtors.
- First-semester academic performance showed clear differences across the three outcome groups.
- Age at enrollment also differed between outcome groups.
- Some variables, such as International status, showed much weaker evidence of association in the analysis.

These are observations from this dataset. They should not be interpreted as evidence that any individual variable causes a student to drop out.

I also kept sample sizes in mind when looking at course-level dropout rates. For example, a very high rate in a course with only a small number of students is less stable than a similar rate in a much larger group.

### 3. Feature sets

I created two processed datasets:

```text
data/processed/enrollment_only.csv
data/processed/semester1.csv
```

The first contains information available at enrollment.

The second adds first-semester academic variables.

Second-semester information was excluded from these early prediction stages because it would not be available at the corresponding prediction point.

### 4. Preprocessing

The modeling pipeline handles categorical and numerical variables separately.

Categorical variables are one-hot encoded, while numerical variables are standardized.

I also used a stratified train-test split so that the outcome proportions were preserved between the training and test sets.

### 5. Baseline models

I started with three relatively standard classification models:

- Logistic Regression
- Decision Tree
- Random Forest

This gave me a reference point before tuning the models.

### 6. Hyperparameter tuning

I used 5-fold stratified cross-validation and GridSearchCV.

The main tuning metric was **Macro F1** because the project contains three classes with different frequencies.

Class weighting was also included in the search.

### 7. Feature interpretation

I looked at:

- Logistic Regression coefficients
- Decision Tree feature importance
- Random Forest feature importance
- Aggregated importance at the original variable level

One thing that stood out was the role of first-semester academic variables once they became available. Variables such as first-semester approved units, grades, and evaluations became important in the Semester 1 models.

Again, feature importance describes how the fitted model uses the variables. It does not mean that a variable is a direct cause of the outcome.

---

## Final model results

The final held-out test results were:

| Prediction Stage | Model | Accuracy | Macro F1 |
|---|---|---:|---:|
| Enrollment-only | Logistic Regression | 56.27% | 0.5359 |
| Enrollment-only | Decision Tree | 51.07% | 0.5033 |
| Enrollment-only | Random Forest | 58.64% | 0.5432 |
| Semester 1 | Logistic Regression | 71.07% | 0.6732 |
| Semester 1 | Decision Tree | 60.56% | 0.5905 |
| Semester 1 | Random Forest | 70.96% | 0.6521 |

One of the clearest patterns in the project is the change between the two prediction stages.

Adding first-semester information increased Macro F1 for all three models:

| Model | Change in Macro F1 |
|---|---:|
| Logistic Regression | +0.1373 |
| Decision Tree | +0.0871 |
| Random Forest | +0.1089 |

This suggests that first-semester academic information contains useful predictive information for distinguishing the three student outcomes in this dataset.

It also highlights an important practical trade-off: earlier predictions have less information available, while later predictions can use information about how a student performed during their first semester.

---

## Looking at individual classes

The three-class problem is not equally easy across all outcomes.

For the Semester 1 Logistic Regression model, the test-set results were:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Dropout | 0.8143 | 0.6796 | 0.7409 |
| Enrolled | 0.3975 | 0.5975 | 0.4774 |
| Graduate | 0.8337 | 0.7715 | 0.8014 |

The **Enrolled** class was harder to identify than Dropout and Graduate.

This is one reason I chose Macro F1 as the main metric rather than looking only at overall accuracy.

---

## What I learned from the project

The biggest thing I took away from this project is that building a model is only one part of machine learning.

The prediction stage matters.

If I only look at the final model score, I miss an important part of the problem. A model using first-semester grades is answering a different practical question from a model that has to make a prediction at enrollment.

I also learned that:

- Accuracy can hide differences between classes.
- Cross-validation is useful when tuning models.
- Feature importance needs careful interpretation.
- Association does not mean causation.
- Small groups can produce unstable percentages.
- A model should be evaluated on data that was not used to fit it.

---

## Limitations

There are several limitations to this project.

### Dataset limitations

The analysis is based on one dataset containing 4,424 student records. The results should not automatically be generalized to every university or student population.

### Prediction limitations

The model predicts the three recorded outcomes in the dataset. It does not explain why an individual student reaches a particular outcome.

### Feature interpretation

Feature importance and model coefficients describe relationships used by the fitted models. They are not causal estimates.

### Class imbalance

The target classes are not equally represented, and the Enrolled class is smaller than the other two classes. This makes class-level metrics particularly important.

### Small subgroup sizes

Some course-level groups are relatively small. I therefore avoided treating a high observed dropout percentage in a tiny group as strong evidence by itself.

---

## Future work

If I continue developing this project, I would like to explore:

- SHAP-based model explanations
- Probability calibration
- More careful error analysis
- Threshold-based dropout risk identification
- A simple dashboard for exploring predictions
- External validation on another student dataset
- A more explicit early-warning framework using only information available at each intervention point

I would also like to investigate whether the model's probability estimates are reliable enough to be useful for an actual risk-screening workflow.

---

## Project structure

```text
student-droupout-ml/
│
├── data/
│   ├── raw/
│   │   └── student_dropout.csv
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
├── .gitignore
└── README.md
```

---

## Running the project

Clone the repository and create a virtual environment:

```powershell
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

The notebooks can then be opened in VS Code or Jupyter and run from top to bottom.

---

## Tools

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter

---

## A note on the project

I built this project as a way to understand the full machine learning workflow rather than just practice individual algorithms.

The most useful part for me was comparing the two prediction stages. It made the project feel less like a model comparison exercise and more like an actual question about when information becomes available and what that means for prediction.

There is still a lot I can improve, but this gives me a solid base to build on.
