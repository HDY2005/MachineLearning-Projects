# Employee Attrition Prediction

## 📌 Project Overview

Employee attrition — when employees leave an organization — is costly and disruptive, affecting productivity, team morale, and recruitment/training expenses. This project builds a machine learning pipeline to predict whether an employee is likely to leave a company based on demographic, job-related, and workplace-satisfaction attributes. Such a model can help HR teams proactively identify at-risk employees and design targeted retention strategies.

## 🎯 Objective

The goal is to build a binary classification model that predicts employee **Attrition** (`Yes`/`No`) using historical HR data, and to evaluate different modeling strategies for handling the class imbalance inherent in attrition data (most employees stay, a minority leave).

## 📊 Dataset

The dataset contains **1,470 employee records** with **35 attributes**, including the target variable `Attrition`. There are no missing values or duplicate rows in the data.

Key feature categories include:

| Category | Example Features |
|---|---|
| **Demographics** | Age, Gender, Marital Status, Education, Education Field |
| **Job Information** | Department, Job Role, Job Level, Business Travel |
| **Compensation** | Monthly Income, Daily Rate, Hourly Rate, Monthly Rate, Percent Salary Hike, Stock Option Level |
| **Work Environment** | Distance From Home, OverTime, Environment Satisfaction |
| **Job Satisfaction** | Job Satisfaction, Job Involvement, Relationship Satisfaction |
| **Work-Life Factors** | Work-Life Balance |
| **Career Attributes** | Total Working Years, Years At Company, Years In Current Role, Years Since Last Promotion, Years With Current Manager, Number of Companies Worked, Training Times Last Year, Performance Rating |

The target variable, `Attrition`, is mapped to a binary format (`1` = employee left, `0` = employee stayed) for modeling.

## 🔍 Exploratory Data Analysis

The EDA focused on understanding how key factors relate to attrition:

- **Age distribution & attrition by age group** — employee ages were binned into groups to visualize how attrition varies across career stages.
- **Monthly income by age and gender** — average income trends were compared across male and female employees as they age.
- **OverTime vs. Attrition** — attrition rates were compared between employees who regularly work overtime and those who do not, highlighting overtime as a notable behavioral factor.

## ⚙️ Data Preprocessing

The following preprocessing steps were applied before modeling:

- **Constant-value columns removed:** `EmployeeCount`, `Over18`, and `StandardHours` were dropped since they contained a single unique value across all records and carry no predictive signal.
- **Target encoding:** `Attrition` was mapped from `Yes`/`No` to `1`/`0`.
- **Categorical encoding:** Categorical columns (`BusinessTravel`, `Department`, `EducationField`, `Gender`, `JobRole`, `MaritalStatus`, `OverTime`) were transformed using `LabelEncoder`.
- **Feature scaling:** `StandardScaler` was applied within the Logistic Regression pipeline.
- **Train/test split:** Data was split into training (80%) and test (20%) sets using stratified sampling (`stratify=Y`) to preserve the class balance of the target variable, with `random_state=123` for reproducibility.
- **Class imbalance handling:** Given that far fewer employees leave than stay, two complementary techniques were used:
  - **SMOTE** (Synthetic Minority Over-sampling Technique) to oversample the minority class in the training data.
  - **Class weighting** via XGBoost's `scale_pos_weight` parameter, computed from the training set's class ratio.

## 🤖 Machine Learning Models

Several classification models were evaluated:

- **Logistic Regression** — a scaled, regularized baseline model.
- **Random Forest Classifier** — an ensemble tree-based baseline.
- **Gradient Boosting Classifier** — evaluated as a baseline and then optimized via hyperparameter tuning.
- **XGBoost Classifier** — evaluated with both SMOTE-balanced training data and class-weighted training, followed by decision-threshold tuning.

## 🔧 Hyperparameter Tuning

`RandomizedSearchCV` was used to tune tree-based ensemble models:

- **Gradient Boosting:** tuned over `n_estimators`, `learning_rate`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `subsample`, and `max_features` (50 iterations, 5-fold CV, optimizing accuracy). The best configuration found was:
  - `n_estimators=300`, `learning_rate=0.05`, `max_depth=3`, `min_samples_split=10`, `min_samples_leaf=1`, `max_features='sqrt'`, `subsample=1.0`
  - Best cross-validated accuracy: **0.8724**

- **XGBoost:** tuned over `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, and `gamma` using cross-validated randomized search.

In addition to hyperparameter search, **decision-threshold tuning** was performed on the class-weighted XGBoost model's predicted probabilities (testing thresholds from 0.25 to 0.5) to balance precision and recall for the minority (attrition) class.

## 📈 Model Evaluation

Models were evaluated using **accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices**, with particular attention to performance on the minority "Attrition = Yes" class, since this is the class of practical interest for HR intervention.

**Tuned Gradient Boosting** (test set):

| Metric | Class 0 (Stayed) | Class 1 (Left) |
|---|---|---|
| Precision | 0.89 | 0.71 |
| Recall | 0.97 | 0.36 |
| F1-score | 0.93 | 0.48 |

- Accuracy: **0.8741**
- ROC-AUC: **0.8329**

**Class-weighted XGBoost with threshold tuning** (test set, decision threshold = 0.40):

| Metric | Class 0 (Stayed) | Class 1 (Left) |
|---|---|---|
| Precision | 0.91 | 0.65 |
| Recall | 0.95 | 0.51 |
| F1-score | 0.93 | 0.57 |

- Accuracy: **0.88**

Adjusting the decision threshold allowed a better trade-off between precision and recall on the minority class compared to the default 0.5 threshold, meaningfully improving the model's ability to identify employees likely to leave.

## 🏆 Final Model

The **final model is an XGBoost Classifier with class weighting (`scale_pos_weight`) and a tuned decision threshold**. While the hyperparameter-tuned Gradient Boosting model achieved slightly higher overall accuracy, the weighted XGBoost model with threshold tuning provided a substantially better **recall and F1-score for the minority "Attrition" class** — the outcome that matters most for a real-world retention use case, where failing to flag an at-risk employee is more costly than a false positive.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)

## 📁 Project Structure

```text
Employee-Attrition-Prediction/
│
├── data/
│   └── data.csv
├── notebooks/
│   └── Attrition.ipynb
├── README.md
└── requirements.txt
```

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/Employee-Attrition-Prediction.git
   cd Employee-Attrition-Prediction
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the notebook**
   ```bash
   jupyter notebook notebooks/Attrition.ipynb
   ```

## 🔮 Future Improvements

- More advanced feature engineering (e.g., interaction terms, tenure ratios)
- Exploring additional models such as LightGBM or CatBoost
- More exhaustive hyperparameter optimization (e.g., Bayesian optimization)
- Model explainability using SHAP or feature importance analysis
- Deployment as an API or interactive dashboard for HR teams
- Further refinement of the classification decision threshold based on business cost trade-offs

## 👨‍💻 Author

**Harsh Deep Yadav**
GitHub: [https://github.com/HDY2005](#)
LinkedIn: [https://www.linkedin.com/in/harsh-deep-yadav-295ba2348/](#)