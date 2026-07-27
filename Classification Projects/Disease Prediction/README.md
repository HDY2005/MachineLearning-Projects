# 🩺 Disease Prediction using Machine Learning

A Machine Learning project that predicts the most likely disease based on the symptoms provided by the user. This project demonstrates the complete Machine Learning workflow—from data preprocessing and exploratory data analysis (EDA) to model training, evaluation, and prediction.

---

## 📌 Project Overview

Healthcare professionals often rely on a patient's symptoms to narrow down possible diseases. This project automates that process using Machine Learning models trained on symptom data.

The notebook includes:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Data visualization
- Feature encoding
- Model training
- Model evaluation
- Disease prediction on new symptom inputs
- Saving the trained model for future use

---

## 🎯 Objectives

- Analyze relationships between symptoms and diseases.
- Compare multiple Machine Learning algorithms.
- Build an accurate disease prediction model.
- Create a reusable prediction pipeline.

---

## 📂 Project Structure

```
Disease-Prediction/
│
├── Disease_Predictor.ipynb      # Main Jupyter Notebook
├── dataset.csv                  # Dataset (if included)
├── model.pkl                    # Saved trained model
└── README.md
```

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Pickle

---

## 📊 Data Preprocessing

The following preprocessing steps were performed:

- Loaded the dataset using Pandas.
- Checked dataset shape and information.
- Checked for missing values.
- Removed unnecessary characters using Regex.
- Label Encoded disease names.
- Split data into training and testing sets.

**Note:** Duplicate rows were intentionally kept because removing them reduced the model's performance and resulted in an unrealistic class distribution.

---

## 📈 Exploratory Data Analysis (EDA)

Several visualizations were created to better understand the dataset, including:

- Disease frequency distribution
- Correlation heatmap
- Top 20 most common symptoms

These visualizations help identify symptom importance and disease distribution.

---

## 🤖 Machine Learning Models

The following models were trained and evaluated:

- Decision Tree Classifier
- Random Forest Classifier

The dataset was divided into training and testing sets before model training.

---

## 📏 Model Evaluation

The models were evaluated using:

- Accuracy Score
- Classification Report
- Cross Validation

After comparison, the **Random Forest Classifier** produced the best performance and was selected as the final model.

---

## 💡 Example Prediction

Input Symptoms:

```
itching
skin_rash
nodal_skin_eruptions
continuous_sneezing
```

Predicted Disease:

```
Fungal Infection
```

*(Example prediction only.)*

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Disease-Prediction.git
```

### 2. Move into the project directory

```bash
cd Disease-Prediction
```

### 3. Install dependencies


### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```
Disease_Predictor.ipynb
```

Run all cells sequentially.

---

## 📦 Required Libraries

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Or install manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 📁 Model Saving

The trained model can be saved using:

```python
import pickle

pickle.dump(model, open("model.pkl", "wb"))
```

Load the model later:

```python
import pickle

model = pickle.load(open("model.pkl", "rb"))
```

---

## 🔮 Future Improvements

- Build a Streamlit web application
- Create a Flask/Django API
- Add symptom autocomplete
- Improve feature engineering
- Train additional ML models (XGBoost, LightGBM, CatBoost)
- Add confidence scores for predictions

---

## 📚 Learning Outcomes

This project demonstrates:

- Data Cleaning
- Exploratory Data Analysis
- Data Visualization
- Feature Engineering
- Label Encoding
- Machine Learning
- Model Evaluation
- Cross Validation
- Model Serialization
- Predictive Analytics

---

## 👨‍💻 Author

**Harsh Deep Yadav**

B.Tech Artificial Intelligence & Machine Learning

---

## ⭐ If you found this project useful, consider giving it a Star!