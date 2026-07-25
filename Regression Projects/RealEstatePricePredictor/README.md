# 🏠 House Price Prediction using Machine Learning

## 1. 📌 Project Overview

This project builds a machine learning regression pipeline that predicts house prices based on structural and locational features such as area, number of bedrooms/bathrooms, amenities, and neighborhood preference. House price prediction is a classic and valuable machine learning problem because real estate pricing depends on many interacting, non-obvious factors — a data-driven model helps buyers, sellers, and agents estimate fair market value more objectively and consistently than manual estimation alone.

## 2. 🎯 Problem Statement

Manually pricing a house requires weighing dozens of factors (size, amenities, location desirability, furnishing status, etc.) at once, which is difficult to do consistently and is prone to human bias. Machine learning addresses this by learning the relationship between these features and historical sale prices, enabling data-driven, repeatable price estimates for new properties.

## 3. 📊 Dataset

| Detail | Value |
|---|---|
| **Source** | `Housing.csv` — **[Add dataset source link, e.g. Kaggle]** |
| **Rows** | 545 |
| **Columns** | 13 (12 features + 1 target) |
| **Target Variable** | `price` (house price, in USD; rescaled to millions during preprocessing) |
| **Missing Values** | None found |

## 4. 🧾 Features Used

| Feature | Description |
|---|---|
| `area` | Total area of the house (in square feet) |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `stories` | Number of stories in the house |
| `mainroad` | Whether the house is connected to the main road (Yes/No → 1/0) |
| `guestroom` | Whether a guest room is available (Yes/No → 1/0) |
| `basement` | Whether the house has a basement (Yes/No → 1/0) |
| `hotwaterheating` | Whether hot water heating is available (Yes/No → 1/0) |
| `airconditioning` | Whether air conditioning is available (Yes/No → 1/0) |
| `parking` | Number of parking spots available |
| `prefarea` | Whether the house is in a preferred/desirable area (Yes/No → 1/0) |
| `furnishingstatus` | Furnishing level: furnished / semi-furnished / unfurnished → 1 / 0.5 / 0 |
| `price` *(target)* | Sale price of the house, in millions of USD (rounded to 2 decimals) |

## 5. 🛠️ Technologies Used

* 🐍 Python
* 🐼 Pandas
* 🔢 NumPy
* 📈 Matplotlib
* 🎨 Seaborn
* 🤖 Scikit-learn
* 📓 Jupyter Notebook
* 💾 Joblib (model persistence)

## 6. 🔄 Project Workflow

```
Dataset (Housing.csv)
        ↓
Data Cleaning (null check, dtype check)
        ↓
Categorical Encoding (Yes/No → 1/0, furnishing status)
        ↓
Outlier Removal (IQR method on price)
        ↓
Exploratory Data Analysis (EDA)
        ↓
Train-Test Split
        ↓
Model Training (Ridge Regression)
        ↓
Model Evaluation (MAE, MSE, R²)
        ↓
Prediction on New Data
        ↓
Model Saving (.pkl)
```

## 7. 🧹 Data Preprocessing

* **Missing Values:** Checked using `isnull().sum()` — the dataset had no missing values.
* **Encoding Categorical Variables:** Binary Yes/No columns (`mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`) were mapped to `1`/`0`. The `furnishingstatus` column was mapped to `1` (furnished), `0.5` (semi-furnished), and `0` (unfurnished).
* **Target Transformation:** `price` was rescaled from raw USD to millions of USD and rounded to 2 decimal places for readability.
* **Outlier Removal:** Applied the **IQR (Interquartile Range)** method on the `price` column to remove extreme outliers and improve model stability.
* **Feature Scaling:** Not applied in the final model (a `StandardScaler` + `PolynomialFeatures` pipeline was tested but not used in the final Ridge model).
* **Train-Test Split:** The dataset was split using `train_test_split` with `test_size=0.2` and `random_state=123`.

> ⚠️ **Note:** Due to how the split output was unpacked in the notebook, the model was effectively trained on a smaller portion of the data (~106 rows) and evaluated on the larger portion (~424 rows). This is worth revisiting, and combined with the dataset's overall small size (545 rows), it's a likely contributor to the moderate evaluation scores below.

## 8. 🔍 Exploratory Data Analysis (EDA)

* **Scatter Plot (Area vs. Price):** Used to visually inspect the relationship between house area and price.
* **Correlation Heatmap:** Used `seaborn.heatmap()` over the full correlation matrix to identify which features are most strongly related to each other and to price.
* **Feature vs. Price Correlation (Bar Chart):** A horizontal bar chart ranking each feature's correlation with `price`, making it easy to spot the strongest predictors (`area`, `bathrooms`, `airconditioning`, and `stories` showed the strongest positive correlations).

## 9. 🤖 Machine Learning Model

* **Algorithm Used:** Ridge Regression (`sklearn.linear_model.Ridge`, `alpha=10`)
* **Why It Was Selected:** Several models were tested (Linear Regression, Polynomial Regression with scaling, Decision Tree, Random Forest, Gradient Boosting), and Ridge Regression produced the best R² score among them.
* **How It Works:** Ridge Regression is a linear regression variant that adds an L2 regularization penalty to the loss function. This discourages overly large coefficients, which helps reduce overfitting — especially useful on smaller datasets like this one.

## 10. 📏 Model Evaluation

| Metric | Value |
|---|---|
| **MAE** (Mean Absolute Error) | 0.7025 (≈ $702K) |
| **MSE** (Mean Squared Error) | 0.8944 |
| **R² Score** | 0.6440 |

*(Price values are in millions of USD, matching the target variable's scale.)*

## 11. 🔮 Sample Prediction

**Input:**

| area | bedrooms | bathrooms | stories | mainroad | guestroom | basement | hotwaterheating | airconditioning | parking | prefarea | furnishingstatus |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5000 | 3 | 2 | 2 | 1 | 0 | 1 | 0 | 1 | 2 | 1 | 1 |

**Output:**

```
Predicted Price ≈ $6.677 million
```

## 12. 🗂️ Project Structure

```
RealEstatePricePredictor/
│
├── Data/
│   └── Housing.csv
│
├── house_price_prediction.ipynb
├── house_price_model.pkl
├── requirements.txt
└── README.md
```

## 13. ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook
```

## 14. ▶️ How to Use

1. Place `Housing.csv` inside the `Data/` folder (or update the file path in the notebook).
2. Open `house_price_prediction.ipynb` in Jupyter Notebook.
3. Run all cells sequentially to: load data → clean & encode → remove outliers → explore data → train the Ridge Regression model → evaluate performance.
4. Use the trained model to predict prices on new input data (see [Sample Prediction](#11--sample-prediction)).
5. The trained model is saved automatically as `house_price_model.pkl` using `joblib`, and can be reloaded with `joblib.load("house_price_model.pkl")` for future predictions without retraining.

## 15. 📈 Results

The final Ridge Regression model achieved an **R² score of 0.644**, meaning it explains roughly 64% of the variance in house prices using the available features. The Mean Absolute Error of ~$702K reflects the moderate precision achievable with a relatively small dataset (545 rows). Performance is expected to improve with a larger dataset, additional features (e.g. location coordinates, year built), and further model tuning.

## 16. 🚀 Future Improvements

* 🌲 Try ensemble models such as Random Forest or Gradient Boosting (already scaffolded but commented out in the notebook)
* ⚡ Add XGBoost for potentially stronger performance
* 🎛️ Perform systematic hyperparameter tuning (e.g. `GridSearchCV`)
* 🧪 Engineer additional features (e.g. price per square foot, location data)
* 🌐 Deploy the model as an interactive web app using Streamlit
* 🔎 Add model explainability using SHAP to interpret individual predictions
* 📦 Expand the dataset size to improve generalization and reduce error

## 17. 🎓 Learning Outcomes

Through this project, the following concepts were practiced:

* Cleaning and preprocessing real-world tabular data
* Encoding categorical variables for machine learning models
* Detecting and removing outliers using the IQR method
* Performing exploratory data analysis with correlation heatmaps and scatter plots
* Comparing multiple regression algorithms to select the best performer
* Applying regularization (Ridge Regression) to control overfitting
* Evaluating regression models using MAE, MSE, and R²
* Saving and persisting trained models with `joblib`

## 18. ✅ Conclusion

This project demonstrates a complete end-to-end regression workflow — from raw data cleaning to a saved, reusable prediction model — for estimating house prices based on structural and amenity-related features. While the current Ridge Regression model achieves reasonable accuracy given the dataset size, it establishes a solid foundation for further improvement through more data, feature engineering, and advanced modeling techniques.

## 19. 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

## 20. 🙏 Acknowledgements

* [Kaggle](https://www.kaggle.com/) for hosting accessible housing datasets
* [Scikit-learn](https://scikit-learn.org/) for the machine learning tools used
* [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/) for data manipulation
* [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for data visualization