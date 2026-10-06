# credit-card-default-prediction
Machine learning project for predicting credit card defaults using the UCI Credit Card Default dataset, feature engineering, SMOTE, PCA, and multiple classification models.
# Credit Card Default Prediction

Machine learning project for predicting whether a credit card customer will default on their payment in the following month using the UCI Credit Card Default Dataset.

## Project Overview

Credit card default prediction is a binary classification problem where the objective is to identify customers who are likely to default on their credit card payment in the next month.

This project explores customer demographic information, credit limits, repayment history, billing amounts, and payment behavior to build predictive machine learning models.

The project compares five classification algorithms and evaluates them using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.

## Objectives

The main objectives of this project are to:

- Explore and understand the credit card default dataset
- Clean inconsistent categorical values
- Engineer meaningful financial and behavioral features
- Handle class imbalance using SMOTE
- Build a reproducible machine learning pipeline
- Compare multiple classification algorithms
- Tune the classification decision threshold
- Evaluate models using appropriate classification metrics
- Identify the model with the best balance between precision and recall

## Dataset

The project uses the **UCI Credit Card Default Dataset**, containing information about 30,000 credit card customers in Taiwan.

The target variable is:

```text
default.payment.next.month
```

Where:

- `0` = Customer did not default
- `1` = Customer defaulted

The dataset contains 24 predictive features after removing the `ID` column.

### Main Feature Categories

- Demographic information
- Credit limit
- Repayment status
- Bill amounts
- Payment amounts

## Data Preprocessing

Several preprocessing steps were applied before model training.

### Data Cleaning

Invalid categorical values were corrected:

- `EDUCATION` values `0, 4, 5, 6` were grouped into category `4`
- `MARRIAGE` value `0` was reassigned to category `3`

### Feature Engineering

Additional features were created to capture customer financial behavior:

- `TOTAL_BILL_AMT`
- `TOTAL_PAY_AMT`
- `PAY_RATIO`
- `BILL_TREND`
- `AVG_PAY_DELAY`
- `DELAY_TREND`
- `PAY_VOLATILITY`
- `UTILIZATION`

These features capture spending levels, repayment behavior, payment consistency, debt trends, and credit utilization.

## Machine Learning Pipeline

The project uses a consistent pipeline for each model:

```text
Raw Features
     ↓
RobustScaler
     ↓
PCA (15 Components)
     ↓
SMOTE
     ↓
Machine Learning Model
```

### RobustScaler

RobustScaler was selected because financial datasets can contain significant outliers. It scales features using statistics that are less sensitive to extreme values.

### PCA

Principal Component Analysis with 15 components is used to reduce dimensionality and simplify the feature space.

### SMOTE

SMOTE is used to address class imbalance by generating synthetic examples of the minority class.

## Train-Test Split

The dataset is divided into:

- 80% training data
- 20% testing data

A stratified split is used to preserve the distribution of default and non-default customers.

The project uses:

```python
RANDOM_STATE = 42
```

to ensure reproducibility.

## Models Evaluated

Five classification algorithms were evaluated:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. K-Nearest Neighbors (KNN)
5. Naive Bayes

## Decision Threshold

Instead of using the standard classification threshold of `0.50`, this project uses:

```text
Threshold = 0.45
```

A customer is classified as a potential defaulter when:

```text
probability >= 0.45
```

The lower threshold is intended to increase recall and identify more potential defaulters.

## Evaluation Metrics

The following metrics are used:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

Because the dataset is imbalanced, particular attention is given to **Recall and F1-Score** rather than accuracy alone.

## Results

| Rank | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Random Forest | 0.746 | 0.447 | 0.622 | **0.520** | **0.764** |
| 2 | Gradient Boosting | 0.719 | 0.415 | 0.658 | 0.509 | 0.761 |
| 3 | KNN | 0.633 | 0.340 | 0.705 | 0.459 | 0.729 |
| 4 | Logistic Regression | 0.566 | 0.298 | 0.710 | 0.420 | 0.712 |
| 5 | Naive Bayes | 0.292 | 0.233 | **0.959** | 0.375 | 0.711 |

## Best Performing Model

**Random Forest** achieved the best overall balance between precision and recall, resulting in the highest F1-score of `0.520`.

It achieved:

```text
Accuracy:   0.746
Precision:  0.447
Recall:     0.622
F1-Score:   0.520
ROC-AUC:    0.764
```

Although Naive Bayes achieved very high recall (`0.959`), its low precision and accuracy indicate a large number of false positive predictions.

Therefore, Random Forest provides the most practical balance among the evaluated models.

## Visualizations

The project includes:

- Confusion matrices for each model
- F1-score comparison
- Model performance leaderboard

These visualizations help identify model strengths, weaknesses, and classification trade-offs.

## Project Structure

```text
credit-card-default-prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── notebooks/
│   └── credit_card_default_prediction.ipynb
│
├── data/
│   └── README.md
│
├── src/
│   └── README.md
│
├── reports/
│   └── Credit_Card_Default_Report.pdf
│
└── images/
    └── model_comparison.png
```

## Technologies Used

- Python
- NumPy
- pandas
- Matplotlib
- Seaborn
- scikit-learn
- imbalanced-learn
- Jupyter Notebook / Google Colab

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/credit-card-default-prediction.git
```

Move into the project directory:

```bash
cd credit-card-default-prediction
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Open the notebook:

```text
notebooks/credit_card_default_prediction.ipynb
```

The original notebook is designed to run in Google Colab.

Before running the notebook, make sure the required dataset is available at the expected path or upload the dataset to the Colab environment.

## Future Improvements

Potential improvements include:

- Hyperparameter tuning using Grid Search or Random Search
- Evaluation of XGBoost and LightGBM
- Model explainability using SHAP
- More systematic threshold optimization
- Cross-validation
- Model deployment through an API or web application
- Real-time credit risk scoring
- Model monitoring after deployment

## Author

**Temesgen Bahlbi**

This project was developed as a machine learning study focused on credit risk prediction and classification.

## License

This project is intended for educational and research purposes.
