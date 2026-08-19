# Bank Marketing Campaign — Imbalanced Binary Classification

A machine learning project for predicting whether a bank customer will subscribe to a term-deposit offer.

The project focuses on **imbalanced binary classification**, **categorical feature engineering**, **model comparison**, **hyperparameter tuning**, and choosing evaluation metrics that reflect the actual business objective.

---

## 📌 Problem Statement

A bank runs a marketing campaign in which customers are contacted about a term-deposit plan.

The goal is to predict:

> **Will a given customer subscribe to the term deposit?**

The target variable is binary:

- `yes` → customer subscribes
- `no` → customer does not subscribe

The dataset is highly imbalanced, with roughly **89% negative samples and 11% positive samples**.

Because of this imbalance, accuracy alone is not a reliable evaluation metric. A model that predicts `no` for almost every customer can achieve high accuracy while being practically useless.

---

## 🎯 Business Objective

The model can help the sales team prioritize customers who are more likely to convert.

For example:

- Customers with a high predicted probability can be prioritized by the sales team.
- Customers with a very low probability can receive lower priority.
- This can reduce wasted sales effort and help allocate limited resources more effectively.

The project therefore gives particular importance to **Recall**, because missing an actual potential customer (False Negative) can have a larger business consequence than contacting a customer who ultimately does not subscribe (False Positive).

---

## 🧠 Models Used

The project compares multiple classification approaches:

1. **Logistic Regression** — baseline model
2. **Random Forest** — bagging-based ensemble
3. **XGBoost** — gradient boosting
4. **LightGBM** — gradient boosting
5. **CatBoost** — gradient boosting with native categorical-feature handling

The ensemble models are based on decision-tree learners.

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost

---

## 📊 Dataset

The Bank Marketing dataset contains customer information, campaign information, and economic indicators.

The dataset used in the project contains approximately:

- **41,000 rows**
- **21 columns**
- **20 input features**
- **1 target column**

The features include both numerical and categorical variables.

### Example feature groups

| Feature | Description |
|---|---|
| `age` | Customer age |
| `job` | Type of occupation |
| `marital` | Marital status |
| `education` | Education level |
| `default` | Credit default status |
| `housing` | Housing loan status |
| `loan` | Personal loan status |
| `contact` | Contact communication type |
| `month` | Month of last contact |
| `day_of_week` | Day of the week of contact |
| `duration` | Duration of the last contact |
| `campaign` | Number of contacts during the current campaign |
| `pdays` | Days since previous campaign contact |
| `previous` | Number of contacts before the current campaign |
| `poutcome` | Outcome of the previous campaign |
| Economic indicators | Employment, inflation, interest-rate and related indicators |
| `y` | Term-deposit subscription target |

---

## 🔍 Important Feature Decision

The `duration` feature is removed.

The reason is that call duration is only known **after the call has already happened** and can directly reveal whether the customer was interested.

Using it for prediction would introduce information that would not realistically be available before making the call.

This makes `duration` a form of **data leakage / unfair predictive signal** for the intended business use case.

---

## ⚙️ Data Preprocessing

Two feature representations are used depending on the model.

### Version A — Encoded Features

Used for models that require numerical input:

- Logistic Regression
- Random Forest

Categorical variables are converted into numerical features using **one-hot encoding**.

For a categorical variable with `n` categories, `n - 1` encoded columns are retained to avoid unnecessary multicollinearity.

### Version B — Native Categorical Features

Used for models capable of handling categorical variables directly:

- XGBoost
- LightGBM
- CatBoost

Categorical columns are explicitly marked as categorical rather than one-hot encoded.

This allows the models to apply their native categorical handling.

---

## 📈 Evaluation Metrics

Because the dataset is imbalanced, several metrics are considered:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Confusion Matrix
- Classification Report
- ROC Curve
- Precision-Recall Curve

### Why Recall matters

The project treats **False Negatives** as more costly than False Positives.

A False Negative means:

> The customer was actually interested in the term deposit, but the model predicted that they were not.

Missing such a customer can mean losing a potential conversion.

Therefore, the model should not be judged purely by accuracy.

---

## 🔬 Model Evaluation

The initial models produced F1 scores in approximately the **0.46–0.49 range** after applying class-weighting strategies.

The lecture analysis reports:

| Model | Observation |
|---|---|
| Logistic Regression | Strong baseline |
| Random Forest | Comparable performance |
| XGBoost | No major advantage over baseline |
| LightGBM | PR-AUC around `0.49` |
| CatBoost | Strong categorical-data handling, but no major advantage over baseline |

The reported Logistic Regression PR-AUC was around **0.46**, while LightGBM reached around **0.49**. The improvement was considered too small to justify the additional model complexity in a production setting.

The analysis therefore favors **Logistic Regression** when considering training time, inference time, and overall performance.

---

## ⚖️ Class Imbalance Handling

The dataset contains approximately:

- **89% → `no`**
- **11% → `yes`**

To reduce the effect of class imbalance, class weights are considered.

For a roughly 89:11 class distribution, the minority class receives substantially more weight so that mistakes on positive samples have a greater influence during training.

This is particularly useful when Recall for the positive class is important.

---

## 🎚️ Threshold Tuning

Most classifiers output a probability rather than a final class directly.

The default classification threshold is:

```text
0.5
```

If the predicted probability is greater than the threshold, the sample is classified as positive.

However, the threshold can be changed according to the business objective.

### Lower threshold

If the business wants to catch more potential subscribers:

```text
Lower threshold → Higher Recall
```

This can increase the number of customers identified as potential leads, while also increasing False Positives.

The project demonstrates threshold analysis, including evaluating a threshold around `0.2`.

---

## 🔧 Hyperparameter Tuning

`GridSearchCV` is used for hyperparameter optimization.

The tuning process uses cross-validation to evaluate different combinations of hyperparameters and identify promising configurations.

Parameters explored vary by model and include settings such as:

- Tree depth
- Learning rate
- Number of estimators
- Class weights
- Other model-specific parameters

For CatBoost, for example, candidate depths and learning-rate combinations were explored.

---

## 📉 ROC Curve vs Precision-Recall Curve

Both ROC and Precision-Recall curves are examined.

### ROC Curve

The ROC curve evaluates the relationship between:

- True Positive Rate
- False Positive Rate

ROC-AUC provides a single numerical summary of the ROC curve.

### Precision-Recall Curve

The Precision-Recall curve is particularly useful for this project because the positive class is relatively rare.

It focuses on:

- Precision
- Recall

The project therefore gives more attention to PR-AUC when comparing models on the imbalanced dataset.

---

## 🧪 Project Workflow

```text
Load Dataset
     ↓
Understand Business Problem
     ↓
Explore Dataset
     ↓
Identify Numerical & Categorical Features
     ↓
Remove Duration / Leakage-Prone Feature
     ↓
Create Encoded & Native-Categorical Data Versions
     ↓
Train Baseline Logistic Regression
     ↓
Train Ensemble Models
     ↓
Hyperparameter Tuning
     ↓
Handle Class Imbalance
     ↓
Evaluate Precision / Recall / F1
     ↓
Compare ROC-AUC & PR-AUC
     ↓
Tune Classification Threshold
     ↓
Select Model Based on Business Requirements
```

---

## 📁 Suggested Project Structure

```text
bank-marketing-classification/
│
├── data/
│   └── bank-full.csv
│
├── notebooks/
│   └── bank_marketing_classification.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
│
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── precision_recall_curve.png
│
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/bank-marketing-classification.git
cd bank-marketing-classification
```

Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn xgboost lightgbm catboost
```

---

## ▶️ Running the Project

Open the notebook:

```bash
jupyter notebook
```

Then run:

```text
notebooks/bank_marketing_classification.ipynb
```

---

## 💡 Key Learnings

This project demonstrates practical concepts beyond simply training a classifier:

- Handling highly imbalanced datasets
- Why accuracy can be misleading
- Choosing metrics according to business cost
- Precision vs Recall trade-off
- F1 Score
- ROC-AUC and PR-AUC
- One-hot encoding
- Native categorical feature handling
- Data leakage
- Class weighting
- Hyperparameter tuning with GridSearchCV
- Threshold optimization
- Comparing simple models with complex ensemble models
- Considering training and inference cost when selecting a production model

---

## 🏁 Conclusion

The main takeaway from this project is that **the most complex model is not automatically the best model**.

Although boosting models such as LightGBM and CatBoost can handle categorical features effectively, the experiments did not show a sufficiently large performance advantage over the Logistic Regression baseline.

For a production system, the project therefore favors **Logistic Regression** because it provides competitive performance while being simpler and cheaper to train and serve.

The final model choice should ultimately depend on the business objective, especially the relative cost of False Positives and False Negatives.
