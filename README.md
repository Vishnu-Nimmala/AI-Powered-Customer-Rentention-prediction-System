<div align="center">

# 🧠 AI Powered Customer Retention System

### Telecom Churn Prediction Using Logistic Regression

<img src="static/images/churn_banner.png" width="900">

</div>

---

## 📌 Project Overview

The **AI Powered Customer Retention System** is a Machine Learning web application that predicts whether a telecom customer is likely to churn or stay with the company.

This system helps telecom organizations identify high-risk customers and take proactive retention actions to reduce customer loss and improve business growth.

---

## 🚀 Live Demo

### 🌐 Application Link

https://https://ai-powered-customer-rentention.onrender.com

### 📂 GitHub Repository

https://https://github.com/Vishnu-Nimmala/AI-Powered-Customer-Rentention-prediction-System

---

## 🎯 Business Problem

Customer churn is one of the biggest challenges faced by telecom companies.

Acquiring a new customer costs much more than retaining an existing customer. Therefore, predicting churn in advance enables organizations to take preventive actions and improve customer retention.

---

## 🛠 Technologies Used

<table>
<tr>
<td><b>Programming Language</b></td>
<td>Python</td>
</tr>

<tr>
<td><b>Libraries</b></td>
<td>Pandas, NumPy, Scikit-Learn, Imbalanced-Learn, Joblib</td>
</tr>

<tr>
<td><b>Frontend</b></td>
<td>HTML, CSS, JavaScript</td>
</tr>

<tr>
<td><b>Backend</b></td>
<td>Flask</td>
</tr>

<tr>
<td><b>Deployment</b></td>
<td>Render</td>
</tr>

</table>

---

## 📊 Machine Learning Pipeline

```text
Dataset
   ↓
Data Cleaning
   ↓
Missing Value Handling
   ↓
Categorical Encoding
   ↓
Variable Transformation
   ↓
Outlier Treatment
   ↓
Feature Selection
   ↓
SMOTE Balancing
   ↓
Logistic Regression
   ↓
Model Deployment
```

---

## 🔧 Data Preprocessing

### Missing Value Handling

✔ Numerical Features → Median Imputation

✔ Categorical Features → Mode Imputation

### Categorical Encoding

#### One Hot Encoding

- Gender
- Partner
- Dependents

#### Ordinal Encoding

- Phone Service
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract Type
- Payment Method

---

## 📈 Feature Engineering

### Variable Transformation

✔ Yeo-Johnson Transformation

Used to reduce skewness and improve feature distribution.

Examples:

- tenure_yeo_tri
- TotalCharges_rep_yeo_CapMS

### Outlier Handling

Applied robust outlier treatment techniques to minimize the impact of extreme values.

---

## 🎯 Feature Selection

### Variance Threshold

Removed low variance and quasi-constant features.

### Pearson Correlation

Selected statistically significant features using correlation analysis and p-values.

---

## ⚖ Data Balancing

### SMOTE (Synthetic Minority Oversampling Technique)

Benefits:

- Handles class imbalance
- Improves Recall
- Better churn detection
- Reduces model bias

---

## 🤖 Model Used

### Logistic Regression

Configuration:

```python
LogisticRegression(
    penalty='l2',
    solver='sag'
)
```

Advantages:

✔ Fast

✔ Interpretable

✔ Suitable for Binary Classification

✔ Good Generalization

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve
- AUC Score

---

## 🎨 Application Features

### Input Features

- Senior Citizen
- Gender
- Partner
- Dependents
- Tenure
- Monthly Charges
- Total Charges
- Phone Service
- Internet Service
- Contract Type
- Payment Method
- Streaming Services
- SIM Operator

### Prediction Output

✅ Churn Probability

✅ Retention Probability

✅ Risk Level

✅ Customer Retention Status

---

## 📂 Project Structure

```bash
AI_powered_churn_prediction_system/
│
├── app.py
├── requirements.txt
├── model.pkl
├── scaler.pkl
├── encoder.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── preprocessing/
│   ├── balancing_data.py
│   ├── Cat_to_num.py
│   └── Fs.py
│
└── README.md
```


## 👩‍💻 Developed By

### Nimmala Vishnu

**Data Analyst & ML Engineer**

📧 nimmalavishnu602@gmail.com

🏢 ViharaTech Private Limited

---

## ⭐ Project Highlights

✅ Telecom Customer Churn Prediction

✅ End-to-End Machine Learning Pipeline

✅ Flask Web Application

✅ Logistic Regression Model

✅ Data Balancing Using SMOTE

✅ Feature Engineering & Selection

✅ Interactive User Interface

✅ Deployment on Render

---

<div align="center">

### ⭐ If you like this project, please give it a star ⭐

</div>
