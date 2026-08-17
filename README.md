# Bank Marketing Classification

## a. Problem Statement

The objective of this project is to develop and compare multiple machine learning classification models for predicting whether a bank customer will subscribe to a term deposit.

The target variable is `deposit`, where `yes` indicates that the customer subscribed to a term deposit and `no` indicates that the customer did not subscribe.

The following classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

The models were evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

## b. Dataset Description

The Bank Marketing dataset was obtained from Kaggle.

- Problem Type: Binary Classification
- Number of Instances: 45,211
- Number of Input Features: 16
- Target Variable: `deposit`
- Feature Types: Numerical and Categorical

The dataset contains customer demographic, financial, and marketing campaign information.

### Target Variable

| Value | Meaning |
|---|---|
| yes | Customer subscribed to a term deposit |
| no | Customer did not subscribe to a term deposit |

### Preprocessing

Numerical features were standardized using StandardScaler.

Categorical features were converted using One-Hot Encoding.

The dataset was divided into training and testing sets using an 80:20 split with stratified sampling.

## c. Github Repository Link

**GitHub Repository:** ADD GITHUB LINK HERE

## d. Models Used

### Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8262 | 0.9071 | 0.8278 | 0.7996 | 0.8135 | 0.6513 |
| Decision Tree | 0.7944 | 0.7933 | 0.7894 | 0.7722 | 0.7807 | 0.5874 |
| kNN | 0.8173 | 0.8796 | 0.8199 | 0.7873 | 0.8033 | 0.6333 |
| Naive Bayes | 0.7201 | 0.8042 | 0.7837 | 0.5652 | 0.6568 | 0.4472 |
| Random Forest (Ensemble) | **0.8621** | **0.9193** | **0.8301** | **0.8913** | **0.8596** | **0.7262** |

### Model Observations

**Logistic Regression**

Logistic Regression achieved an accuracy of 82.62% and an AUC of 0.9071. Its precision and recall were relatively balanced, resulting in an F1 score of 0.8135. It demonstrated strong overall classification performance.

**Decision Tree**

The Decision Tree achieved an accuracy of 79.44% and an AUC of 0.7933. Its performance was lower than Logistic Regression, kNN, and Random Forest across most metrics.

**kNN**

kNN achieved an accuracy of 81.73% and an AUC of 0.8796. Its precision, recall, and F1 score were relatively strong, but it performed below Logistic Regression and Random Forest.

**Naive Bayes**

Naive Bayes achieved the lowest overall performance, with an accuracy of 72.01%, recall of 56.52%, F1 score of 0.6568, and MCC of 0.4472.

**Random Forest (Ensemble)**

Random Forest achieved the best overall performance, with an accuracy of 86.21%, AUC of 0.9193, precision of 83.01%, recall of 89.13%, F1 score of 85.96%, and MCC of 0.7262.

Its strong performance across all six metrics indicates that combining multiple decision trees was effective at capturing complex relationships in the dataset.

### Overall Winner

**Random Forest is the overall winner for this dataset.**

It achieved the highest Accuracy, AUC, Recall, F1 Score, MCC, and average score among the evaluated models.

## Streamlit Application

The Streamlit application provides an interactive interface for evaluating the trained classification models.

### Features

- CSV test-data upload
- Model selection dropdown
- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- MCC
- Confusion matrix
- Prediction summary

### Live Streamlit App

**Streamlit App:** ADD STREAMLIT APP LINK HERE

## Project Structure

project-folder/

|-- app.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|
|-- model/
    |-- logistic_regression.pkl
    |-- decision_tree.pkl
    |-- knn.pkl
    |-- naive_bayes.pkl
    |-- random_forest.pkl

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Seaborn
- Streamlit
