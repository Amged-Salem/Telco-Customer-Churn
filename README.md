
# Telecom Customer Churn Analysis

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Objectives](#objectives)
4. [Methodology](#methodology)
   - [Model Types](#model-types)
     - [Clustering Models](#1-clustering-models)
     - [Classification Models](#2-classification-models)
     - [Regression Models](#3-regression-models)
   - [Data Preprocessing](#data-preprocessing)
5. [Results](#results)
6. [How to Use](#how-to-use)
7. [Conclusion](#conclusion)

---

## Project Overview

This project aims to analyze customer churn behavior in the telecom sector by segmenting customers into clusters and predicting their churn likelihood using various machine learning models. The project is divided into two main parts:
1. **Clustering Analysis**: Segmenting customers with K-Means clustering.
2. **Predictive Modeling**: Using a range of machine learning models to predict customer churn.

## Dataset

- **Source**: Kaggle's [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) dataset.
- **Description**: Contains customer data, such as demographics, account details, and service usage information, with a target variable `Churn` indicating customer retention.

## Objectives

1. **Cluster Analysis**: Identify customer segments with similar behaviors.
2. **Predictive Modeling**: Use machine learning models to predict customer churn, aiming for high accuracy and insightful ROC metrics.

## Methodology

### Model Types

The models are organized into three categories based on their purpose: **Clustering**, **Classification**, and **Regression**.

#### 1. Clustering Models
   - **K-Means Clustering**: Used to segment customers into groups based on feature similarity, providing insights into customer types.

#### 2. Classification Models
   - **Random Forest Classifier**
   - **AdaBoost Classifier**
   - **Gradient Boosting Classifier**
   - **XGBoost**
   - **Support Vector Machine (SVM)**
   - **Logistic Regression**
   - **K-Nearest Neighbors (KNN)**
   - **Decision Tree Classifier**
   - **Naive Bayes**
   
   **Objective**: Predict whether a customer will churn, based on accuracy, ROC AUC, and classification metrics.

#### 3. Regression Models
   - **Logistic Regression** (for binary classification with probability outputs)
   - **Decision Tree Regressor** (used for probability estimates if adapted to a regression format)
   
   **Objective**: Provide churn probability estimates where suitable, especially for borderline churn cases.

### Data Preprocessing

1. **Data Cleaning**: Handling missing values, encoding categorical features, and scaling numeric features.
2. **Balancing**: Using SMOTE sampling to address class imbalance in the target variable.
3. **Feature Scaling**: Standardizing or normalizing data as required for different models.

## Results

- **Clustering Analysis**: K-Means clustering grouped customers based on usage patterns, helping identify different customer types.
- **Classification & Regression Models**: Achieved up to 98% accuracy and strong ROC scores across various models, indicating reliable predictions for churn likelihood.

## How to Use

1. **Requirements**: Ensure installation of libraries like `pandas`, `scikit-learn`, `imblearn`, `seaborn`, `matplotlib`, `plotly`, and `xgboost`.
2. **Running the Notebooks**: Load the dataset (`Telco-Customer-Churn.csv`) and execute each cell to reproduce results.

## Conclusion

This project provides a complete approach to customer segmentation and churn prediction. The clustering analysis reveals insights into customer behavior, while the classification and regression models offer actionable predictions to improve retention strategies.

--- 
