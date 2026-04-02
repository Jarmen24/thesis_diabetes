# 🧠 Predictive Modeling

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

> The centralized repository for data preprocessing, model training, and performance evaluation pipelines. 

## 📖 Overview

This repository contains the machine learning workflows used to build and export predictive models. It encompasses the entire data science lifecycle—from raw data ingestion and feature engineering to hyperparameter tuning and final model deployment. 

The models developed here are designed for integration into production environments, providing the core analytical engine for predictive health risk assessments.

## ⚙️ Methodology & Pipeline

The training pipeline is built primarily using Python and the `scikit-learn` ecosystem, focusing on rigorous validation and handling of real-world data constraints.

### 1. Data Preprocessing
* **Feature Scaling:** Standardization and normalization techniques applied to ensure all features contribute equally to the model.
* **Cleaning:** Pipeline steps for handling missing values, encoding categorical variables, and removing outliers.

### 2. Data Balancing
* **SMOTE (Synthetic Minority Over-sampling Technique):** Implemented to handle severe class imbalances in the training datasets, ensuring the model does not become biased toward majority classes and accurately identifies minority risk cases.

### 3. Model Training
* Utilization of various classification algorithms (e.g., Logistic Regression, Random Forests) to determine the best fit for the predictive task.
* Grid search and cross-validation for hyperparameter tuning.

### 4. Performance Evaluation
* **ROC-AUC Optimization:** Models are heavily evaluated using the Receiver Operating Characteristic - Area Under Curve metric. This ensures a strong balance between true positive rates (sensitivity) and false positive rates, which is critical for risk-based predictions.
* Additional metrics: Precision, Recall, and F1-Score reports are generated for every iteration.
