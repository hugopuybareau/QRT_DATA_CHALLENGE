# QRT Data Challenge

This repository contains my submission for the QRT Data Challenge, where I finished **3rd**. The challenge focused on predicting the outcomes of football matches (home win, draw, away win) using advanced data preprocessing, feature engineering, and machine learning models.

## Project Overview

The objective was to build a model capable of accurately predicting match results based on player and team statistics. The project involved handling missing data, aggregating player features by position, and optimizing models for maximum predictive performance.

## Key Steps

### 1. Data Preprocessing
- **Missing Data Handling**: Managed missing values using median imputation.
- **Feature Engineering**: Aggregated player statistics by team and position, providing both positional and team-level insights for the model.
- **Data Augmentation**: Created additional features based on match outcomes and contextual information (home/away effects, etc.).

### 2. Model Selection and Optimization
- **Model Exploration**: Tried multiple models including **XGBoost**, **CatBoost**, and **TabNet** to find the best fit for the dataset.
- **Hyperparameter Tuning**: Leveraged **Hyperopt** for optimizing hyperparameters to boost model accuracy.
- **Ensemble Techniques**: Combined several models using a **StackingClassifier** to further improve predictive performance.

### 3. Cross-validation and Final Submission
- Used **Stratified KFold** cross-validation to avoid overfitting and ensure robust performance.
- The final model was selected based on a combination of accuracy metrics and cross-validation scores. However, the exact solution that led to the **3rd place** finish is not included in this repository.

## Usage

To run the provided notebooks, clone the repository and install the required dependencies:
git clone https://github.com/hugopuybareau/QRT_DATA_CHALLENGE.git
cd QRT_DATA_CHALLENGE

