# 🏠 House Price Prediction

An end-to-end machine learning project that predicts house prices based on various property characteristics. The project includes data preprocessing, multiple regression models, cross-validation, hyperparameter tuning, model evaluation, and a Streamlit web application for making predictions.

## 🚀 Project Overview

The goal of this project is to build a machine learning model capable of predicting house sale prices from property-related features.

The project follows a complete machine learning workflow:

- Data preprocessing
- Handling missing values
- Categorical feature encoding
- Train-test splitting
- Multiple regression models
- Cross-validation
- Hyperparameter tuning using GridSearchCV
- Model comparison
- Feature importance analysis
- Residual analysis
- Model serialization
- Streamlit deployment

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib

## 📊 Dataset

The project uses the Ames Housing dataset, which contains information about residential properties and their corresponding sale prices.

For the final model, the following features were used:

- MSSubClass
- MSZoning
- LotArea
- LotConfig
- BldgType
- OverallCond
- YearBuilt
- YearRemodAdd
- Exterior1st
- BsmtFinSF2
- TotalBsmtSF

`Id` was removed because it is only an identifier and does not represent a meaningful property characteristic.

`SalePrice` is used as the target variable.

## 🔄 Data Preprocessing

The preprocessing workflow includes:

1. Removing rows with missing target values.
2. Separating features and target variable.
3. Splitting the data into training and testing sets.
4. Handling missing feature values using imputers.
5. Encoding categorical variables using OneHotEncoder.
6. Using a Scikit-learn Pipeline to combine preprocessing and model training.

The preprocessing steps are fitted only on the training data and subsequently applied to the test data to avoid data leakage.

## 🤖 Models Evaluated

Several regression algorithms were evaluated:

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

Models were compared using:

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

## 🔍 Cross-Validation & Hyperparameter Tuning

5-fold cross-validation was used to evaluate model performance more reliably.

GridSearchCV was used for hyperparameter tuning of the major tree-based models.

Random Forest achieved the best cross-validation performance among the final candidate models and was selected as the primary model.

## 🏆 Final Model

The final model is a tuned Random Forest Regressor.

The final evaluation on the held-out test set achieved approximately:

| Metric | Score |
|---|---:|
| R² Score | 0.835 |
| RMSE | 35,565 |
| MAE | 22,863 |

The model explains approximately 83.5% of the variance in house prices on the held-out test data.

## 📈 Model Analysis

### Actual vs Predicted

The actual-vs-predicted plot showed that most predictions were reasonably close to the ideal prediction line.

### Residual Analysis

Residuals were generally centered around zero, although larger errors were observed for some higher-priced properties.

### Feature Importance

Feature importance from the Random Forest model was also analyzed to understand which property characteristics contributed most to predictions.

## 🌐 Streamlit Application

A Streamlit web application was developed to allow users to enter property details and receive an estimated house price.

### Application Workflow

```text
User Input
    ↓
Saved ML Pipeline
    ↓
Missing Value Handling
    ↓
Categorical Encoding
    ↓
Random Forest Model
    ↓
Predicted House Price