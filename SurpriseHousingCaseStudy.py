# Import necessary libraries and suppress warnings
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Display settings for pandas
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# Load dataset
housing = pd.read_csv("train.csv")

# Data Cleaning
missing_cols = ['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu',
                'GarageType','GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature']

for col in missing_cols:
    housing[col].fillna("None", inplace=True)

# Fill numeric missing values with the mode of each column
for col in housing.columns:
    if housing[col].dtype in [np.float64, np.int64]:
        housing[col].fillna(housing[col].mode()[0], inplace=True)
    else:
        housing[col].fillna(housing[col].mode()[0], inplace=True)

# Log transform SalePrice to address skewness
housing['SalePrice'] = np.log(housing['SalePrice'])

# Drop unnecessary columns
housing.drop("Id", axis=1, inplace=True)

# Convert selected columns to object (categorical) type
housing[['MSSubClass','OverallQual','OverallCond']] = housing[['MSSubClass','OverallQual','OverallCond']].astype('object')

# Feature engineering: Create new 'Age' column (difference between YrSold and YearBuilt)
housing["Age"] = housing["YrSold"] - housing["YearBuilt"]

# Drop 'YrSold' and 'YearBuilt'
housing.drop(columns=["YrSold", "YearBuilt"], axis=1, inplace=True)

# Prepare numeric and categorical features
housing_num = housing.select_dtypes(include=['int64', 'float64'])
housing_cat = housing.select_dtypes(include='object')

# One-hot encoding for categorical variables
housing_cat_dm = pd.get_dummies(housing_cat, drop_first=True, dtype=int)

# Concatenate numeric and encoded categorical data
house = pd.concat([housing_num, housing_cat_dm], axis=1)

# Splitting the dataset into features and target variable
X = house.drop(['SalePrice'], axis=1).copy()
y = house["SalePrice"].copy()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling the numeric features
scaler = StandardScaler()
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Function to evaluate the models
def eval_metrics(y_train, y_train_pred, y_test, y_pred):
    print("\nEvaluation Metrics:\n")
    print(f"R² score (Train): {r2_score(y_train, y_train_pred):.2f}")
    print(f"R² score (Test): {r2_score(y_test, y_pred):.2f}")
    
    # RMSE for train and test data
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"RMSE (Train): {rmse_train:.2f}")
    print(f"RMSE (Test): {rmse_test:.2f}")

# Hyperparameter tuning with Ridge and Lasso using GridSearchCV
params = {'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]}

# Ridge Regression
print("Performing Ridge Regression with Cross-Validation...")
ridge = Ridge()
ridgeCV = GridSearchCV(estimator=ridge, param_grid=params, scoring='neg_mean_absolute_error', cv=5, 
                       return_train_score=True, verbose=1, n_jobs=-1)
ridgeCV.fit(X_train, y_train)
ridge = Ridge(alpha=ridgeCV.best_params_['alpha'])
ridge.fit(X_train, y_train)

# Prediction and evaluation for Ridge
y_train_pred_ridge = ridge.predict(X_train)
y_test_pred_ridge = ridge.predict(X_test)
eval_metrics(y_train, y_train_pred_ridge, y_test, y_test_pred_ridge)

# Lasso Regression
print("\nPerforming Lasso Regression with Cross-Validation...")
lasso = Lasso(max_iter=5000)
lassoCV = GridSearchCV(estimator=lasso, param_grid=params, scoring='neg_mean_absolute_error', cv=5, 
                       return_train_score=True, verbose=1, n_jobs=-1)
lassoCV.fit(X_train, y_train)
lasso = Lasso(alpha=lassoCV.best_params_['alpha'])
lasso.fit(X_train, y_train)

# Prediction and evaluation for Lasso
y_train_pred_lasso = lasso.predict(X_train)
y_test_pred_lasso = lasso.predict(X_test)
eval_metrics(y_train, y_train_pred_lasso, y_test, y_test_pred_lasso)

# Comparing Ridge and Lasso Coefficients
betas = pd.DataFrame(index=X.columns)
betas['Ridge'] = ridge.coef_
betas['Lasso'] = lasso.coef_

print("\nTop 10 Features Based on Ridge Coefficients:")
print(betas['Ridge'].sort_values(ascending=False)[:10])

print("\nTop 10 Features Based on Lasso Coefficients:")
print(betas['Lasso'].sort_values(ascending=False)[:10])

# Visualizing Alpha (Lambda) values and their impact on MAE
ridgeCV_res = pd.DataFrame(ridgeCV.cv_results_)
lassoCV_res = pd.DataFrame(lassoCV.cv_results_)

plt.figure(figsize=[10, 6])
plt.plot(ridgeCV_res['param_alpha'], ridgeCV_res['mean_train_score'], label='Ridge Train', marker='o')
plt.plot(ridgeCV_res['param_alpha'], ridgeCV_res['mean_test_score'], label='Ridge Test', marker='o')
plt.plot(lassoCV_res['param_alpha'], lassoCV_res['mean_train_score'], label='Lasso Train', marker='x')
plt.plot(lassoCV_res['param_alpha'], lassoCV_res['mean_test_score'], label='Lasso Test', marker='x')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Negative Mean Absolute Error')
plt.title('Model Performance: Ridge vs Lasso')
plt.legend()
plt.grid(True)
plt.show()