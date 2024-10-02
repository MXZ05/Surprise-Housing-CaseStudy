---

# Housing Price Prediction using Ridge and Lasso Regression

This repository contains a project to predict house prices using Ridge and Lasso regression models. The goal is to preprocess the dataset, build predictive models, and optimize hyperparameters using cross-validation.

## Dataset

The dataset can be downloaded from Kaggle: [Housing Price Dataset](https://www.kaggle.com/code/sid9300/assignment-surprise-housing-l-r/input).

## Key Steps:
1. **Data Preprocessing**: 
   - Handle missing values.
   - Log-transform `SalePrice` to address skewness.
   - Feature engineering and scaling.
2. **Modeling**:
   - Ridge and Lasso regression models with hyperparameter tuning using `GridSearchCV`.
   - Comparison of model performance using R² score and RMSE.
3. **Feature Importance**:
   - Analyze and compare important features based on model coefficients.
4. **Visualization**:
   - Plot performance based on different regularization strengths.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MXZ05/Surprise-Housing-CaseStudy.git
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Ensure the dataset is in the project directory.
2. Run the script:
   ```bash
   python SurpriseHousingCaseStudy.py
   ```

## License

This project is licensed under the [MIT License](LICENSE).

---
