# Home Credit Default Risk – ML Project

This project predicts whether a loan applicant will default, based on financial and demographic data from the [Home Credit Kaggle competition](https://www.kaggle.com/competitions/home-credit-default-risk). The data includes information from applications, credit history, and external sources.

Rather than optimizing for the highest AUC, the goal was to build a solid and interpretable baseline model using only a limited number of features. This project served as my first complete end-to-end machine learning pipeline, including feature engineering, cross-validation, ensembling, and automated evaluation.

## Project Structure

The repository is organized as follows:

```
Folders:

experiments/        Experimental trials
logs/hpo/           Log files from Hyper Parameter Optimization
notebooks/          Exploratory data analysis and model development
outputs/            Generated features and predictions
src/creditutils/    Utility functions for feature engineering and evaluation
submissions/        Submission files for Kaggle

Files in main directory:
README.md           Project overview and documentation
pyproject.toml      Project metadata and dependencies
```

## Setup

To run this project locally:

1. Clone the repository: https://github.com/timgruenecker/Home_Credit_Project.git

2. (Recommended) Create a virtual environment.

3. Install all required packages using the `pyproject.toml`, executing the command: `pip install -e .`

4. The `data/` directory is not included in this repository. You need to download the  
[competition data](https://www.kaggle.com/competitions/home-credit-default-risk/data)
from Kaggle and place the relevant files (e.g., `application_train.csv`, `application_test.csv`, etc.) into a `data/` folder at the project root.



The project uses Python 3.10+ and relies on common packages such as pandas, numpy, scikit-learn, lightgbm, xgboost, catboost, wandb and featuretools.

## Data Description

The main training dataset consists of over 300,000 loan applicants. Each row represents a unique applicant with features including:

- Contract and credit type
- Demographic information
- Employment and income
- Credit history
- Bureau and previous loan data (from supplementary files)

The target variable `TARGET` is binary: 1 indicates a client defaulted on the loan, 0 means they repaid it.

For a full description of the data and additional files, refer to the [Kaggle competition data page](https://www.kaggle.com/competitions/home-credit-default-risk/data).

## Pipeline and Modeling

The modeling pipeline is organized into the following main steps:

1. **Exploratory Data Analysis (EDA)**  
   - Initial insights into the dataset structure and distributions
   - Separate analysis for application train and test data

2. **Feature Engineering**  
   - Manual feature generation and domain-inspired transformations
   - Automated feature generation with Featuretools (Deep Feature Synthesis)
   - Categorical preprocessing and alignment between train/test

3. **Feature Selection**  
   - Automated selection using importance from LightGBM, correlation filtering and SHAP

4. **Hyperparameter Optimization**  
   - Conducted separately for LightGBM, XGBoost, and CatBoost
   - Optimization with Optuna, early stopping and hyperband

5. **Model Training and Evaluation**  
   - Models trained with stratified 5-fold cross-validation
   - Evaluation using ROC AUC on Out-Of-Fold (OOF) predictions

6. **Ensembling**  
   - Generation of OOF predictions for all base models
   - Stacked ensemble using LightGBM as meta-learner
   - Final comparison of ensemble vs. individual models

7. **Logging**  
   - Tracking and Logging with Weights and Biases: [WandB Website](https://wandb.ai/site/)

Each step is documented in a separate notebook or script, ensuring a clear and reproducible workflow.


## Results

The following AUC scores were achieved on the out-of-fold (OOF) validation sets and Kaggle leaderboard. The **total number of features is 63**:

| Model                  | Private Score  | Public Score  |
|------------------------|----------|----------------|
| LightGBM (single)      | 0.79102  | 0.78752        |
| CatBoost (single)      | 0.78895  | 0.78715        |
| XGBoost (single)       | 0.78694  | 0.78368        |
| Stacked Ensemble (Lin) | 0.79006  | 0.78869        |
| Stacked Ensemble (LGB) | 0.79162  | 0.78832        |
| Stacked Ensemble (Cat) | 0.79103  | 0.78928        |
| Stacked Ensemble (XGB) | 0.79169  | 0.78834        |

The LightGBM model already performed strongly on its own. Ensembling provided slight improvements in private score performance, especially with a XGBoost meta-model.


>Based on the **private leaderboard score**, the best-performing model was the stacked ensemble using XGBoost as the meta-learner.


# Notes and Design Decisions:

- The EXT_SOURCE Featues were the most important features from the whole dataset. As some had over 50 % missing values, imputation with separate models was tried. But it did not improve AUC, so fallback strategies were preferred.

- AutoFeat-generated features: Introduced too much noise and led to worse validation performance.

- For Feature Selection the most efficient selection-pipeline was to first include as many features as possible and then filtering with importance, correlation and lastly SHAP.

# Noch Fazit oder Reflektion oder sowas einbauen, dass es nicht so abrupt aufhört 