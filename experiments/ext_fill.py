"""
This script was an experimental attempt to impute missing values in the EXT_SOURCE_1 and EXT_SOURCE_3 columns
using a supervised learning approach. Specifically, a RandomForestRegressor was trained on non-missing samples
to predict missing values based on one-hot encoded features. However, the imputation did not improve downstream
model performance and was therefore discarded in the final pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def load_data(path: Path) -> pd.DataFrame:
    print(f"Loading data from {path}")
    df = pd.read_csv(path)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def prepare_features(df: pd.DataFrame, target_col: str, drop_cols: list) -> (pd.DataFrame, pd.Series, list):
    feat_cols = [c for c in df.columns if c not in ([target_col] + drop_cols)]
    X_full = df[feat_cols]
    X_dummies = pd.get_dummies(X_full, dummy_na=True, drop_first=True)
    y = df[target_col]
    print(f"Feature matrix for {target_col}: {X_dummies.shape[0]} rows, {X_dummies.shape[1]} features after one-hot encoding")
    return X_dummies, y, X_dummies.columns.tolist()


def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    print(f"Training RandomForestRegressor on {len(y_train)} samples and {X_train.shape[1]} features")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    print(f"Validation R² score: {score:.4f}")
    return model


def impute_column(df: pd.DataFrame, col: str, model: RandomForestRegressor, feat_cols: list) -> pd.DataFrame:
    missing_mask = df[col].isna()
    n_missing = missing_mask.sum()
    print(f"Column {col}: {n_missing} missing values")
    if n_missing == 0:
        print(f"No missing values in {col}, skipping.")
        return df

    X_pred_dummies = pd.get_dummies(
        df.loc[missing_mask, [c for c in df.columns if c not in ['SK_ID_CURR', 'EXT_SOURCE_1', 'EXT_SOURCE_3', 'TARGET']]],
        dummy_na=True,
        drop_first=True
    )
    X_pred = pd.DataFrame(X_pred_dummies, index=X_pred_dummies.index)
    X_pred = X_pred.reindex(columns=feat_cols, fill_value=0)
    print(f"Predicting {n_missing} values in {col}")
    preds = model.predict(X_pred)
    df.loc[missing_mask, col] = preds
    print(f"Completed filling for {col}")
    return df


def main():
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / 'data'
    output_dir = script_dir.parent / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = data_dir / 'application_train.csv'
    output_path = output_dir / 'application_train_ext_fill.csv'

    df = load_data(input_path)

    models = {}
    features = {}

    for col in ['EXT_SOURCE_1', 'EXT_SOURCE_3']:
        df_train = df[df[col].notna()].copy()
        X, y, feat_cols = prepare_features(
            df_train,
            col,
            drop_cols=['EXT_SOURCE_1', 'EXT_SOURCE_3', 'SK_ID_CURR', 'TARGET']
        )
        model = train_model(X, y)
        models[col] = model
        features[col] = feat_cols

    for col in ['EXT_SOURCE_1', 'EXT_SOURCE_3']:
        df = impute_column(df, col, models[col], features[col])

    df.to_csv(output_path, index=False)
    print(f"Saved filled file to {output_path}")


if __name__ == '__main__':
    main()
