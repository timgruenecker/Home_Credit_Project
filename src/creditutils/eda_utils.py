# src/eda_utils.py

import os
import pandas as pd
import matplotlib.pyplot as plt

def load_datasets(data_dir, tables):
    """
    Load datasets from Parquet if available, else from CSV (and convert to Parquet).
    
    Parameters:
        data_dir (str): Path to the directory with .csv/.parquet files
        tables (list): List of table names (without extension)

    Returns:
        dict: Dictionary of DataFrames {table_name: DataFrame}
    """
    datasets = {}

    for name in tables:
        csv_path = os.path.join(data_dir, f"{name}.csv")
        parquet_path = os.path.join(data_dir, f"{name}.parquet")

        if os.path.exists(parquet_path):
            df = pd.read_parquet(parquet_path)
            print(f"Loaded {name} from Parquet")
        elif os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df.to_parquet(parquet_path, index=False)
            print(f"Converted {name} from CSV to Parquet")
        else:
            raise FileNotFoundError(f"Neither CSV nor Parquet found for {name}")

        datasets[name] = df

    return datasets

def show_basic_info(df, head=3):
    """
    Print shape, dtype counts, and show first rows.
    """
    print(f"  shape: {df.shape}")
    dtypes = df.dtypes.value_counts().to_dict()
    print(f"  dtypes: {dtypes}")
    display(df.head(head))

def missing_value_summary(df, top_n=10):
    """
    Print and plot the top_n columns by missing-value fraction.
    """
    na = df.isnull().mean().sort_values(ascending=False)
    top = na.head(top_n)
    print(top.to_frame(name='missing_ratio'))
    top.plot.barh(title='Missing Ratio', figsize=(6, top_n*0.3))
    plt.xlabel('Fraction missing')
    plt.show()

def zero_value_summary(df, top_n=10):
    """
    Print and plot the top_n columns by zero-value fraction.
    """
    zeros = (df == 0).mean().sort_values(ascending=False)
    top = zeros.head(top_n)
    print(top.to_frame(name='zero_ratio'))
    top.plot.barh(title='Zero Value Ratio', figsize=(6, top_n*0.3))
    plt.xlabel('Fraction zeros')
    plt.show()
