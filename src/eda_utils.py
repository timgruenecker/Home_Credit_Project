# src/eda_utils.py

import os
import pandas as pd
import matplotlib.pyplot as plt

def load_datasets(data_dir):
    """
    Load every CSV in data_dir into a dict of DataFrames.
    Key = filename without extension.
    """
    dfs = {}
    for fname in os.listdir(data_dir):
        if fname.lower().endswith('.csv'):
            key = fname.replace('.csv', '')
            path = os.path.join(data_dir, fname)
            dfs[key] = pd.read_csv(path)
    return dfs

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
