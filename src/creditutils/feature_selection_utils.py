from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

def evaluate_auc(X, y, model, cv, name=None, verbose=True):
    """
    Evaluate a model using cross-validated AUC.

    Parameters:
        X (DataFrame): Feature matrix
        y (Series): Target vector
        model: scikit-learn compatible model
        cv: cross-validation strategy (e.g. StratifiedKFold)
        name (str): Optional name to print
        verbose (bool): If True, prints the result

    Returns:
        mean_auc (float), std_auc (float)
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    if verbose:
        label = f"[{name}] " if name else ""
        print(f"{label}AUC: {scores.mean():.5f} ± {scores.std():.5f}")
    return scores.mean(), scores.std()
