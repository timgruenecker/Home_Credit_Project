import numpy as np
import logging
from sklearn.metrics import roc_auc_score
from optuna.exceptions import TrialPruned
from lightgbm import LGBMClassifier

def run_oof_cv(
    model,
    X,
    y,
    cv,
    trial=None,
    log_prefix="HPO",
    fit_params=None
):
    """Run 5-fold CV, report to Optuna, log fold AUCs and average."""
    fit_params = fit_params or {}
    scores = []

    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val,   y_val   = X.iloc[valid_idx], y.iloc[valid_idx]

        # prepare per-fold arguments
        params = fit_params.copy()
        params['eval_set'] = [(X_val, y_val)]

        # drop unsupported verbose for LGBMClassifier
        if isinstance(model, LGBMClassifier) and 'verbose' in params:
            params.pop('verbose')

        model.fit(X_train, y_train, **params)

        preds = model.predict_proba(X_val)[:, 1]
        fold_score = roc_auc_score(y_val, preds)
        scores.append(fold_score)
        avg_score = np.mean(scores)

        if trial is not None:
            trial.report(avg_score, fold_idx)
            if trial.should_prune():
                logging.info(f"{log_prefix} Trial {trial.number} pruned at fold {fold_idx}")
                raise TrialPruned()

        logging.info(
            f"{log_prefix} Trial {trial.number:>3} | Fold {fold_idx} | "
            f"AUC {fold_score:.5f} | Avg@{fold_idx} {avg_score:.5f}"
        )

    return np.mean(scores)
