import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from boruta import BorutaPy

def correlation_filter(df: pd.DataFrame, threshold: float = 0.9) -> list:
    """
    Returns a list of features to drop due to high correlation.
    """
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return to_drop

def rfe_feature_selection(X: pd.DataFrame, y: pd.Series, cv: int = 3, scoring: str = 'roc_auc') -> tuple:
    """
    Returns a tuple containing:
      - a list of selected feature names using RFECV with RandomForestClassifier,
      - the fitted RFECV object for further inspection.
      
    Parameters:
      X (pd.DataFrame): Feature set.
      y (pd.Series): Target variable.
      cv (int): Number of cross-validation folds.
      scoring (str): Scoring metric for RFECV.
    """
    rf = RandomForestClassifier(n_estimators=300, max_depth=7, random_state=42, n_jobs=-1)
    rfecv = RFECV(estimator=rf, step=1, cv=cv, scoring=scoring, verbose=2)
    rfecv.fit(X, y)
    selected_features = list(X.columns[rfecv.support_])
    return selected_features, rfecv

def boruta_feature_selection(X: pd.DataFrame, y: pd.Series, max_iter: int = 100, random_state: int = 42) -> list:
    """
    Returns a list of selected features using the Boruta algorithm with a RandomForestClassifier.
    """
    rf = RandomForestClassifier(n_estimators=300, max_depth=7, random_state=random_state, n_jobs=-1)
    boruta_selector = BorutaPy(estimator=rf, n_estimators='auto', verbose=2, random_state=random_state, max_iter=max_iter)
    boruta_selector.fit(X.values, y.values)
    selected_features = list(X.columns[boruta_selector.support_])
    return selected_features
