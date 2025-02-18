import optuna
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from joblib import dump
import os
import yaml

BASE_DIR = os.path.join(os.getcwd(), "../")

def objective(trial, X_train: pd.DataFrame, y_train: pd.Series):
    # Define the hyperparameter search space.
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "num_leaves": trial.suggest_int("num_leaves", 15, 150),
        "random_state": 42
    }
    
    # Create a LightGBM classifier with the suggested hyperparameters.
    model = lgb.LGBMClassifier(**param, n_jobs=-1)
    
    # Use StratifiedKFold for robust cross-validation.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, scoring="roc_auc", cv=cv, n_jobs=-1)
    
    # Return the mean ROC AUC score.
    return scores.mean()

def finetune_model(X_train: pd.DataFrame, y_train: pd.Series, model_output_name: str = 'best_model.pkl', n_trials: int = 50):
    
    model_output_path = os.path.join(BASE_DIR, 'artifacts', model_output_name)

    """
    Finetunes a LightGBM classifier using Optuna and saves the best model.
    
    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        model_output_path (str): Path to save the best model.
        n_trials (int): Number of Optuna trials.
        
    Returns:
        best_model: The best LightGBM model found.
        best_params: The best hyperparameters.
    """
    # Create an Optuna study to maximize the ROC AUC score.
    study = optuna.create_study(
        study_name="lgbm_finetuning",
        direction="maximize",
        )
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_jobs=4,  n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_trial.params
    print("Best Parameters:", best_params)
    
    # Add fixed parameters.
    best_params["random_state"] = 42
    
    # Train the final LightGBM model with the best hyperparameters.
    best_model = lgb.LGBMClassifier(**best_params, n_jobs=-1,class_weight='balanced')
    best_model.fit(X_train, y_train)
    
    # Save the best model.
    dump(best_model, model_output_path)
    print(f"Best model saved to {model_output_path}")
    
    return best_model, best_params

if __name__ == '__main__':
    from data.load_data import load_data
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "processed_data.csv"))
    constants_path = os.path.join(BASE_DIR, "src", "utils", "constants.yaml")
    with open(constants_path, "r") as file:
        constants = yaml.safe_load(file)
    FEATURES_TO_USE = constants["FEATURES_TO_USE"]
    X = df[FEATURES_TO_USE]
    y = df["fraude"]