# src/model_trainer.py

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
import xgboost as xgb
# FIX: Use mean_squared_error (MSE) instead of root_mean_squared_error (RMSE) for compatibility
from sklearn.metrics import mean_squared_error as MSE, r2_score, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
import shap


def evaluate_classification(y_test: np.ndarray, y_pred_proba: np.ndarray, y_pred: np.ndarray) -> dict:
    """Evaluates Frequency model performance."""
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_pred_proba)

    return {
        'AUC': auc,
        'Accuracy': report['accuracy'],
        'Precision (Claim)': report['1']['precision'],
        'Recall (Claim)': report['1']['recall'],
        'F1-Score (Claim)': report['1']['f1-score'],
        'Report': report
    }


def evaluate_regression(y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    """Evaluates Severity model performance."""

    # Calculate RMSE manually using MSE
    mse = MSE(y_test, y_pred)
    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred)

    return {
        'RMSE': rmse,
        'R-squared': r2
    }


def train_and_evaluate(model_type: str, X_train, y_train, X_test, y_test, model_name: str) -> tuple:
    """Trains a model and evaluates it based on its type (Classification/Regression)."""

    # Model Selection
    if model_name == 'LogisticRegression':
        model = LogisticRegression(solver='liblinear', random_state=42)
    elif model_name == 'LinearRegression':
        model = LinearRegression()
    elif model_name == 'RandomForestClassifier':
        model = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1)
    elif model_name == 'RandomForestRegressor':
        model = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1)
    elif model_name == 'XGBClassifier':
        model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False,
                                  eval_metric='logloss', random_state=42, n_jobs=-1)
    elif model_name == 'XGBRegressor':
        model = xgb.XGBRegressor(
            objective='reg:squarederror', random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    print(f"\n--- Training {model_name} ({model_type}) ---")
    model.fit(X_train, y_train)

    if model_type == 'Classification':
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        metrics = evaluate_classification(y_test, y_pred_proba, y_pred)

    elif model_type == 'Regression':
        y_pred = model.predict(X_test)
        metrics = evaluate_regression(y_test, y_pred)

    else:
        raise ValueError("Invalid model_type.")

    return model, metrics


def get_feature_names(preprocessor: ColumnTransformer, features_list: list) -> list:
    """Extracts final feature names after one-hot encoding."""
    # This is a robust way to get OHE feature names
    ohe_names = list(preprocessor.named_transformers_[
                     'cat']['onehot'].get_feature_names_out(features_list))
    # Note: ColumnTransformer re-orders numerical features alphabetically
    num_names = preprocessor.transformers_[0][2]
    return num_names + ohe_names


def run_shap_analysis(model, X_test, preprocessor, features_list: list, top_n: int = 10) -> pd.DataFrame:
    """Performs SHAP analysis for model interpretation."""

    # 1. Prepare Explainer
    explainer = shap.TreeExplainer(model)

    # 2. Calculate SHAP values on a subset of the test set for speed
    X_test_subset = X_test[:1000]
    shap_values = explainer.shap_values(X_test_subset)

    # 3. Aggregate SHAP values
    if isinstance(shap_values, list):
        # For binary classification (XGBClassifier), shap_values returns a list of two arrays.
        shap_values = shap_values[1]  # Focus on the positive class (Claimed=1)

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    # 4. Get readable feature names
    feature_names = get_feature_names(preprocessor, features_list)

    # 5. Create importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Absolute_SHAP': mean_abs_shap
    }).sort_values(by='Mean_Absolute_SHAP', ascending=False)

    return importance_df.head(top_n)
