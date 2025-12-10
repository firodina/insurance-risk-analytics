import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


def prepare_data(df: pd.DataFrame, test_size: float = 0.2) -> dict:
    """
    Prepares and splits data for Frequency (Classification) 
    and Severity (Regression) models with robust type handling.
    """

    df_clean = df.copy()

    # ------------------------------------------------------------
    # 1. Target & Derived Fields
    # ------------------------------------------------------------
    df_clean['ClaimStatus'] = (df_clean['TotalClaims'] > 0).astype(int)
    df_clean['Margin'] = df_clean['TotalPremium'] - df_clean['TotalClaims']

    # ------------------------------------------------------------
    # 2. Vehicle Age
    # ------------------------------------------------------------
    reg_col = next((c for c in ['RegistrationYear', 'VehicleYear', 'registrationyear', 'vehicleyear']
                    if c in df_clean.columns), None)

    if reg_col:
        df_clean['VehicleAge'] = 2025 - \
            pd.to_numeric(df_clean[reg_col], errors='coerce')
        df_clean['VehicleAge'].fillna(
            df_clean['VehicleAge'].median(), inplace=True)
    else:
        print("Warning: No valid registration year found for VehicleAge.")
        df_clean['VehicleAge'] = df_clean.get('VehicleAge', 0)

    # ------------------------------------------------------------
    # 3. Feature List
    # ------------------------------------------------------------
    EXCLUDE_COLS = ['PolicyID', 'TransactionMonth', 'TotalPremium', 'TotalClaims',
                    'ClaimStatus', 'ClaimSeverity', 'Margin']

    FEATURES = [col for col in df_clean.columns if col not in EXCLUDE_COLS]
    X_temp = df_clean[FEATURES].copy()

    # ------------------------------------------------------------
    # 4. Numeric Columns: force float
    # ------------------------------------------------------------
    MANDATORY_NUMERIC_COLS = [
        'RegistrationYear', 'VehicleYear', 'Cylinders', 'cubiccapacity',
        'kilowatts', 'NumberOfDoors', 'SumInsured', 'CalculatedPremiumPerTerm',
        'CapitalOutstanding', 'NumberOfVehiclesInFleet', 'VehicleAge', 'CustomValueEstimate'
    ]

    NUMERIC_FEATURES = []
    for col in MANDATORY_NUMERIC_COLS:
        if col in X_temp.columns:
            X_temp[col] = pd.to_numeric(X_temp[col], errors='coerce')
            NUMERIC_FEATURES.append(col)

    # ------------------------------------------------------------
    # 5. Categorical Columns: force string for all values
    # ------------------------------------------------------------
    CATEGORICAL_FEATURES = [
        col for col in FEATURES if col not in NUMERIC_FEATURES]

    for col in CATEGORICAL_FEATURES:
        X_temp[col] = X_temp[col].apply(lambda x: str(
            x).strip() if pd.notnull(x) else "missing")

    # ------------------------------------------------------------
    # 6. Preprocessing Pipeline
    # ------------------------------------------------------------
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )

    # ------------------------------------------------------------
    # 7. Frequency Model (Classification)
    # ------------------------------------------------------------
    y_freq = df_clean['ClaimStatus']
    X_processed_freq = preprocessor.fit_transform(X_temp)

    X_train_freq, X_test_freq, y_train_freq, y_test_freq = train_test_split(
        X_processed_freq, y_freq,
        test_size=test_size, random_state=42,
        stratify=y_freq
    )

    # ------------------------------------------------------------
    # 8. Severity Model (Regression)
    # ------------------------------------------------------------
    df_sev = df_clean[df_clean['ClaimStatus'] == 1]
    X_sev = X_temp.loc[df_sev.index]
    y_sev = df_sev['TotalClaims']

    X_processed_sev = preprocessor.transform(X_sev)

    X_train_sev, X_test_sev, y_train_sev, y_test_sev = train_test_split(
        X_processed_sev, y_sev,
        test_size=test_size, random_state=42
    )

    # ------------------------------------------------------------
    # 9. Return datasets and preprocessor
    # ------------------------------------------------------------
    return {
        'freq': (X_train_freq, X_test_freq, y_train_freq, y_test_freq),
        'sev': (X_train_sev, X_test_sev, y_train_sev, y_test_sev),
        'preprocessor': preprocessor
    }
