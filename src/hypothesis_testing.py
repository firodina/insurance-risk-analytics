# src/hypothesis_testing.py

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind, f_oneway


def prepare_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Claim Frequency and Margin per policy."""
    df_metrics = df.copy()

    # 1. Claim Status: Essential for Frequency and Margin calculation
    df_metrics['ClaimStatus'] = np.where(df_metrics['TotalClaims'] > 0, 1, 0)

    # 2. Margin (Profit/Loss per Policy)
    df_metrics['Margin'] = df_metrics['TotalPremium'] - \
        df_metrics['TotalClaims']

    # 3. Claim Severity (Only for policies with claims)
    df_metrics['ClaimSeverity'] = np.where(df_metrics['ClaimStatus'] == 1,
                                           df_metrics['TotalClaims'], np.nan)

    return df_metrics


def run_chi_squared_test(df: pd.DataFrame, feature: str, alpha: float = 0.05) -> dict:
    """Tests H0: No difference in Claim Frequency across groups."""
    print(f"\n--- Testing Claim Frequency for: {feature} ---")

    # Create contingency table: [Feature Group] x [Claimed (1), No Claim (0)]
    contingency_table = pd.crosstab(df[feature], df['ClaimStatus'])

    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    result = {
        'Test': 'Chi-Squared (Frequency)',
        'Feature': feature,
        'p_value': p_value,
        'Reject_H0': p_value < alpha,
        'Chi2_Stat': chi2
    }
    return result


def run_mean_test(df: pd.DataFrame, feature: str, metric: str, groups: list, alpha: float = 0.05) -> dict:
    """Tests H0: No difference in mean (Severity or Margin) between two groups."""
    print(f"--- Testing Mean {metric} for: {feature} (Groups: {groups}) ---")

    group_a = df[df[feature] == groups[0]][metric].dropna()
    group_b = df[df[feature] == groups[1]][metric].dropna()

    # T-test for two independent samples. Use equal_var=False (Welch's T-test)
    # as variances are rarely equal in insurance data.
    t_stat, p_value = ttest_ind(group_a, group_b, equal_var=False)

    result = {
        'Test': f"T-Test ({metric})",
        'Feature': feature,
        'Group_A': groups[0],
        'Group_B': groups[1],
        'Mean_A': group_a.mean(),
        'Mean_B': group_b.mean(),
        'p_value': p_value,
        'Reject_H0': p_value < alpha,
        'T_Stat': t_stat
    }
    return result


def run_anova_test(df: pd.DataFrame, feature: str, metric: str, groups: list, alpha: float = 0.05) -> dict:
    """Tests H0: No difference in mean (Severity) across multiple groups (ANOVA)."""
    print(f"--- Testing Mean {metric} for: {feature} (Multiple Groups) ---")

    # Prepare list of series, one for each group
    group_series = [df[df[feature] == group][metric].dropna()
                    for group in groups]

    # Filter out empty groups if any
    group_series = [s for s in group_series if not s.empty]

    f_stat, p_value = f_oneway(*group_series)

    result = {
        'Test': f"ANOVA ({metric})",
        'Feature': feature,
        'Groups_Tested': groups,
        'p_value': p_value,
        'Reject_H0': p_value < alpha,
        'F_Stat': f_stat
    }
    return result
