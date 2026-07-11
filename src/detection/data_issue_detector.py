import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict

class DataIssueDetector:
    """Detector for identifying covariate and schema drift between reference and production datasets."""

    def detect_covariate_drift(self, reference_df: pd.DataFrame, production_df: pd.DataFrame) -> Dict:
        """Check each overlapping feature for distribution shift/drift.
        
        Uses the Kolmogorov-Smirnov (KS) test for numerical columns and the
        Chi-Squared Contingency test for categorical columns.
        """
        results = {}
        for col in reference_df.columns:
            if col not in production_df.columns:
                continue

            ref_col = reference_df[col].dropna()
            prod_col = production_df[col].dropna()

            # Skip if either is completely empty
            if ref_col.empty or prod_col.empty:
                continue

            # Determine if numerical
            if pd.api.types.is_numeric_dtype(reference_df[col]):
                # Numerical features: KS two-sample test
                try:
                    stat, p = stats.ks_2samp(ref_col, prod_col)
                    
                    # Handle possible NaN outcomes from stats test
                    if np.isnan(stat) or np.isnan(p):
                        stat, p = 0.0, 1.0

                    results[col] = {
                        "test": "KS",
                        "statistic": float(stat),
                        "p_value": float(p),
                        "drift": bool(p < 0.05)
                    }
                except Exception:
                    results[col] = {
                        "test": "KS",
                        "statistic": 0.0,
                        "p_value": 1.0,
                        "drift": False
                    }
            else:
                # Categorical features: chi-squared contingency test.
                ref_counts = ref_col.value_counts()
                prod_counts = prod_col.value_counts()
                all_categories = ref_counts.index.union(prod_counts.index)

                if len(all_categories) <= 1:
                    results[col] = {
                        "test": "chi2_contingency",
                        "statistic": 0.0,
                        "p_value": 1.0,
                        "dof": 0,
                        "drift": False
                    }
                    continue

                # Build aligned contingency dataframe
                contingency = pd.DataFrame({
                    'reference': ref_counts,
                    'production': prod_counts
                }).reindex(all_categories).fillna(0)

                try:
                    chi2, p, dof, _ = stats.chi2_contingency(contingency.values)
                    
                    if np.isnan(chi2) or np.isnan(p):
                        chi2, p, dof = 0.0, 1.0, 0

                    results[col] = {
                        "test": "chi2_contingency",
                        "statistic": float(chi2),
                        "p_value": float(p),
                        "dof": int(dof),
                        "drift": bool(p < 0.05)
                    }
                except Exception:
                    results[col] = {
                        "test": "chi2_contingency",
                        "statistic": 0.0,
                        "p_value": 1.0,
                        "dof": 0,
                        "drift": False
                    }
        return results

    def detect_schema_drift(self, reference_schema: Dict[str, str], production_schema: Dict[str, str]) -> Dict:
        """Detect missing columns, new columns, and type changes between schemas."""
        ref_cols = set(reference_schema.keys())
        prod_cols = set(production_schema.keys())
        return {
            "missing_columns": list(ref_cols - prod_cols),
            "new_columns": list(prod_cols - ref_cols),
            "type_changes": {
                col: (str(reference_schema[col]), str(production_schema[col]))
                for col in (ref_cols & prod_cols)
                if reference_schema[col] != production_schema[col]
            }
        }
