import pandas as pd
import numpy as np
import scipy.stats
from typing import Dict, Any

class DataIssueDetector:
    """Detector for data quality issues, including covariate distribution drift and schema drift."""

    def detect_covariate_drift(self, reference_df: pd.DataFrame, production_df: pd.DataFrame) -> Dict[str, Any]:
        """Checks numerical and categorical features for statistical distribution drift.
        
        Uses two-sample Kolmogorov-Smirnov test for numerical features, and
        Chi-squared contingency test for categorical features.
        
        Args:
            reference_df: Baseline training or historical dataframe.
            production_df: Observed production dataframe to check.
            
        Returns:
            Dictionary mapping feature column names to their drift analysis results.
        """
        results = {}
        if reference_df.empty or production_df.empty:
            return results

        # Only check columns that exist in both dataframes
        shared_cols = reference_df.columns.intersection(production_df.columns)
        
        for col in shared_cols:
            # Check numerical types
            is_numeric = pd.api.types.is_numeric_dtype(reference_df[col]) and pd.api.types.is_numeric_dtype(production_df[col])
            
            if is_numeric:
                ref_clean = reference_df[col].dropna()
                prod_clean = production_df[col].dropna()
                
                # Check for empty samples
                if ref_clean.empty or prod_clean.empty:
                    continue
                
                # Run KS test
                stat, p = scipy.stats.ks_2samp(ref_clean, prod_clean)
                
                # Check for NaN results (e.g. all values identical)
                if np.isnan(p) or np.isnan(stat):
                    results[col] = {
                        "test": "KS",
                        "statistic": 0.0,
                        "p_value": 1.0,
                        "drift": False
                    }
                else:
                    results[col] = {
                        "test": "KS",
                        "statistic": float(stat),
                        "p_value": float(p),
                        "drift": bool(p < 0.05)
                    }
            else:
                ref_clean = reference_df[col].dropna()
                prod_clean = production_df[col].dropna()
                
                # Check for empty samples
                if ref_clean.empty or prod_clean.empty:
                    continue
                
                # Build contingency table for Chi-squared test
                ref_counts = ref_clean.value_counts()
                prod_counts = prod_clean.value_counts()
                
                all_categories = ref_counts.index.union(prod_counts.index)
                if len(all_categories) <= 1:
                    # Cannot perform chi2 test with 1 or 0 categories
                    results[col] = {
                        "test": "chi2_contingency",
                        "statistic": 0.0,
                        "p_value": 1.0,
                        "dof": 0,
                        "drift": False
                    }
                    continue
                
                contingency = pd.DataFrame({
                    "reference": ref_counts,
                    "production": prod_counts
                }).reindex(all_categories).fillna(0)
                
                try:
                    chi2, p, dof, _ = scipy.stats.chi2_contingency(contingency.values)
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

    def detect_schema_drift(self, reference_schema: Dict[str, Any], production_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Compares schemas to identify missing columns, new columns, and type mismatches.
        
        Args:
            reference_schema: Dictionary mapping column names to dtype strings (e.g. {'age': 'int64'}).
            production_schema: Observed schema dictionary mapping column names to dtype strings.
            
        Returns:
            Dictionary containing 'missing_columns', 'new_columns', and 'type_changes'.
        """
        ref_cols = set(reference_schema.keys())
        prod_cols = set(production_schema.keys())
        
        type_changes = {}
        for col in ref_cols.intersection(prod_cols):
            ref_type = str(reference_schema[col])
            prod_type = str(production_schema[col])
            if ref_type != prod_type:
                type_changes[col] = (ref_type, prod_type)
                
        return {
            "missing_columns": list(ref_cols - prod_cols),
            "new_columns": list(prod_cols - ref_cols),
            "type_changes": type_changes
        }
