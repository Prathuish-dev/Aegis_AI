import numpy as np
from scipy import stats
from typing import Union, List, Dict

class DriftDetector:
    """Statistical drift detector implementing Kolmogorov-Smirnov test and Population Stability Index."""

    def __init__(self, alpha: float = 0.05):
        """Initializes the DriftDetector with a significance level alpha."""
        self.alpha = alpha

    def detect_feature_drift(self, reference: Union[List, np.ndarray], 
                             production: Union[List, np.ndarray]) -> Dict:
        """Kolmogorov-Smirnov two-sample test for continuous feature drift.
        
        Args:
            reference: Baseline dataset (list or numpy array).
            production: Comparison dataset (list or numpy array).
            
        Returns:
            Dictionary containing:
                - "drift_detected": bool
                - "ks_statistic": float
                - "p_value": float
                - "severity": str ("none", "moderate", "severe")
        """
        ref_arr = np.asarray(reference, dtype=float)
        prod_arr = np.asarray(production, dtype=float)
        
        # Remove NaNs
        ref_arr = ref_arr[~np.isnan(ref_arr)]
        prod_arr = prod_arr[~np.isnan(prod_arr)]
        
        # Check for empty inputs
        if ref_arr.size == 0 or prod_arr.size == 0:
            return {
                "drift_detected": False,
                "ks_statistic": 0.0,
                "p_value": 1.0,
                "severity": "none"
            }
            
        # Run KS test
        ks_stat, p_value = stats.ks_2samp(ref_arr, prod_arr)
        
        # Handle nan result in scipy test (e.g. from single-value or zero variance arrays)
        if np.isnan(p_value) or np.isnan(ks_stat):
            return {
                "drift_detected": False,
                "ks_statistic": 0.0,
                "p_value": 1.0,
                "severity": "none"
            }
            
        return {
            "drift_detected": bool(p_value < self.alpha),
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
            "severity": self._severity(p_value)
        }

    def compute_psi(self, reference: Union[List, np.ndarray], 
                    production: Union[List, np.ndarray], 
                    bins: int = 10) -> Dict:
        """Population Stability Index (PSI) for measuring distribution shift.
        
        IMPORTANT:
        1. Bin edges are computed on reference data and reused for production data.
        2. Proportions are used (sums to 1.0) instead of density.
        
        Args:
            reference: Baseline dataset (list or numpy array).
            production: Comparison dataset (list or numpy array).
            bins: Number of bins to use (default: 10).
            
        Returns:
            Dictionary containing:
                - "psi": float
                - "severity": str ("none", "moderate", "severe")
        """
        ref_arr = np.asarray(reference, dtype=float)
        prod_arr = np.asarray(production, dtype=float)
        
        # Remove NaNs
        ref_arr = ref_arr[~np.isnan(ref_arr)]
        prod_arr = prod_arr[~np.isnan(prod_arr)]
        
        # Check for empty inputs
        if ref_arr.size == 0 or prod_arr.size == 0:
            return {
                "psi": 0.0,
                "severity": "none"
            }

        # Step 1: derive bin edges from the reference distribution
        ref_counts, bin_edges = np.histogram(ref_arr, bins=bins)
        
        # Step 2: clip production values to be within bin edges to avoid out-of-bounds discarding
        # and ensure proportions sum to 1.0
        prod_clipped = np.clip(prod_arr, bin_edges[0], bin_edges[-1])
        prod_counts, _ = np.histogram(prod_clipped, bins=bin_edges)

        # Step 3: convert counts to proportions (sum = 1.0)
        ref_pct = ref_counts / len(ref_arr)
        prod_pct = prod_counts / len(prod_arr)

        # Step 4: avoid log(0) with a small epsilon
        ref_pct = np.where(ref_pct == 0, 1e-4, ref_pct)
        prod_pct = np.where(prod_pct == 0, 1e-4, prod_pct)

        # Compute PSI
        psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))

        # Handle nan result (if any)
        if np.isnan(psi):
            return {
                "psi": 0.0,
                "severity": "none"
            }

        severity = "none"
        if psi < 0.1:
            severity = "none"
        elif psi < 0.25:
            severity = "moderate"
        else:
            severity = "severe"

        return {
            "psi": float(psi),
            "severity": severity
        }

    def _severity(self, p_value: float) -> str:
        """Helper to determine the severity based on p-value of KS test."""
        if p_value < 0.01:
            return "severe"
        elif p_value < self.alpha:
            return "moderate"
        else:
            return "none"
