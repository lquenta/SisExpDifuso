"""
Defuzzification methods for fuzzy inference systems.
Implements centroid, bisector, and mean of maxima methods.
"""

import numpy as np
from typing import Tuple, Union, Optional
from enum import Enum


class DefuzzificationMethod(Enum):
    """Available defuzzification methods."""
    CENTROID = "centroid"
    BISECTOR = "bisector"
    MEAN_OF_MAXIMA = "mean_of_maxima"


class Defuzzifier:
    """Defuzzification engine for fuzzy outputs."""
    
    def __init__(self, method: DefuzzificationMethod = DefuzzificationMethod.CENTROID):
        """
        Initialize defuzzifier.
        
        Args:
            method: Defuzzification method to use
        """
        self.method = method
    
    def defuzzify(self, x: np.ndarray, membership: np.ndarray, 
                  universe: Optional[Tuple[float, float]] = None) -> float:
        """
        Defuzzify aggregated membership function.
        
        Args:
            x: Input domain points
            membership: Aggregated membership values
            universe: Universe of discourse (min, max) for validation
            
        Returns:
            Defuzzified crisp value
            
        Raises:
            ValueError: If inputs are invalid or method fails
        """
        # Validate inputs
        if len(x) != len(membership):
            raise ValueError("x and membership arrays must have same length")
        
        if len(x) == 0:
            raise ValueError("Input arrays cannot be empty")
        
        # Ensure membership values are in [0, 1]
        membership = np.clip(membership, 0.0, 1.0)
        
        # Check if membership function is all zeros
        if np.all(membership == 0):
            # Return center of universe if provided, otherwise center of x range
            if universe:
                return (universe[0] + universe[1]) / 2
            else:
                return (np.min(x) + np.max(x)) / 2
        
        # Apply selected defuzzification method
        if self.method == DefuzzificationMethod.CENTROID:
            return self._centroid(x, membership)
        elif self.method == DefuzzificationMethod.BISECTOR:
            return self._bisector(x, membership)
        elif self.method == DefuzzificationMethod.MEAN_OF_MAXIMA:
            return self._mean_of_maxima(x, membership)
        else:
            raise ValueError(f"Unknown defuzzification method: {self.method}")
    
    def _centroid(self, x: np.ndarray, membership: np.ndarray) -> float:
        """
        Centroid defuzzification: y* = ∫ y * μ(y) dy / ∫ μ(y) dy
        
        Args:
            x: Input domain points
            membership: Membership values
            
        Returns:
            Centroid value
        """
        # Calculate numerator: ∫ y * μ(y) dy
        numerator = np.trapz(x * membership, x)
        
        # Calculate denominator: ∫ μ(y) dy
        denominator = np.trapz(membership, x)
        
        if denominator == 0:
            # Fallback to center of x range
            return (np.min(x) + np.max(x)) / 2
        
        return numerator / denominator
    
    def _bisector(self, x: np.ndarray, membership: np.ndarray) -> float:
        """
        Bisector defuzzification: y* such that ∫[min, y*] μ(y) dy = ∫[y*, max] μ(y) dy
        
        Args:
            x: Input domain points
            membership: Membership values
            
        Returns:
            Bisector value
        """
        # Calculate cumulative integral
        cumulative = np.cumsum(membership) * (x[1] - x[0])  # Approximate integration
        
        # Total area
        total_area = cumulative[-1]
        
        if total_area == 0:
            return (np.min(x) + np.max(x)) / 2
        
        # Find bisector point
        target_area = total_area / 2
        
        # Find index where cumulative area reaches target
        bisector_idx = np.searchsorted(cumulative, target_area)
        
        # Handle edge cases
        if bisector_idx == 0:
            return x[0]
        elif bisector_idx >= len(x):
            return x[-1]
        
        # Interpolate between points for more accurate result
        if bisector_idx < len(cumulative):
            # Linear interpolation
            area_before = cumulative[bisector_idx - 1]
            area_after = cumulative[bisector_idx]
            
            if area_after > area_before:
                # Interpolate between x[bisector_idx-1] and x[bisector_idx]
                weight = (target_area - area_before) / (area_after - area_before)
                return x[bisector_idx - 1] + weight * (x[bisector_idx] - x[bisector_idx - 1])
            else:
                return x[bisector_idx]
        else:
            return x[-1]
    
    def _mean_of_maxima(self, x: np.ndarray, membership: np.ndarray) -> float:
        """
        Mean of maxima defuzzification: y* = mean of all x where μ(x) = max(μ)
        
        Args:
            x: Input domain points
            membership: Membership values
            
        Returns:
            Mean of maxima value
        """
        # Find maximum membership value
        max_membership = np.max(membership)
        
        if max_membership == 0:
            return (np.min(x) + np.max(x)) / 2
        
        # Find all points with maximum membership
        max_indices = np.where(membership == max_membership)[0]
        
        # Return mean of x values at maximum membership points
        return np.mean(x[max_indices])


def defuzzify(method: Union[str, DefuzzificationMethod], x: np.ndarray, 
              membership: np.ndarray, universe: Optional[Tuple[float, float]] = None) -> float:
    """
    Convenience function for defuzzification.
    
    Args:
        method: Defuzzification method name or enum
        x: Input domain points
        membership: Aggregated membership values
        universe: Universe of discourse (min, max)
        
    Returns:
        Defuzzified crisp value
    """
    if isinstance(method, str):
        method = DefuzzificationMethod(method)
    
    defuzzifier = Defuzzifier(method)
    return defuzzifier.defuzzify(x, membership, universe)


# Utility functions for defuzzification analysis

def compare_defuzzification_methods(x: np.ndarray, membership: np.ndarray, 
                                  universe: Optional[Tuple[float, float]] = None) -> dict:
    """
    Compare all defuzzification methods on the same membership function.
    
    Args:
        x: Input domain points
        membership: Aggregated membership values
        universe: Universe of discourse (min, max)
        
    Returns:
        Dictionary with results from all methods
    """
    results = {}
    
    for method in DefuzzificationMethod:
        try:
            defuzzifier = Defuzzifier(method)
            result = defuzzifier.defuzzify(x, membership, universe)
            results[method.value] = {
                "value": result,
                "success": True,
                "error": None
            }
        except Exception as e:
            results[method.value] = {
                "value": None,
                "success": False,
                "error": str(e)
            }
    
    return results


def analyze_defuzzification_sensitivity(x: np.ndarray, membership: np.ndarray,
                                      noise_level: float = 0.01) -> dict:
    """
    Analyze sensitivity of defuzzification to small perturbations.
    
    Args:
        x: Input domain points
        membership: Aggregated membership values
        noise_level: Standard deviation of Gaussian noise to add
        
    Returns:
        Dictionary with sensitivity analysis
    """
    # Get baseline results
    baseline = compare_defuzzification_methods(x, membership)
    
    # Test with noise
    noise_results = []
    n_trials = 10
    
    for _ in range(n_trials):
        # Add Gaussian noise
        noisy_membership = membership + np.random.normal(0, noise_level, len(membership))
        noisy_membership = np.clip(noisy_membership, 0.0, 1.0)
        
        # Get results with noise
        noisy_result = compare_defuzzification_methods(x, noisy_membership)
        noise_results.append(noisy_result)
    
    # Calculate sensitivity metrics
    sensitivity = {}
    
    for method in DefuzzificationMethod:
        if baseline[method.value]["success"]:
            baseline_value = baseline[method.value]["value"]
            noisy_values = []
            
            for result in noise_results:
                if result[method.value]["success"]:
                    noisy_values.append(result[method.value]["value"])
            
            if noisy_values:
                # Calculate standard deviation of results
                std_dev = np.std(noisy_values)
                mean_deviation = np.mean(np.abs(np.array(noisy_values) - baseline_value))
                
                sensitivity[method.value] = {
                    "baseline": baseline_value,
                    "std_deviation": std_dev,
                    "mean_deviation": mean_deviation,
                    "relative_sensitivity": std_dev / abs(baseline_value) if baseline_value != 0 else float('inf'),
                    "success_rate": len(noisy_values) / n_trials
                }
            else:
                sensitivity[method.value] = {
                    "baseline": baseline_value,
                    "std_deviation": None,
                    "mean_deviation": None,
                    "relative_sensitivity": None,
                    "success_rate": 0.0
                }
        else:
            sensitivity[method.value] = {
                "baseline": None,
                "std_deviation": None,
                "mean_deviation": None,
                "relative_sensitivity": None,
                "success_rate": 0.0
            }
    
    return sensitivity


def validate_defuzzification_inputs(x: np.ndarray, membership: np.ndarray) -> list:
    """
    Validate inputs for defuzzification.
    
    Args:
        x: Input domain points
        membership: Membership values
        
    Returns:
        List of validation warnings/errors
    """
    warnings = []
    
    # Check array lengths
    if len(x) != len(membership):
        warnings.append("x and membership arrays have different lengths")
    
    # Check for empty arrays
    if len(x) == 0:
        warnings.append("Input arrays are empty")
        return warnings
    
    # Check for NaN or infinite values
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        warnings.append("x array contains NaN or infinite values")
    
    if np.any(np.isnan(membership)) or np.any(np.isinf(membership)):
        warnings.append("membership array contains NaN or infinite values")
    
    # Check membership value range
    if np.any(membership < 0) or np.any(membership > 1):
        warnings.append("membership values outside [0, 1] range")
    
    # Check for monotonic x values
    if not np.all(np.diff(x) > 0):
        warnings.append("x values are not strictly increasing")
    
    # Check for zero membership
    if np.all(membership == 0):
        warnings.append("All membership values are zero")
    
    return warnings
