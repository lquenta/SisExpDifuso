"""
Membership functions for fuzzy logic system.
Supports triangular, trapezoidal, and Gaussian functions with extensible architecture.
"""

import numpy as np
from typing import Callable, Dict, Any, Union
from abc import ABC, abstractmethod
from .schema import TriangularMF, TrapezoidalMF, GaussianMF, MembershipFunction


class MembershipFunctionBase(ABC):
    """Base class for membership functions."""
    
    @abstractmethod
    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate membership function at point(s) x."""
        pass
    
    @abstractmethod
    def get_support(self) -> tuple[float, float]:
        """Get support interval [min, max] where function is non-zero."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get function parameters as dictionary."""
        pass


class TriangularMembershipFunction(MembershipFunctionBase):
    """Triangular membership function: μ(x) = max(min((x-a)/(b-a), (c-x)/(c-b)), 0)"""
    
    def __init__(self, a: float, b: float, c: float):
        """
        Initialize triangular membership function.
        
        Args:
            a: Left vertex (x-coordinate)
            b: Peak vertex (x-coordinate) 
            c: Right vertex (x-coordinate)
        """
        if not (a <= b <= c):
            raise ValueError("Triangular vertices must satisfy a <= b <= c")
        
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
    
    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate triangular membership function."""
        x = np.asarray(x)
        
        # Handle edge cases
        if self.a == self.b == self.c:
            return np.where(x == self.a, 1.0, 0.0)
        
        # Left side: (x-a)/(b-a) for a <= x <= b
        left = np.where(
            (self.a < self.b) & (x >= self.a) & (x <= self.b),
            (x - self.a) / (self.b - self.a),
            0.0
        )
        
        # Right side: (c-x)/(c-b) for b <= x <= c
        right = np.where(
            (self.b < self.c) & (x >= self.b) & (x <= self.c),
            (self.c - x) / (self.c - self.b),
            0.0
        )
        
        # Peak at b
        peak = np.where(x == self.b, 1.0, 0.0)
        
        # Combine and take maximum
        result = np.maximum(np.maximum(left, right), peak)
        
        # Ensure non-negative
        result = np.maximum(result, 0.0)
        
        return result if isinstance(x, np.ndarray) else float(result)
    
    def get_support(self) -> tuple[float, float]:
        """Get support interval [a, c]."""
        return (self.a, self.c)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get triangular parameters."""
        return {"type": "tri", "a": self.a, "b": self.b, "c": self.c}


class TrapezoidalMembershipFunction(MembershipFunctionBase):
    """Trapezoidal membership function (future extension)."""
    
    def __init__(self, a: float, b: float, c: float, d: float):
        """
        Initialize trapezoidal membership function.
        
        Args:
            a: Left foot (x-coordinate)
            b: Left shoulder (x-coordinate)
            c: Right shoulder (x-coordinate)
            d: Right foot (x-coordinate)
        """
        if not (a <= b <= c <= d):
            raise ValueError("Trapezoidal vertices must satisfy a <= b <= c <= d")
        
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)
    
    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate trapezoidal membership function."""
        x = np.asarray(x)
        
        # Left ramp: (x-a)/(b-a) for a <= x <= b
        left_ramp = np.where(
            (self.a < self.b) & (x >= self.a) & (x <= self.b),
            (x - self.a) / (self.b - self.a),
            0.0
        )
        
        # Plateau: 1.0 for b <= x <= c
        plateau = np.where(
            (x >= self.b) & (x <= self.c),
            1.0,
            0.0
        )
        
        # Right ramp: (d-x)/(d-c) for c <= x <= d
        right_ramp = np.where(
            (self.c < self.d) & (x >= self.c) & (x <= self.d),
            (self.d - x) / (self.d - self.c),
            0.0
        )
        
        # Combine and take maximum
        result = np.maximum(np.maximum(left_ramp, plateau), right_ramp)
        
        # Ensure non-negative
        result = np.maximum(result, 0.0)
        
        return result if isinstance(x, np.ndarray) else float(result)
    
    def get_support(self) -> tuple[float, float]:
        """Get support interval [a, d]."""
        return (self.a, self.d)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get trapezoidal parameters."""
        return {"type": "trap", "a": self.a, "b": self.b, "c": self.c, "d": self.d}


class GaussianMembershipFunction(MembershipFunctionBase):
    """Gaussian membership function (future extension)."""
    
    def __init__(self, center: float, width: float):
        """
        Initialize Gaussian membership function.
        
        Args:
            center: Center (mean) of the Gaussian
            width: Width (standard deviation) of the Gaussian
        """
        if width <= 0:
            raise ValueError("Gaussian width must be positive")
        
        self.center = float(center)
        self.width = float(width)
    
    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate Gaussian membership function."""
        x = np.asarray(x)
        
        # Gaussian: exp(-0.5 * ((x - center) / width)^2)
        result = np.exp(-0.5 * ((x - self.center) / self.width) ** 2)
        
        return result if isinstance(x, np.ndarray) else float(result)
    
    def get_support(self) -> tuple[float, float]:
        """Get approximate support interval (3-sigma rule)."""
        return (self.center - 3 * self.width, self.center + 3 * self.width)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get Gaussian parameters."""
        return {"type": "gauss", "center": self.center, "width": self.width}


class MembershipFunctionFactory:
    """Factory for creating membership functions from schema objects."""
    
    @staticmethod
    def create(mf_config: MembershipFunction) -> MembershipFunctionBase:
        """
        Create membership function from schema configuration.
        
        Args:
            mf_config: Membership function configuration (TriangularMF, TrapezoidalMF, or GaussianMF)
            
        Returns:
            MembershipFunctionBase: Instantiated membership function
        """
        if isinstance(mf_config, TriangularMF):
            return TriangularMembershipFunction(mf_config.a, mf_config.b, mf_config.c)
        elif isinstance(mf_config, TrapezoidalMF):
            return TrapezoidalMembershipFunction(mf_config.a, mf_config.b, mf_config.c, mf_config.d)
        elif isinstance(mf_config, GaussianMF):
            return GaussianMembershipFunction(mf_config.center, mf_config.width)
        else:
            raise ValueError(f"Unsupported membership function type: {type(mf_config)}")


def create_membership_function(mf_config: MembershipFunction) -> MembershipFunctionBase:
    """Convenience function to create membership function from schema."""
    return MembershipFunctionFactory.create(mf_config)


# Utility functions for membership function analysis

def analyze_universe_coverage(variables: list, universe: tuple[float, float]) -> Dict[str, Any]:
    """
    Analyze coverage of universe by membership functions.
    
    Args:
        variables: List of variables with their terms
        universe: Universe of discourse [min, max]
        
    Returns:
        Dictionary with coverage analysis
    """
    min_univ, max_univ = universe
    
    # Create fine grid for analysis
    x = np.linspace(min_univ, max_univ, 1001)
    
    # Calculate maximum membership across all terms
    max_membership = np.zeros_like(x)
    
    for var in variables:
        for term in var.terms:
            mf = create_membership_function(term.mf)
            membership = mf(x)
            max_membership = np.maximum(max_membership, membership)
    
    # Find gaps (regions with membership < 0.05)
    gaps = []
    in_gap = False
    gap_start = None
    
    for i, mu in enumerate(max_membership):
        if mu < 0.05 and not in_gap:
            in_gap = True
            gap_start = x[i]
        elif mu >= 0.05 and in_gap:
            in_gap = False
            gaps.append((gap_start, x[i]))
    
    # Handle gap at the end
    if in_gap:
        gaps.append((gap_start, x[-1]))
    
    # Calculate coverage percentage
    total_gap_length = sum(end - start for start, end in gaps)
    coverage_percentage = (1.0 - total_gap_length / (max_univ - min_univ)) * 100
    
    return {
        "coverage_percentage": coverage_percentage,
        "gaps": gaps,
        "min_membership": float(np.min(max_membership)),
        "max_membership": float(np.max(max_membership)),
        "has_gaps": len(gaps) > 0,
        "gap_warning": coverage_percentage < 95.0
    }


def analyze_overlap(variables: list) -> Dict[str, Any]:
    """
    Analyze overlap between adjacent membership functions.
    
    Args:
        variables: List of variables with their terms
        
    Returns:
        Dictionary with overlap analysis
    """
    overlap_analysis = {}
    
    for var in variables:
        if len(var.terms) < 2:
            continue
            
        overlaps = []
        
        # Sort terms by their peak positions (for triangular: b parameter)
        sorted_terms = sorted(var.terms, key=lambda t: t.mf.b if hasattr(t.mf, 'b') else 0)
        
        for i in range(len(sorted_terms) - 1):
            term1 = sorted_terms[i]
            term2 = sorted_terms[i + 1]
            
            mf1 = create_membership_function(term1.mf)
            mf2 = create_membership_function(term2.mf)
            
            # Find intersection points
            support1 = mf1.get_support()
            support2 = mf2.get_support()
            
            # Create grid for intersection analysis
            min_x = min(support1[0], support2[0])
            max_x = max(support1[1], support2[1])
            x = np.linspace(min_x, max_x, 1001)
            
            mu1 = mf1(x)
            mu2 = mf2(x)
            
            # Calculate intersection area
            intersection = np.minimum(mu1, mu2)
            intersection_area = np.trapz(intersection, x)
            
            # Calculate individual areas
            area1 = np.trapz(mu1, x)
            area2 = np.trapz(mu2, x)
            
            # Normalized overlap (intersection / average area)
            avg_area = (area1 + area2) / 2
            normalized_overlap = intersection_area / avg_area if avg_area > 0 else 0
            
            overlaps.append({
                "term1": term1.label,
                "term2": term2.label,
                "intersection_area": float(intersection_area),
                "normalized_overlap": float(normalized_overlap),
                "overlap_ratio": float(intersection_area / avg_area) if avg_area > 0 else 0
            })
        
        overlap_analysis[var.name] = overlaps
    
    return overlap_analysis
