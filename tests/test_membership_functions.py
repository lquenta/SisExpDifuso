"""
Tests for membership functions.
"""

import pytest
import numpy as np

from fuzzy_panel.core.mfuncs import (
    TriangularMembershipFunction, TrapezoidalMembershipFunction, 
    GaussianMembershipFunction, MembershipFunctionFactory,
    create_membership_function, analyze_universe_coverage, analyze_overlap
)
from fuzzy_panel.core.schema import TriangularMF, TrapezoidalMF, GaussianMF, Variable, Term


class TestTriangularMembershipFunction:
    """Test triangular membership function."""
    
    def test_triangular_evaluation(self):
        """Test triangular membership function evaluation."""
        mf = TriangularMembershipFunction(a=0.0, b=5.0, c=10.0)
        
        # Test peak
        assert mf(5.0) == 1.0
        
        # Test boundaries
        assert mf(0.0) == 0.0
        assert mf(10.0) == 0.0
        
        # Test middle points
        assert mf(2.5) == 0.5
        assert mf(7.5) == 0.5
        
        # Test outside support
        assert mf(-1.0) == 0.0
        assert mf(11.0) == 0.0
    
    def test_triangular_array_evaluation(self):
        """Test triangular membership function with array input."""
        mf = TriangularMembershipFunction(a=0.0, b=5.0, c=10.0)
        x = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
        y = mf(x)
        
        expected = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
        np.testing.assert_array_almost_equal(y, expected)
    
    def test_triangular_edge_cases(self):
        """Test triangular membership function edge cases."""
        # Degenerate triangle (a == b == c)
        mf = TriangularMembershipFunction(a=5.0, b=5.0, c=5.0)
        assert mf(5.0) == 1.0
        assert mf(4.0) == 0.0
        assert mf(6.0) == 0.0
        
        # Left triangle (a == b)
        mf = TriangularMembershipFunction(a=0.0, b=0.0, c=10.0)
        assert mf(0.0) == 1.0
        assert mf(5.0) == 0.5
        assert mf(10.0) == 0.0
        
        # Right triangle (b == c)
        mf = TriangularMembershipFunction(a=0.0, b=10.0, c=10.0)
        assert mf(0.0) == 0.0
        assert mf(5.0) == 0.5
        assert mf(10.0) == 1.0
    
    def test_triangular_support(self):
        """Test triangular membership function support."""
        mf = TriangularMembershipFunction(a=2.0, b=5.0, c=8.0)
        support = mf.get_support()
        assert support == (2.0, 8.0)
    
    def test_triangular_parameters(self):
        """Test triangular membership function parameters."""
        mf = TriangularMembershipFunction(a=2.0, b=5.0, c=8.0)
        params = mf.get_parameters()
        assert params == {"type": "tri", "a": 2.0, "b": 5.0, "c": 8.0}


class TestTrapezoidalMembershipFunction:
    """Test trapezoidal membership function."""
    
    def test_trapezoidal_evaluation(self):
        """Test trapezoidal membership function evaluation."""
        mf = TrapezoidalMembershipFunction(a=0.0, b=2.0, c=8.0, d=10.0)
        
        # Test plateau
        assert mf(5.0) == 1.0
        
        # Test boundaries
        assert mf(0.0) == 0.0
        assert mf(10.0) == 0.0
        
        # Test ramp points
        assert mf(1.0) == 0.5
        assert mf(9.0) == 0.5
    
    def test_trapezoidal_support(self):
        """Test trapezoidal membership function support."""
        mf = TrapezoidalMembershipFunction(a=0.0, b=2.0, c=8.0, d=10.0)
        support = mf.get_support()
        assert support == (0.0, 10.0)


class TestGaussianMembershipFunction:
    """Test Gaussian membership function."""
    
    def test_gaussian_evaluation(self):
        """Test Gaussian membership function evaluation."""
        mf = GaussianMembershipFunction(center=5.0, width=2.0)
        
        # Test center
        assert mf(5.0) == 1.0
        
        # Test symmetry
        assert abs(mf(3.0) - mf(7.0)) < 1e-10
        
        # Test decay
        assert mf(5.0) > mf(6.0) > mf(7.0)
    
    def test_gaussian_support(self):
        """Test Gaussian membership function support."""
        mf = GaussianMembershipFunction(center=5.0, width=2.0)
        support = mf.get_support()
        expected = (5.0 - 3 * 2.0, 5.0 + 3 * 2.0)
        assert support == expected


class TestMembershipFunctionFactory:
    """Test membership function factory."""
    
    def test_create_triangular(self):
        """Test creating triangular membership function."""
        mf_config = TriangularMF(a=0.0, b=5.0, c=10.0)
        mf = MembershipFunctionFactory.create(mf_config)
        
        assert isinstance(mf, TriangularMembershipFunction)
        assert mf(5.0) == 1.0
    
    def test_create_trapezoidal(self):
        """Test creating trapezoidal membership function."""
        mf_config = TrapezoidalMF(a=0.0, b=2.0, c=8.0, d=10.0)
        mf = MembershipFunctionFactory.create(mf_config)
        
        assert isinstance(mf, TrapezoidalMembershipFunction)
        assert mf(5.0) == 1.0
    
    def test_create_gaussian(self):
        """Test creating Gaussian membership function."""
        mf_config = GaussianMF(center=5.0, width=2.0)
        mf = MembershipFunctionFactory.create(mf_config)
        
        assert isinstance(mf, GaussianMembershipFunction)
        assert mf(5.0) == 1.0
    
    def test_create_invalid_type(self):
        """Test creating invalid membership function type."""
        with pytest.raises(ValueError):
            MembershipFunctionFactory.create("invalid")


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_analyze_universe_coverage(self):
        """Test universe coverage analysis."""
        # Create test variables
        mf1 = TriangularMF(a=0.0, b=0.0, c=5.0)
        mf2 = TriangularMF(a=5.0, b=10.0, c=10.0)
        term1 = Term(label="Low", mf=mf1)
        term2 = Term(label="High", mf=mf2)
        var = Variable(
            name="TestVar",
            kind="input",
            universe=[0.0, 10.0],
            terms=[term1, term2]
        )
        
        coverage = analyze_universe_coverage([var], (0.0, 10.0))
        
        assert "coverage_percentage" in coverage
        assert "gaps" in coverage
        assert "has_gaps" in coverage
        assert coverage["coverage_percentage"] > 0
    
    def test_analyze_overlap(self):
        """Test overlap analysis."""
        # Create test variables with overlapping terms
        mf1 = TriangularMF(a=0.0, b=5.0, c=10.0)
        mf2 = TriangularMF(a=5.0, b=10.0, c=15.0)
        term1 = Term(label="Low", mf=mf1)
        term2 = Term(label="High", mf=mf2)
        var = Variable(
            name="TestVar",
            kind="input",
            universe=[0.0, 15.0],
            terms=[term1, term2]
        )
        
        overlap = analyze_overlap([var])
        
        assert "TestVar" in overlap
        assert len(overlap["TestVar"]) == 1  # One pair of adjacent terms
        assert "normalized_overlap" in overlap["TestVar"][0]
