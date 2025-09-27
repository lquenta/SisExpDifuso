"""
Tests for schema validation and models.
"""

import pytest
import numpy as np
from pydantic import ValidationError

from fuzzy_panel.core.schema import (
    TriangularMF, TrapezoidalMF, GaussianMF, Term, Variable, 
    LogicConfig, Rule, RuleConclusion, FuzzySystemConfig, create_example_config
)


class TestTriangularMF:
    """Test triangular membership function."""
    
    def test_valid_triangular(self):
        """Test valid triangular membership function."""
        mf = TriangularMF(a=0.0, b=5.0, c=10.0)
        assert mf.a == 0.0
        assert mf.b == 5.0
        assert mf.c == 10.0
    
    def test_invalid_triangular_order(self):
        """Test invalid triangular vertex order."""
        with pytest.raises(ValidationError):
            TriangularMF(a=5.0, b=0.0, c=10.0)
    
    def test_triangular_validation(self):
        """Test triangular validation."""
        with pytest.raises(ValidationError):
            TriangularMF(a="invalid", b=5.0, c=10.0)


class TestTrapezoidalMF:
    """Test trapezoidal membership function."""
    
    def test_valid_trapezoidal(self):
        """Test valid trapezoidal membership function."""
        mf = TrapezoidalMF(a=0.0, b=2.0, c=8.0, d=10.0)
        assert mf.a == 0.0
        assert mf.b == 2.0
        assert mf.c == 8.0
        assert mf.d == 10.0
    
    def test_invalid_trapezoidal_order(self):
        """Test invalid trapezoidal vertex order."""
        with pytest.raises(ValidationError):
            TrapezoidalMF(a=0.0, b=8.0, c=2.0, d=10.0)


class TestGaussianMF:
    """Test Gaussian membership function."""
    
    def test_valid_gaussian(self):
        """Test valid Gaussian membership function."""
        mf = GaussianMF(center=5.0, width=2.0)
        assert mf.center == 5.0
        assert mf.width == 2.0
    
    def test_invalid_gaussian_width(self):
        """Test invalid Gaussian width."""
        with pytest.raises(ValidationError):
            GaussianMF(center=5.0, width=0.0)


class TestTerm:
    """Test fuzzy term."""
    
    def test_valid_term(self):
        """Test valid term."""
        mf = TriangularMF(a=0.0, b=5.0, c=10.0)
        term = Term(label="Medium", mf=mf)
        assert term.label == "Medium"
        assert term.mf == mf
    
    def test_empty_label(self):
        """Test empty label validation."""
        mf = TriangularMF(a=0.0, b=5.0, c=10.0)
        with pytest.raises(ValidationError):
            Term(label="", mf=mf)


class TestVariable:
    """Test fuzzy variable."""
    
    def test_valid_variable(self):
        """Test valid variable."""
        mf = TriangularMF(a=0.0, b=5.0, c=10.0)
        term = Term(label="Medium", mf=mf)
        var = Variable(
            name="TestVar",
            kind="input",
            universe=[0.0, 10.0],
            terms=[term]
        )
        assert var.name == "TestVar"
        assert var.kind == "input"
        assert var.universe == [0.0, 10.0]
        assert len(var.terms) == 1
    
    def test_invalid_universe(self):
        """Test invalid universe."""
        mf = TriangularMF(a=0.0, b=5.0, c=10.0)
        term = Term(label="Medium", mf=mf)
        with pytest.raises(ValidationError):
            Variable(
                name="TestVar",
                kind="input",
                universe=[10.0, 0.0],  # Invalid: min > max
                terms=[term]
            )
    
    def test_duplicate_term_labels(self):
        """Test duplicate term labels."""
        mf1 = TriangularMF(a=0.0, b=5.0, c=10.0)
        mf2 = TriangularMF(a=0.0, b=5.0, c=10.0)
        term1 = Term(label="Medium", mf=mf1)
        term2 = Term(label="Medium", mf=mf2)
        
        with pytest.raises(ValidationError):
            Variable(
                name="TestVar",
                kind="input",
                universe=[0.0, 10.0],
                terms=[term1, term2]
            )


class TestRule:
    """Test fuzzy rule."""
    
    def test_valid_rule(self):
        """Test valid rule."""
        conclusion = RuleConclusion(variable="Output", term="High")
        rule = Rule(
            id="R1",
            if_condition="Input is High",
            then_conclusions=[conclusion],
            weight=1.0
        )
        assert rule.id == "R1"
        assert rule.if_condition == "Input is High"
        assert len(rule.then_conclusions) == 1
        assert rule.weight == 1.0
    
    def test_invalid_weight(self):
        """Test invalid rule weight."""
        conclusion = RuleConclusion(variable="Output", term="High")
        with pytest.raises(ValidationError):
            Rule(
                id="R1",
                if_condition="Input is High",
                then_conclusions=[conclusion],
                weight=1.5  # Invalid: weight > 1.0
            )


class TestFuzzySystemConfig:
    """Test fuzzy system configuration."""
    
    def test_valid_config(self):
        """Test valid system configuration."""
        config = create_example_config()
        assert config.schema_version == "1.0"
        assert config.project == "Politica_Arancelaria"
        assert len(config.variables) == 3
        assert len(config.rules) == 1
    
    def test_no_input_variables(self):
        """Test configuration with no input variables."""
        mf = TriangularMF(a=0.0, b=5.0, c=10.0)
        term = Term(label="Medium", mf=mf)
        var = Variable(
            name="Output",
            kind="output",
            universe=[0.0, 10.0],
            terms=[term]
        )
        
        with pytest.raises(ValidationError):
            FuzzySystemConfig(
                project="Test",
                variables=[var],
                rules=[]
            )
    
    def test_no_output_variables(self):
        """Test configuration with no output variables."""
        mf = TriangularMF(a=0.0, b=5.0, c=10.0)
        term = Term(label="Medium", mf=mf)
        var = Variable(
            name="Input",
            kind="input",
            universe=[0.0, 10.0],
            terms=[term]
        )
        
        with pytest.raises(ValidationError):
            FuzzySystemConfig(
                project="Test",
                variables=[var],
                rules=[]
            )
    
    def test_duplicate_variable_names(self):
        """Test configuration with duplicate variable names."""
        mf = TriangularMF(a=0.0, b=5.0, c=10.0)
        term = Term(label="Medium", mf=mf)
        var1 = Variable(
            name="TestVar",
            kind="input",
            universe=[0.0, 10.0],
            terms=[term]
        )
        var2 = Variable(
            name="TestVar",  # Duplicate name
            kind="output",
            universe=[0.0, 10.0],
            terms=[term]
        )
        
        with pytest.raises(ValidationError):
            FuzzySystemConfig(
                project="Test",
                variables=[var1, var2],
                rules=[]
            )
    
    def test_get_variable(self):
        """Test getting variable by name."""
        config = create_example_config()
        var = config.get_variable("Déficit")
        assert var is not None
        assert var.name == "Déficit"
        
        var = config.get_variable("NonExistent")
        assert var is None
    
    def test_get_input_variables(self):
        """Test getting input variables."""
        config = create_example_config()
        input_vars = config.get_input_variables()
        assert len(input_vars) == 2
        assert all(var.kind == "input" for var in input_vars)
    
    def test_get_output_variables(self):
        """Test getting output variables."""
        config = create_example_config()
        output_vars = config.get_output_variables()
        assert len(output_vars) == 1
        assert all(var.kind == "output" for var in output_vars)
