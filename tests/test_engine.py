"""
Tests for fuzzy inference engine.
"""

import pytest
import numpy as np

from fuzzy_panel.core.engine import FuzzyInferenceEngine, FuzzyInferenceResult, create_inference_engine
from fuzzy_panel.core.schema import create_example_config


class TestFuzzyInferenceEngine:
    """Test fuzzy inference engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = create_example_config()
        self.engine = create_inference_engine(self.config)
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        assert self.engine is not None
        assert self.engine.config == self.config
        assert self.engine.resolution == 1001
    
    def test_fuzzify(self):
        """Test fuzzification."""
        input_values = {"Déficit": 5.0, "Presión": 50.0}
        membership_values = self.engine.fuzzify(input_values)
        
        # Check that membership values are returned
        assert len(membership_values) > 0
        
        # Check that all values are in [0, 1]
        for value in membership_values.values():
            assert 0.0 <= value <= 1.0
    
    def test_evaluate_rules(self):
        """Test rule evaluation."""
        input_values = {"Déficit": 5.0, "Presión": 50.0}
        membership_values = self.engine.fuzzify(input_values)
        fired_rules = self.engine.evaluate_rules(membership_values)
        
        # Check that fired rules are returned
        assert isinstance(fired_rules, list)
        
        # Check that activation degrees are in [0, 1]
        for rule in fired_rules:
            assert 0.0 <= rule["activation_degree"] <= 1.0
    
    def test_apply_implication(self):
        """Test implication application."""
        mf = self.engine.membership_functions[("Tarifa", "Moderada")]
        x = np.linspace(0, 100, 101)
        activation_degree = 0.5
        
        implied = self.engine.apply_implication(activation_degree, mf, x)
        
        # Check that implied values are in [0, activation_degree]
        assert np.all(implied >= 0.0)
        assert np.all(implied <= activation_degree)
    
    def test_aggregate_outputs(self):
        """Test output aggregation."""
        # Create mock fired rules
        fired_rules = [{
            "id": "R1",
            "condition": "test",
            "conclusion": {"variable": "Tarifa", "term": "Moderada"},
            "activation_degree": 0.5,
            "weight": 1.0,
            "note": "test"
        }]
        
        aggregated = self.engine.aggregate_outputs(fired_rules)
        
        # Check that aggregated functions are returned
        assert "Tarifa" in aggregated
        x, membership = aggregated["Tarifa"]
        assert len(x) == self.engine.resolution
        assert len(membership) == self.engine.resolution
        assert np.all(membership >= 0.0)
    
    def test_defuzzify_outputs(self):
        """Test output defuzzification."""
        # Create mock aggregated functions
        x = np.linspace(0, 100, 101)
        membership = np.exp(-0.5 * ((x - 50) / 10) ** 2)  # Gaussian-like
        aggregated = {"Tarifa": (x, membership)}
        
        outputs = self.engine.defuzzify_outputs(aggregated)
        
        # Check that outputs are returned
        assert "Tarifa" in outputs
        assert isinstance(outputs["Tarifa"], float)
        assert 0.0 <= outputs["Tarifa"] <= 100.0
    
    def test_infer(self):
        """Test complete inference process."""
        input_values = {"Déficit": 5.0, "Presión": 50.0}
        result = self.engine.infer(input_values)
        
        # Check result type
        assert isinstance(result, FuzzyInferenceResult)
        
        # Check outputs
        assert "Tarifa" in result.outputs
        assert isinstance(result.outputs["Tarifa"], float)
        
        # Check membership values
        assert len(result.membership_values) > 0
        
        # Check fired rules
        assert isinstance(result.fired_rules, list)
        
        # Check aggregated functions
        assert "Tarifa" in result.aggregated_functions
        
        # Check explanation
        assert "input_values" in result.explanation
        assert "fuzzification" in result.explanation
        assert "rule_evaluation" in result.explanation
        assert "outputs" in result.explanation
        assert "summary" in result.explanation
    
    def test_infer_invalid_inputs(self):
        """Test inference with invalid inputs."""
        # Test unknown variable
        with pytest.raises(ValueError):
            self.engine.infer({"UnknownVar": 5.0})
        
        # Test missing variable
        with pytest.raises(ValueError):
            self.engine.infer({"Déficit": 5.0})  # Missing "Presión"
        
        # Test value outside universe
        with pytest.raises(ValueError):
            self.engine.infer({"Déficit": 15.0, "Presión": 50.0})  # Déficit > 10
    
    def test_get_linguistic_output(self):
        """Test linguistic output interpretation."""
        input_values = {"Déficit": 5.0, "Presión": 50.0}
        result = self.engine.infer(input_values)
        linguistic = self.engine.get_linguistic_output(result)
        
        # Check that linguistic outputs are returned
        assert "Tarifa" in linguistic
        assert "best_term" in linguistic["Tarifa"]
        assert "confidence" in linguistic["Tarifa"]
        assert "all_terms" in linguistic["Tarifa"]
        
        # Check confidence is in [0, 1]
        assert 0.0 <= linguistic["Tarifa"]["confidence"] <= 1.0


class TestFuzzyInferenceResult:
    """Test fuzzy inference result."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = create_example_config()
        self.engine = create_inference_engine(self.config)
        self.result = self.engine.infer({"Déficit": 5.0, "Presión": 50.0})
    
    def test_get_output(self):
        """Test getting output value."""
        output = self.result.get_output("Tarifa")
        assert output is not None
        assert isinstance(output, float)
        
        output = self.result.get_output("NonExistent")
        assert output is None
    
    def test_get_membership(self):
        """Test getting membership degree."""
        membership = self.result.get_membership("Déficit", "Medio")
        assert isinstance(membership, float)
        assert 0.0 <= membership <= 1.0
        
        membership = self.result.get_membership("NonExistent", "Term")
        assert membership == 0.0
    
    def test_get_fired_rules(self):
        """Test getting fired rules."""
        fired_rules = self.result.get_fired_rules()
        assert isinstance(fired_rules, list)
    
    def test_get_aggregated_function(self):
        """Test getting aggregated function."""
        aggregated = self.result.get_aggregated_function("Tarifa")
        assert aggregated is not None
        x, membership = aggregated
        assert len(x) > 0
        assert len(membership) > 0
        
        aggregated = self.result.get_aggregated_function("NonExistent")
        assert aggregated is None


class TestIntegration:
    """Integration tests."""
    
    def test_complete_workflow(self):
        """Test complete fuzzy inference workflow."""
        config = create_example_config()
        engine = create_inference_engine(config)
        
        # Test multiple input scenarios
        test_cases = [
            {"Déficit": 2.0, "Presión": 20.0},  # Low values
            {"Déficit": 5.0, "Presión": 50.0},  # Medium values
            {"Déficit": 8.0, "Presión": 80.0},  # High values
        ]
        
        for inputs in test_cases:
            result = engine.infer(inputs)
            
            # Check that inference completes successfully
            assert result is not None
            assert "Tarifa" in result.outputs
            assert isinstance(result.outputs["Tarifa"], float)
            assert 0.0 <= result.outputs["Tarifa"] <= 100.0
    
    def test_deterministic_results(self):
        """Test that results are deterministic."""
        config = create_example_config()
        engine = create_inference_engine(config)
        inputs = {"Déficit": 5.0, "Presión": 50.0}
        
        # Run inference multiple times
        results = []
        for _ in range(5):
            result = engine.infer(inputs)
            results.append(result.outputs["Tarifa"])
        
        # Check that all results are identical
        assert all(r == results[0] for r in results)
    
    def test_edge_cases(self):
        """Test edge cases."""
        config = create_example_config()
        engine = create_inference_engine(config)
        
        # Test boundary values
        boundary_cases = [
            {"Déficit": 0.0, "Presión": 0.0},    # Minimum values
            {"Déficit": 10.0, "Presión": 100.0}, # Maximum values
        ]
        
        for inputs in boundary_cases:
            result = engine.infer(inputs)
            assert result is not None
            assert "Tarifa" in result.outputs
