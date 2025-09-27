"""
Mamdani-min fuzzy inference engine.
Implements the complete fuzzy inference process with fuzzification, rule evaluation,
implication, aggregation, and defuzzification.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from .schema import FuzzySystemConfig, Variable, Term
from .mfuncs import create_membership_function, MembershipFunctionBase
from .parser import FuzzyRuleParser, FuzzyRuleAST
from .defuzz import Defuzzifier, DefuzzificationMethod


class FuzzyInferenceResult:
    """Result of fuzzy inference process."""
    
    def __init__(self, outputs: Dict[str, float], 
                 membership_values: Dict[Tuple[str, str], float],
                 fired_rules: List[Dict[str, Any]],
                 aggregated_functions: Dict[str, Tuple[np.ndarray, np.ndarray]],
                 explanation: Dict[str, Any]):
        """
        Initialize inference result.
        
        Args:
            outputs: Crisp output values {variable_name: value}
            membership_values: Membership degrees {(variable, term): degree}
            fired_rules: List of fired rules with activation degrees
            aggregated_functions: Aggregated membership functions {variable: (x, y)}
            explanation: Detailed explanation of inference process
        """
        self.outputs = outputs
        self.membership_values = membership_values
        self.fired_rules = fired_rules
        self.aggregated_functions = aggregated_functions
        self.explanation = explanation
    
    def get_output(self, variable_name: str) -> Optional[float]:
        """Get output value for a specific variable."""
        return self.outputs.get(variable_name)
    
    def get_membership(self, variable_name: str, term_name: str) -> float:
        """Get membership degree for a variable-term pair."""
        return self.membership_values.get((variable_name, term_name), 0.0)
    
    def get_fired_rules(self) -> List[Dict[str, Any]]:
        """Get list of fired rules with activation degrees."""
        return self.fired_rules
    
    def get_aggregated_function(self, variable_name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get aggregated membership function for a variable."""
        return self.aggregated_functions.get(variable_name)


class FuzzyInferenceEngine:
    """Mamdani-min fuzzy inference engine."""
    
    def __init__(self, config: FuzzySystemConfig, resolution: int = 1001):
        """
        Initialize fuzzy inference engine.
        
        Args:
            config: Fuzzy system configuration
            resolution: Resolution for output domain discretization
        """
        self.config = config
        self.resolution = resolution
        
        # Pre-compile membership functions
        self.membership_functions: Dict[Tuple[str, str], MembershipFunctionBase] = {}
        self._compile_membership_functions()
        
        # Parse rules
        self.rule_parser = FuzzyRuleParser(config)
        self.parsed_rules = self._parse_rules()
        
        # Create defuzzifiers for each output variable
        self.defuzzifiers: Dict[str, Defuzzifier] = {}
        self._setup_defuzzifiers()
    
    def _compile_membership_functions(self):
        """Pre-compile all membership functions for efficiency."""
        for var in self.config.variables:
            for term in var.terms:
                key = (var.name, term.label)
                self.membership_functions[key] = create_membership_function(term.mf)
    
    def _parse_rules(self) -> List[Dict[str, Any]]:
        """Parse all rules from configuration."""
        condition_texts = [rule.if_condition for rule in self.config.rules]
        parsed_conditions = self.rule_parser.parse_all_conditions(condition_texts)
        
        # Add rule metadata and conclusions
        parsed_rules = []
        for i, parsed_condition in enumerate(parsed_conditions):
            original_rule = self.config.rules[i]
            parsed_rule = {
                "id": original_rule.id,
                "condition_ast": parsed_condition["condition_ast"],
                "conclusion": {
                    "variable": original_rule.then_conclusions[0].variable,
                    "term": original_rule.then_conclusions[0].term
                },
                "weight": original_rule.weight,
                "note": original_rule.note,
                "original_text": parsed_condition["original_text"]
            }
            parsed_rules.append(parsed_rule)
        
        return parsed_rules
    
    def _setup_defuzzifiers(self):
        """Setup defuzzifiers for each output variable."""
        for var in self.config.get_output_variables():
            method = DefuzzificationMethod(var.defuzz or self.config.logic.defuzz_default)
            self.defuzzifiers[var.name] = Defuzzifier(method)
    
    def fuzzify(self, input_values: Dict[str, float]) -> Dict[Tuple[str, str], float]:
        """
        Fuzzify input values to membership degrees.
        
        Args:
            input_values: Input variable values {variable_name: value}
            
        Returns:
            Membership degrees {(variable, term): degree}
        """
        membership_values = {}
        
        for var_name, value in input_values.items():
            var = self.config.get_variable(var_name)
            if var is None:
                continue
            
            for term in var.terms:
                key = (var_name, term.label)
                mf = self.membership_functions[key]
                membership_values[key] = float(mf(value))
        
        return membership_values
    
    def evaluate_rules(self, membership_values: Dict[Tuple[str, str], float]) -> List[Dict[str, Any]]:
        """
        Evaluate all rules and return fired rules with activation degrees.
        
        Args:
            membership_values: Fuzzified membership degrees
            
        Returns:
            List of fired rules with activation degrees
        """
        fired_rules = []
        
        for rule in self.parsed_rules:
            # Evaluate condition
            activation_degree = self.rule_parser.evaluate_condition(
                rule["condition_ast"], {}, membership_values
            )
            
            # Apply rule weight
            activation_degree *= rule["weight"]
            
            # Only include rules with non-zero activation
            if activation_degree > 0:
                fired_rules.append({
                    "id": rule["id"],
                    "condition": rule["original_text"],
                    "conclusion": rule["conclusion"],
                    "activation_degree": activation_degree,
                    "weight": rule["weight"],
                    "note": rule.get("note")
                })
        
        return fired_rules
    
    def apply_implication(self, activation_degree: float, 
                         membership_function: MembershipFunctionBase,
                         x: np.ndarray) -> np.ndarray:
        """
        Apply Mamdani implication: μ_imp(y) = min(α, μ_consequent(y))
        
        Args:
            activation_degree: Rule activation degree (α)
            membership_function: Consequent membership function
            x: Output domain points
            
        Returns:
            Implied membership function
        """
        consequent_membership = membership_function(x)
        return np.minimum(activation_degree, consequent_membership)
    
    def aggregate_outputs(self, fired_rules: List[Dict[str, Any]]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Aggregate implied membership functions for each output variable.
        
        Args:
            fired_rules: List of fired rules with activation degrees
            
        Returns:
            Aggregated membership functions {variable: (x, y)}
        """
        output_vars = self.config.get_output_variables()
        aggregated_functions = {}
        
        for var in output_vars:
            # Create output domain
            x = np.linspace(var.universe[0], var.universe[1], self.resolution)
            aggregated_membership = np.zeros_like(x)
            
            # Aggregate all rules affecting this variable
            for rule in fired_rules:
                conclusion = rule["conclusion"]
                if conclusion["variable"] == var.name:
                    # Get consequent membership function
                    term_key = (var.name, conclusion["term"])
                    mf = self.membership_functions[term_key]
                    
                    # Apply implication
                    implied_membership = self.apply_implication(
                        rule["activation_degree"], mf, x
                    )
                    
                    # Aggregate using max (Mamdani aggregation)
                    aggregated_membership = np.maximum(aggregated_membership, implied_membership)
            
            aggregated_functions[var.name] = (x, aggregated_membership)
        
        return aggregated_functions
    
    def defuzzify_outputs(self, aggregated_functions: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
        """
        Defuzzify aggregated membership functions to crisp values.
        
        Args:
            aggregated_functions: Aggregated membership functions
            
        Returns:
            Crisp output values {variable_name: value}
        """
        outputs = {}
        
        for var_name, (x, membership) in aggregated_functions.items():
            var = self.config.get_variable(var_name)
            if var is None:
                continue
            
            defuzzifier = self.defuzzifiers[var_name]
            crisp_value = defuzzifier.defuzzify(x, membership, tuple(var.universe))
            outputs[var_name] = crisp_value
        
        return outputs
    
    def create_explanation(self, input_values: Dict[str, float],
                          membership_values: Dict[Tuple[str, str], float],
                          fired_rules: List[Dict[str, Any]],
                          outputs: Dict[str, float]) -> Dict[str, Any]:
        """
        Create detailed explanation of inference process.
        
        Args:
            input_values: Original input values
            membership_values: Fuzzified membership degrees
            fired_rules: Fired rules with activation degrees
            outputs: Final crisp outputs
            
        Returns:
            Detailed explanation dictionary
        """
        explanation = {
            "input_values": input_values,
            "fuzzification": {},
            "rule_evaluation": {},
            "outputs": outputs,
            "summary": {}
        }
        
        # Fuzzification details
        for (var_name, term_name), degree in membership_values.items():
            if var_name not in explanation["fuzzification"]:
                explanation["fuzzification"][var_name] = {}
            explanation["fuzzification"][var_name][term_name] = degree
        
        # Rule evaluation details
        for rule in fired_rules:
            explanation["rule_evaluation"][rule["id"]] = {
                "condition": rule["condition"],
                "activation_degree": rule["activation_degree"],
                "conclusion": rule["conclusion"],
                "weight": rule["weight"],
                "note": rule.get("note")
            }
        
        # Summary statistics
        explanation["summary"] = {
            "total_rules": len(self.parsed_rules),
            "fired_rules": len(fired_rules),
            "max_activation": max([r["activation_degree"] for r in fired_rules]) if fired_rules else 0,
            "avg_activation": np.mean([r["activation_degree"] for r in fired_rules]) if fired_rules else 0
        }
        
        return explanation
    
    def infer(self, input_values: Dict[str, float]) -> FuzzyInferenceResult:
        """
        Perform complete fuzzy inference process.
        
        Args:
            input_values: Input variable values {variable_name: value}
            
        Returns:
            Complete inference result with outputs and explanation
            
        Raises:
            ValueError: If input values are invalid
        """
        # Validate input values
        self._validate_inputs(input_values)
        
        # Step 1: Fuzzification
        membership_values = self.fuzzify(input_values)
        
        # Step 2: Rule evaluation
        fired_rules = self.evaluate_rules(membership_values)
        
        # Step 3: Implication and aggregation
        aggregated_functions = self.aggregate_outputs(fired_rules)
        
        # Step 4: Defuzzification
        outputs = self.defuzzify_outputs(aggregated_functions)
        
        # Step 5: Create explanation
        explanation = self.create_explanation(input_values, membership_values, fired_rules, outputs)
        
        return FuzzyInferenceResult(
            outputs=outputs,
            membership_values=membership_values,
            fired_rules=fired_rules,
            aggregated_functions=aggregated_functions,
            explanation=explanation
        )
    
    def _validate_inputs(self, input_values: Dict[str, float]):
        """Validate input values against system configuration."""
        input_vars = self.config.get_input_variables()
        input_var_names = {var.name for var in input_vars}
        
        # Check for unknown variables
        for var_name in input_values:
            if var_name not in input_var_names:
                raise ValueError(f"Unknown input variable: {var_name}")
        
        # Check for missing required variables
        for var in input_vars:
            if var.name not in input_values:
                raise ValueError(f"Missing required input variable: {var.name}")
        
        # Check value ranges
        for var_name, value in input_values.items():
            var = self.config.get_variable(var_name)
            if var is None:
                continue
            
            min_val, max_val = var.universe
            if not (min_val <= value <= max_val):
                raise ValueError(
                    f"Value {value} for variable {var_name} is outside universe [{min_val}, {max_val}]"
                )
    
    def get_linguistic_output(self, result: FuzzyInferenceResult) -> Dict[str, Dict[str, Any]]:
        """
        Get linguistic interpretation of outputs.
        
        Args:
            result: Inference result
            
        Returns:
            Linguistic outputs {variable: {term: confidence, crisp: value}}
        """
        linguistic_outputs = {}
        
        for var_name, crisp_value in result.outputs.items():
            var = self.config.get_variable(var_name)
            if var is None:
                continue
            
            # Find best matching term
            best_term = None
            best_confidence = 0.0
            
            for term in var.terms:
                key = (var_name, term.label)
                mf = self.membership_functions[key]
                confidence = float(mf(crisp_value))
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_term = term.label
            
            linguistic_outputs[var_name] = {
                "best_term": best_term,
                "confidence": best_confidence,
                "crisp_value": crisp_value,
                "all_terms": {
                    term.label: float(self.membership_functions[(var_name, term.label)](crisp_value))
                    for term in var.terms
                }
            }
        
        return linguistic_outputs


def create_inference_engine(config: FuzzySystemConfig, resolution: int = 1001) -> FuzzyInferenceEngine:
    """Convenience function to create inference engine."""
    return FuzzyInferenceEngine(config, resolution)
