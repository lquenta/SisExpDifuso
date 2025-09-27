"""
Rule parser for natural language fuzzy rules.
Supports parsing rules like "IF (Déficit is Medio) AND (Presión is Alta) THEN Tarifa is Moderada"
"""

import re
from typing import Dict, List, Any, Union, Optional, Tuple
from pyparsing import (
    Word, alphas, alphanums, nums, Literal, 
    Group, Optional, OneOrMore, ZeroOrMore,
    infixNotation, opAssoc, ParseException,
    CaselessLiteral, Suppress, Combine
)
from .schema import FuzzySystemConfig, Variable, Term


class FuzzyRuleAST:
    """Abstract Syntax Tree for fuzzy rules."""
    
    def __init__(self, node_type: str, value: Any = None, children: List['FuzzyRuleAST'] = None):
        self.node_type = node_type  # 'variable', 'term', 'and', 'or', 'not', 'is'
        self.value = value
        self.children = children or []
    
    def __repr__(self):
        if self.children:
            return f"{self.node_type}({self.value}, {self.children})"
        else:
            return f"{self.node_type}({self.value})"


class FuzzyRuleParser:
    """Parser for natural language fuzzy rules."""
    
    def __init__(self, config: FuzzySystemConfig):
        """
        Initialize parser with system configuration.
        
        Args:
            config: Fuzzy system configuration containing variables and terms
        """
        self.config = config
        self.variable_names = {var.name for var in config.variables}
        self.term_names = {}
        
        # Build term lookup: {variable_name: {term_name: term_object}}
        for var in config.variables:
            self.term_names[var.name] = {term.label: term for term in var.terms}
        
        self._setup_grammar()
    
    def _setup_grammar(self):
        """Setup the parsing grammar."""
        # Basic tokens (support accented characters)
        identifier = Word(alphas + "_" + "áéíóúñüÁÉÍÓÚÑÜ", alphanums + "_" + "áéíóúñüÁÉÍÓÚÑÜ")
        number = Combine(Optional(Literal("+") | Literal("-")) + 
                        Word(nums) + 
                        Optional(Literal(".") + Word(nums)))
        
        # Keywords
        IF = CaselessLiteral("IF")
        THEN = CaselessLiteral("THEN")
        AND = CaselessLiteral("AND")
        OR = CaselessLiteral("OR")
        NOT = CaselessLiteral("NOT")
        IS = CaselessLiteral("IS")
        WITH = CaselessLiteral("WITH")
        
        # Parentheses
        lpar = Suppress("(")
        rpar = Suppress(")")
        
        # Variable-term pair: "Variable IS Term"
        variable_term = Group(
            identifier("variable") + IS + identifier("term")
        )
        
        # Negation: "NOT condition"
        negation = Group(NOT + variable_term("condition"))
        
        # Basic condition (variable-term or negation)
        basic_condition = variable_term | negation
        
        # Logical operators with proper precedence
        self.condition_expr = infixNotation(
            basic_condition,
            [
                (NOT, 1, opAssoc.RIGHT, lambda t: FuzzyRuleAST("not", children=[t[0][1]])),
                (AND, 2, opAssoc.LEFT, lambda t: FuzzyRuleAST("and", children=[t[0][0], t[0][2]])),
                (OR, 2, opAssoc.LEFT, lambda t: FuzzyRuleAST("or", children=[t[0][0], t[0][2]])),
            ]
        )
        
        # Conclusion: "Variable IS Term"
        conclusion = Group(identifier("variable") + IS + identifier("term"))
        
        # Rule weight (optional)
        weight = Group(WITH + number("weight"))
        
        # Complete rule: "IF condition THEN conclusion [WITH weight]" or "condition THEN conclusion [WITH weight]" or "condition → conclusion"
        arrow = Literal("→") | Literal("->")
        self.rule_grammar = (
            Optional(IF) + self.condition_expr("condition") + 
            (THEN | arrow) + conclusion("conclusion") + 
            Optional(weight)("weight")
        )
    
    def parse_condition(self, condition_text: str) -> FuzzyRuleAST:
        """
        Parse a condition from natural language text.
        
        Args:
            condition_text: Condition text like "Déficit is Medio AND Presión is Alta"
            
        Returns:
            Parsed condition AST
            
        Raises:
            ParseException: If condition cannot be parsed
            ValueError: If condition references non-existent variables or terms
        """
        try:
            parsed = self.condition_expr.parseString(condition_text, parseAll=True)
        except ParseException as e:
            raise ParseException(f"Error parsing condition: {e}")
        
        # Build AST
        if len(parsed) == 1:
            condition_ast = self._build_ast(parsed[0])
        else:
            condition_ast = self._build_ast(parsed)
        
        # Validate variable and term references
        self._validate_ast_references(condition_ast)
        
        return condition_ast
    
    def parse_rule(self, rule_text: str) -> Dict[str, Any]:
        """
        Parse a single rule from natural language text.
        
        Args:
            rule_text: Rule text like "IF (Déficit is Medio) AND (Presión is Alta) THEN Tarifa is Moderada"
            
        Returns:
            Dictionary with parsed rule components
            
        Raises:
            ParseException: If rule cannot be parsed
            ValueError: If rule references non-existent variables or terms
        """
        try:
            parsed = self.rule_grammar.parseString(rule_text, parseAll=True)
        except ParseException as e:
            raise ParseException(f"Error parsing rule: {e}")
        
        # Extract components
        condition_ast = self._build_ast(parsed.condition)
        conclusion = parsed.conclusion
        weight = float(parsed.weight[0]) if parsed.weight else 1.0
        
        # Validate variable and term references
        self._validate_rule_references(condition_ast, conclusion)
        
        return {
            "condition_ast": condition_ast,
            "conclusion": {
                "variable": conclusion.variable,
                "term": conclusion.term
            },
            "weight": weight,
            "original_text": rule_text
        }
    
    def _build_ast(self, parsed_condition) -> FuzzyRuleAST:
        """Build AST from parsed condition."""
        # If it's already a FuzzyRuleAST, process its children
        if isinstance(parsed_condition, FuzzyRuleAST):
            # Process children if they exist
            if parsed_condition.children:
                processed_children = []
                for child in parsed_condition.children:
                    if hasattr(child, 'variable') and hasattr(child, 'term'):
                        # Convert ParseResults to FuzzyRuleAST
                        processed_children.append(FuzzyRuleAST("is", value={
                            "variable": child.variable,
                            "term": child.term
                        }))
                    else:
                        processed_children.append(self._build_ast(child))
                return FuzzyRuleAST(parsed_condition.node_type, value=parsed_condition.value, children=processed_children)
            else:
                return parsed_condition
        
        if isinstance(parsed_condition, list) and len(parsed_condition) == 1:
            parsed_condition = parsed_condition[0]
        
        if hasattr(parsed_condition, 'variable') and hasattr(parsed_condition, 'term'):
            # Variable-term pair
            return FuzzyRuleAST("is", value={
                "variable": parsed_condition.variable,
                "term": parsed_condition.term
            })
        elif hasattr(parsed_condition, 'condition'):
            # Negation
            return FuzzyRuleAST("not", children=[self._build_ast(parsed_condition.condition)])
        elif len(parsed_condition) == 3:
            # Binary operation (AND/OR)
            left = self._build_ast(parsed_condition[0])
            operator = parsed_condition[1].lower()
            right = self._build_ast(parsed_condition[2])
            return FuzzyRuleAST(operator, children=[left, right])
        else:
            raise ValueError(f"Unexpected parsed condition structure: {parsed_condition}")
    
    def _validate_rule_references(self, condition_ast: FuzzyRuleAST, conclusion: Any):
        """Validate that all variable and term references exist."""
        # Validate condition
        self._validate_ast_references(condition_ast)
        
        # Validate conclusion
        if conclusion.variable not in self.variable_names:
            raise ValueError(f"Unknown variable in conclusion: {conclusion.variable}")
        
        var_name = conclusion.variable
        if var_name not in self.term_names:
            raise ValueError(f"Variable {var_name} has no terms defined")
        
        if conclusion.term not in self.term_names[var_name]:
            raise ValueError(f"Unknown term '{conclusion.term}' for variable '{var_name}'")
    
    def _validate_ast_references(self, ast: FuzzyRuleAST):
        """Recursively validate AST references."""
        if ast.node_type == "is":
            var_name = ast.value["variable"]
            term_name = ast.value["term"]
            
            if var_name not in self.variable_names:
                raise ValueError(f"Unknown variable: {var_name}")
            
            if var_name not in self.term_names:
                raise ValueError(f"Variable {var_name} has no terms defined")
            
            if term_name not in self.term_names[var_name]:
                raise ValueError(f"Unknown term '{term_name}' for variable '{var_name}'")
        
        elif ast.node_type in ["and", "or", "not"]:
            for child in ast.children:
                self._validate_ast_references(child)
        
        else:
            raise ValueError(f"Unknown AST node type: {ast.node_type}")
    
    def evaluate_condition(self, condition_ast: FuzzyRuleAST, input_values: Dict[str, float], 
                          membership_values: Dict[Tuple[str, str], float]) -> float:
        """
        Evaluate condition AST with given input values and membership degrees.
        
        Args:
            condition_ast: Parsed condition AST
            input_values: Input variable values
            membership_values: Pre-computed membership degrees {(variable, term): degree}
            
        Returns:
            Truth value of the condition
        """
        if condition_ast.node_type == "is":
            var_name = condition_ast.value["variable"]
            term_name = condition_ast.value["term"]
            return membership_values.get((var_name, term_name), 0.0)
        
        elif condition_ast.node_type == "and":
            left_val = self.evaluate_condition(condition_ast.children[0], input_values, membership_values)
            right_val = self.evaluate_condition(condition_ast.children[1], input_values, membership_values)
            return min(left_val, right_val)  # Mamdani AND (min)
        
        elif condition_ast.node_type == "or":
            left_val = self.evaluate_condition(condition_ast.children[0], input_values, membership_values)
            right_val = self.evaluate_condition(condition_ast.children[1], input_values, membership_values)
            return max(left_val, right_val)  # Mamdani OR (max)
        
        elif condition_ast.node_type == "not":
            child_val = self.evaluate_condition(condition_ast.children[0], input_values, membership_values)
            return 1.0 - child_val  # Fuzzy NOT
        
        else:
            raise ValueError(f"Unknown AST node type: {condition_ast.node_type}")
    
    def parse_all_conditions(self, conditions: List[str]) -> List[Dict[str, Any]]:
        """
        Parse multiple conditions.
        
        Args:
            conditions: List of condition texts
            
        Returns:
            List of parsed condition dictionaries
        """
        parsed_conditions = []
        for i, condition_text in enumerate(conditions):
            try:
                condition_ast = self.parse_condition(condition_text)
                parsed_conditions.append({
                    "condition_ast": condition_ast,
                    "original_text": condition_text
                })
            except (ParseException, ValueError) as e:
                raise ValueError(f"Error parsing condition {i+1}: {e}")
        
        return parsed_conditions
    
    def parse_all_rules(self, rules: List[str]) -> List[Dict[str, Any]]:
        """
        Parse multiple rules.
        
        Args:
            rules: List of rule texts
            
        Returns:
            List of parsed rule dictionaries
        """
        parsed_rules = []
        for i, rule_text in enumerate(rules):
            try:
                parsed = self.parse_rule(rule_text)
                parsed["rule_id"] = f"R{i+1}"
                parsed_rules.append(parsed)
            except (ParseException, ValueError) as e:
                raise ValueError(f"Error parsing rule {i+1}: {e}")
        
        return parsed_rules


def create_rule_parser(config: FuzzySystemConfig) -> FuzzyRuleParser:
    """Convenience function to create rule parser."""
    return FuzzyRuleParser(config)


# Utility functions for rule analysis

def analyze_rule_complexity(parsed_rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze complexity of parsed rules.
    
    Args:
        parsed_rules: List of parsed rule dictionaries
        
    Returns:
        Dictionary with complexity analysis
    """
    complexities = []
    operator_counts = {"and": 0, "or": 0, "not": 0}
    
    for rule in parsed_rules:
        complexity = _count_ast_complexity(rule["condition_ast"])
        complexities.append(complexity)
        
        # Count operators
        _count_operators(rule["condition_ast"], operator_counts)
    
    return {
        "rule_count": len(parsed_rules),
        "avg_complexity": sum(complexities) / len(complexities) if complexities else 0,
        "max_complexity": max(complexities) if complexities else 0,
        "min_complexity": min(complexities) if complexities else 0,
        "operator_counts": operator_counts,
        "complexity_distribution": {
            "simple": sum(1 for c in complexities if c <= 2),
            "medium": sum(1 for c in complexities if 2 < c <= 5),
            "complex": sum(1 for c in complexities if c > 5)
        }
    }


def _count_ast_complexity(ast: FuzzyRuleAST) -> int:
    """Count complexity of AST (number of nodes)."""
    if ast.node_type in ["and", "or", "not"]:
        return 1 + sum(_count_ast_complexity(child) for child in ast.children)
    else:
        return 1


def _count_operators(ast: FuzzyRuleAST, counts: Dict[str, int]):
    """Count operators in AST."""
    if ast.node_type in ["and", "or", "not"]:
        counts[ast.node_type] += 1
        for child in ast.children:
            _count_operators(child, counts)


def validate_rule_consistency(parsed_rules: List[Dict[str, Any]], config: FuzzySystemConfig) -> List[str]:
    """
    Validate consistency of parsed rules.
    
    Args:
        parsed_rules: List of parsed rule dictionaries
        config: Fuzzy system configuration
        
    Returns:
        List of validation warnings/errors
    """
    warnings = []
    
    # Check for conflicting rules (same condition, different conclusions)
    condition_conclusions = {}
    
    for rule in parsed_rules:
        condition_key = _ast_to_string(rule["condition_ast"])
        conclusion = rule["conclusion"]
        
        if condition_key in condition_conclusions:
            existing = condition_conclusions[condition_key]
            if existing != conclusion:
                warnings.append(
                    f"Conflicting rules: same condition '{condition_key}' "
                    f"leads to different conclusions: {existing} vs {conclusion}"
                )
        else:
            condition_conclusions[condition_key] = conclusion
    
    # Check for unreachable conclusions
    output_vars = {var.name for var in config.get_output_variables()}
    reached_conclusions = set()
    
    for rule in parsed_rules:
        conclusion = rule["conclusion"]
        reached_conclusions.add((conclusion["variable"], conclusion["term"]))
    
    for var in config.get_output_variables():
        for term in var.terms:
            if (var.name, term.label) not in reached_conclusions:
                warnings.append(
                    f"Term '{term.label}' of output variable '{var.name}' "
                    f"is never reached by any rule"
                )
    
    return warnings


def _ast_to_string(ast: FuzzyRuleAST) -> str:
    """Convert AST to string representation."""
    if ast.node_type == "is":
        return f"({ast.value['variable']} is {ast.value['term']})"
    elif ast.node_type == "and":
        left = _ast_to_string(ast.children[0])
        right = _ast_to_string(ast.children[1])
        return f"({left} AND {right})"
    elif ast.node_type == "or":
        left = _ast_to_string(ast.children[0])
        right = _ast_to_string(ast.children[1])
        return f"({left} OR {right})"
    elif ast.node_type == "not":
        child = _ast_to_string(ast.children[0])
        return f"NOT {child}"
    else:
        return f"UNKNOWN({ast.node_type})"
