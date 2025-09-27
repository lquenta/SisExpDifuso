"""
Explanation and traceability system for fuzzy inference.
Provides detailed step-by-step explanations of the inference process.
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from .engine import FuzzyInferenceResult
from .schema import FuzzySystemConfig


class FuzzyExplanation:
    """Detailed explanation of fuzzy inference process."""
    
    def __init__(self, result: FuzzyInferenceResult, config: FuzzySystemConfig):
        """
        Initialize explanation from inference result.
        
        Args:
            result: Fuzzy inference result
            config: Fuzzy system configuration
        """
        self.result = result
        self.config = config
        self.explanation = result.explanation
    
    def get_step_by_step_explanation(self) -> List[Dict[str, Any]]:
        """
        Get step-by-step explanation of the inference process.
        
        Returns:
            List of explanation steps
        """
        steps = []
        
        # Step 1: Input values
        steps.append({
            "step": 1,
            "title": "Input Values",
            "description": "Input variables and their crisp values",
            "details": self._format_input_values(),
            "type": "input"
        })
        
        # Step 2: Fuzzification
        steps.append({
            "step": 2,
            "title": "Fuzzification",
            "description": "Conversion of crisp inputs to membership degrees",
            "details": self._format_fuzzification(),
            "type": "fuzzification"
        })
        
        # Step 3: Rule evaluation
        steps.append({
            "step": 3,
            "title": "Rule Evaluation",
            "description": "Evaluation of fuzzy rules and activation degrees",
            "details": self._format_rule_evaluation(),
            "type": "rules"
        })
        
        # Step 4: Implication and aggregation
        steps.append({
            "step": 4,
            "title": "Implication and Aggregation",
            "description": "Application of Mamdani implication and aggregation",
            "details": self._format_implication_aggregation(),
            "type": "inference"
        })
        
        # Step 5: Defuzzification
        steps.append({
            "step": 5,
            "title": "Defuzzification",
            "description": "Conversion of aggregated membership functions to crisp outputs",
            "details": self._format_defuzzification(),
            "type": "output"
        })
        
        return steps
    
    def _format_input_values(self) -> List[Dict[str, Any]]:
        """Format input values for explanation."""
        details = []
        
        for var_name, value in self.explanation["input_values"].items():
            var = self.config.get_variable(var_name)
            if var is None:
                continue
            
            details.append({
                "variable": var_name,
                "value": value,
                "universe": var.universe,
                "description": f"Input {var_name} = {value} (universe: [{var.universe[0]}, {var.universe[1]}])"
            })
        
        return details
    
    def _format_fuzzification(self) -> List[Dict[str, Any]]:
        """Format fuzzification results for explanation."""
        details = []
        
        for var_name, terms in self.explanation["fuzzification"].items():
            var = self.config.get_variable(var_name)
            if var is None:
                continue
            
            var_details = {
                "variable": var_name,
                "terms": []
            }
            
            for term_name, degree in terms.items():
                var_details["terms"].append({
                    "term": term_name,
                    "membership_degree": degree,
                    "description": f"μ({var_name} is {term_name}) = {degree:.3f}"
                })
            
            details.append(var_details)
        
        return details
    
    def _format_rule_evaluation(self) -> List[Dict[str, Any]]:
        """Format rule evaluation results for explanation."""
        details = []
        
        for rule_id, rule_info in self.explanation["rule_evaluation"].items():
            details.append({
                "rule_id": rule_id,
                "condition": rule_info["condition"],
                "activation_degree": rule_info["activation_degree"],
                "conclusion": rule_info["conclusion"],
                "weight": rule_info["weight"],
                "note": rule_info.get("note"),
                "description": self._create_rule_description(rule_info)
            })
        
        return details
    
    def _create_rule_description(self, rule_info: Dict[str, Any]) -> str:
        """Create human-readable description of rule evaluation."""
        activation = rule_info["activation_degree"]
        weight = rule_info["weight"]
        conclusion = rule_info["conclusion"]
        
        if activation > 0:
            return (f"Rule activated with degree {activation:.3f} "
                   f"(weight: {weight:.3f}) → {conclusion['variable']} is {conclusion['term']}")
        else:
            return f"Rule not activated (activation degree: {activation:.3f})"
    
    def _format_implication_aggregation(self) -> List[Dict[str, Any]]:
        """Format implication and aggregation results for explanation."""
        details = []
        
        for var_name, (x, membership) in self.result.aggregated_functions.items():
            var = self.config.get_variable(var_name)
            if var is None:
                continue
            
            # Find rules affecting this variable
            affecting_rules = [
                rule for rule in self.result.fired_rules
                if rule["conclusion"]["variable"] == var_name
            ]
            
            details.append({
                "variable": var_name,
                "affecting_rules": [rule["id"] for rule in affecting_rules],
                "max_membership": float(np.max(membership)),
                "area": float(np.trapz(membership, x)),
                "description": f"Aggregated function for {var_name} from {len(affecting_rules)} rules"
            })
        
        return details
    
    def _format_defuzzification(self) -> List[Dict[str, Any]]:
        """Format defuzzification results for explanation."""
        details = []
        
        for var_name, crisp_value in self.result.outputs.items():
            var = self.config.get_variable(var_name)
            if var is None:
                continue
            
            # Get linguistic interpretation
            linguistic = self._get_linguistic_interpretation(var_name, crisp_value)
            
            details.append({
                "variable": var_name,
                "crisp_value": crisp_value,
                "defuzz_method": var.defuzz or self.config.logic.defuzz_default,
                "linguistic_interpretation": linguistic,
                "description": f"{var_name} = {crisp_value:.3f} (interpreted as '{linguistic['best_term']}' with confidence {linguistic['confidence']:.3f})"
            })
        
        return details
    
    def _get_linguistic_interpretation(self, var_name: str, crisp_value: float) -> Dict[str, Any]:
        """Get linguistic interpretation of crisp output."""
        var = self.config.get_variable(var_name)
        if var is None:
            return {"best_term": None, "confidence": 0.0}
        
        # Find best matching term
        best_term = None
        best_confidence = 0.0
        all_terms = {}
        
        for term in var.terms:
            # This would need access to membership functions
            # For now, return placeholder
            confidence = 0.0  # Would be calculated from membership function
            all_terms[term.label] = confidence
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_term = term.label
        
        return {
            "best_term": best_term,
            "confidence": best_confidence,
            "all_terms": all_terms
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of inference process."""
        return {
            "total_inputs": len(self.explanation["input_values"]),
            "total_rules": self.explanation["summary"]["total_rules"],
            "fired_rules": self.explanation["summary"]["fired_rules"],
            "max_activation": self.explanation["summary"]["max_activation"],
            "avg_activation": self.explanation["summary"]["avg_activation"],
            "outputs": self.result.outputs,
            "confidence": self._calculate_overall_confidence()
        }
    
    def _calculate_overall_confidence(self) -> float:
        """Calculate overall confidence in the inference result."""
        if not self.result.fired_rules:
            return 0.0
        
        # Use maximum activation degree as overall confidence
        return max(rule["activation_degree"] for rule in self.result.fired_rules)
    
    def get_rule_trace(self, rule_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed trace for a specific rule.
        
        Args:
            rule_id: Rule identifier
            
        Returns:
            Detailed rule trace or None if rule not found
        """
        # Find rule in fired rules
        fired_rule = None
        for rule in self.result.fired_rules:
            if rule["id"] == rule_id:
                fired_rule = rule
                break
        
        if fired_rule is None:
            return None
        
        # Get rule from configuration
        config_rule = None
        for rule in self.config.rules:
            if rule.id == rule_id:
                config_rule = rule
                break
        
        if config_rule is None:
            return None
        
        # Create detailed trace
        trace = {
            "rule_id": rule_id,
            "condition": fired_rule["condition"],
            "conclusion": fired_rule["conclusion"],
            "activation_degree": fired_rule["activation_degree"],
            "weight": fired_rule["weight"],
            "note": fired_rule.get("note"),
            "step_by_step": self._trace_rule_evaluation(rule_id)
        }
        
        return trace
    
    def _trace_rule_evaluation(self, rule_id: str) -> List[Dict[str, Any]]:
        """Create step-by-step trace of rule evaluation."""
        # This would require access to the parsed rule AST
        # For now, return a simplified trace
        return [
            {
                "step": "Parse condition",
                "description": f"Parse rule condition for {rule_id}",
                "result": "Success"
            },
            {
                "step": "Evaluate antecedents",
                "description": "Evaluate all antecedent conditions",
                "result": "Success"
            },
            {
                "step": "Apply logical operators",
                "description": "Apply AND/OR/NOT operators",
                "result": "Success"
            },
            {
                "step": "Apply rule weight",
                "description": "Multiply by rule weight",
                "result": "Success"
            }
        ]
    
    def export_explanation(self, format: str = "text") -> str:
        """
        Export explanation in specified format.
        
        Args:
            format: Export format ("text", "html", "json")
            
        Returns:
            Formatted explanation string
        """
        if format == "text":
            return self._export_text()
        elif format == "html":
            return self._export_html()
        elif format == "json":
            return self._export_json()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_text(self) -> str:
        """Export explanation as plain text."""
        lines = []
        lines.append("FUZZY INFERENCE EXPLANATION")
        lines.append("=" * 50)
        lines.append("")
        
        # Summary
        summary = self.get_summary()
        lines.append("SUMMARY:")
        lines.append(f"  Inputs: {summary['total_inputs']}")
        lines.append(f"  Total rules: {summary['total_rules']}")
        lines.append(f"  Fired rules: {summary['fired_rules']}")
        lines.append(f"  Max activation: {summary['max_activation']:.3f}")
        lines.append(f"  Overall confidence: {summary['confidence']:.3f}")
        lines.append("")
        
        # Step-by-step
        steps = self.get_step_by_step_explanation()
        for step in steps:
            lines.append(f"STEP {step['step']}: {step['title']}")
            lines.append("-" * 30)
            lines.append(step['description'])
            lines.append("")
            
            for detail in step['details']:
                if isinstance(detail, dict):
                    if 'description' in detail:
                        lines.append(f"  • {detail['description']}")
                    elif 'terms' in detail:
                        lines.append(f"  {detail['variable']}:")
                        for term in detail['terms']:
                            lines.append(f"    - {term['description']}")
                else:
                    lines.append(f"  • {detail}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _export_html(self) -> str:
        """Export explanation as HTML."""
        # Simplified HTML export
        html = """
        <html>
        <head>
            <title>Fuzzy Inference Explanation</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .step { margin-bottom: 20px; border: 1px solid #ccc; padding: 10px; }
                .step-title { font-weight: bold; color: #333; }
                .detail { margin-left: 20px; }
            </style>
        </head>
        <body>
            <h1>Fuzzy Inference Explanation</h1>
        """
        
        steps = self.get_step_by_step_explanation()
        for step in steps:
            html += f"""
            <div class="step">
                <div class="step-title">STEP {step['step']}: {step['title']}</div>
                <p>{step['description']}</p>
                <div class="detail">
            """
            
            for detail in step['details']:
                if isinstance(detail, dict) and 'description' in detail:
                    html += f"<p>• {detail['description']}</p>"
            
            html += """
                </div>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _export_json(self) -> str:
        """Export explanation as JSON."""
        import json
        
        data = {
            "summary": self.get_summary(),
            "steps": self.get_step_by_step_explanation(),
            "raw_explanation": self.explanation
        }
        
        return json.dumps(data, indent=2, default=str)


def create_explanation(result: FuzzyInferenceResult, config: FuzzySystemConfig) -> FuzzyExplanation:
    """Convenience function to create explanation."""
    return FuzzyExplanation(result, config)
