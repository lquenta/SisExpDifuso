"""
Report generation system for fuzzy inference results.
Supports HTML, PDF, and DOCX export formats.
"""

import json
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    from jinja2 import Template, Environment, FileSystemLoader
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False

try:
    from docxtpl import DocxTemplate
    DOCXTPL_AVAILABLE = True
except ImportError:
    DOCXTPL_AVAILABLE = False

from core.engine import FuzzyInferenceResult
from core.explain import FuzzyExplanation
from core.schema import FuzzySystemConfig


class FuzzyReportGenerator:
    """Generator for fuzzy inference reports."""
    
    def __init__(self, template_dir: Optional[Union[str, Path]] = None):
        """
        Initialize report generator.
        
        Args:
            template_dir: Directory containing report templates
        """
        self.template_dir = Path(template_dir) if template_dir else None
        
        if JINJA2_AVAILABLE and self.template_dir:
            self.jinja_env = Environment(loader=FileSystemLoader(self.template_dir))
        else:
            self.jinja_env = None
    
    def generate_html_report(self, result: FuzzyInferenceResult, 
                           config: FuzzySystemConfig,
                           explanation: Optional[FuzzyExplanation] = None,
                           include_charts: bool = True) -> str:
        """
        Generate HTML report.
        
        Args:
            result: Fuzzy inference result
            config: System configuration
            explanation: Optional detailed explanation
            include_charts: Whether to include chart placeholders
            
        Returns:
            HTML report string
        """
        # Prepare data for template
        data = self._prepare_report_data(result, config, explanation)
        data["include_charts"] = include_charts
        data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if self.jinja_env:
            # Use custom template if available
            template = self.jinja_env.get_template("fuzzy_report.html")
            return template.render(**data)
        else:
            # Use built-in template
            return self._generate_builtin_html(data)
    
    def generate_pdf_report(self, result: FuzzyInferenceResult,
                          config: FuzzySystemConfig,
                          explanation: Optional[FuzzyExplanation] = None,
                          output_path: Optional[Union[str, Path]] = None) -> Union[str, bytes]:
        """
        Generate PDF report.
        
        Args:
            result: Fuzzy inference result
            config: System configuration
            explanation: Optional detailed explanation
            output_path: Optional path to save PDF file
            
        Returns:
            PDF content as bytes or file path if saved
            
        Raises:
            ImportError: If weasyprint is not available
        """
        if not WEASYPRINT_AVAILABLE:
            raise ImportError("weasyprint is required for PDF generation")
        
        # Generate HTML first
        html_content = self.generate_html_report(result, config, explanation, include_charts=False)
        
        # Convert to PDF
        html_doc = HTML(string=html_content)
        pdf_content = html_doc.write_pdf()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(pdf_content)
            return str(output_path)
        else:
            return pdf_content
    
    def generate_docx_report(self, result: FuzzyInferenceResult,
                           config: FuzzySystemConfig,
                           explanation: Optional[FuzzyExplanation] = None,
                           template_path: Optional[Union[str, Path]] = None,
                           output_path: Optional[Union[str, Path]] = None) -> Union[str, bytes]:
        """
        Generate DOCX report.
        
        Args:
            result: Fuzzy inference result
            config: System configuration
            explanation: Optional detailed explanation
            template_path: Path to DOCX template
            output_path: Optional path to save DOCX file
            
        Returns:
            DOCX content as bytes or file path if saved
            
        Raises:
            ImportError: If docxtpl is not available
        """
        if not DOCXTPL_AVAILABLE:
            raise ImportError("docxtpl is required for DOCX generation")
        
        # Prepare data for template
        data = self._prepare_report_data(result, config, explanation)
        
        # Load template
        if template_path:
            template = DocxTemplate(template_path)
        else:
            # Use built-in template (would need to be created)
            raise ValueError("DOCX template path is required")
        
        # Render document
        template.render(data)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            template.save(str(output_path))
            return str(output_path)
        else:
            # Return as bytes
            import io
            buffer = io.BytesIO()
            template.save(buffer)
            return buffer.getvalue()
    
    def generate_json_report(self, result: FuzzyInferenceResult,
                           config: FuzzySystemConfig,
                           explanation: Optional[FuzzyExplanation] = None) -> str:
        """
        Generate JSON report.
        
        Args:
            result: Fuzzy inference result
            config: System configuration
            explanation: Optional detailed explanation
            
        Returns:
            JSON report string
        """
        data = self._prepare_report_data(result, config, explanation)
        return json.dumps(data, indent=2, default=str)
    
    def _prepare_report_data(self, result: FuzzyInferenceResult,
                           config: FuzzySystemConfig,
                           explanation: Optional[FuzzyExplanation] = None) -> Dict[str, Any]:
        """Prepare data for report templates."""
        data = {
            "project_name": config.project,
            "schema_version": config.schema_version,
            "timestamp": datetime.now().isoformat(),
            "input_values": result.explanation["input_values"],
            "outputs": result.outputs,
            "summary": result.explanation["summary"],
            "fired_rules": result.fired_rules,
            "membership_values": result.membership_values,
            "system_info": {
                "total_variables": len(config.variables),
                "input_variables": len(config.get_input_variables()),
                "output_variables": len(config.get_output_variables()),
                "total_rules": len(config.rules),
                "logic_config": config.logic.dict()
            }
        }
        
        # Add detailed explanation if available
        if explanation:
            data["explanation"] = {
                "steps": explanation.get_step_by_step_explanation(),
                "summary": explanation.get_summary()
            }
        
        # Add linguistic interpretations
        data["linguistic_outputs"] = self._get_linguistic_outputs(result, config)
        
        # Add system analysis
        data["system_analysis"] = self._analyze_system(config)
        
        return data
    
    def _get_linguistic_outputs(self, result: FuzzyInferenceResult, 
                              config: FuzzySystemConfig) -> Dict[str, Any]:
        """Get linguistic interpretations of outputs."""
        linguistic_outputs = {}
        
        for var_name, crisp_value in result.outputs.items():
            var = config.get_variable(var_name)
            if var is None:
                continue
            
            # Find best matching term
            best_term = None
            best_confidence = 0.0
            all_terms = {}
            
            for term in var.terms:
                # This would need access to membership functions
                # For now, return placeholder
                confidence = 0.0
                all_terms[term.label] = confidence
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_term = term.label
            
            linguistic_outputs[var_name] = {
                "crisp_value": crisp_value,
                "best_term": best_term,
                "confidence": best_confidence,
                "all_terms": all_terms
            }
        
        return linguistic_outputs
    
    def _analyze_system(self, config: FuzzySystemConfig) -> Dict[str, Any]:
        """Analyze system configuration."""
        from ..core.mfuncs import analyze_universe_coverage, analyze_overlap
        
        analysis = {
            "variables": {},
            "rules": {},
            "overall": {}
        }
        
        # Analyze each variable
        for var in config.variables:
            coverage = analyze_universe_coverage([var], tuple(var.universe))
            analysis["variables"][var.name] = {
                "universe": var.universe,
                "term_count": len(var.terms),
                "coverage_percentage": coverage["coverage_percentage"],
                "has_gaps": coverage["has_gaps"]
            }
        
        # Analyze rules
        analysis["rules"] = {
            "total_count": len(config.rules),
            "weighted_rules": sum(1 for r in config.rules if r.weight != 1.0),
            "avg_weight": np.mean([r.weight for r in config.rules])
        }
        
        # Overall analysis
        analysis["overall"] = {
            "complexity": "Simple" if len(config.rules) <= 5 else "Complex",
            "completeness": "Complete" if all(
                analysis["variables"][var.name]["coverage_percentage"] > 90 
                for var in config.variables
            ) else "Incomplete"
        }
        
        return analysis
    
    def _generate_builtin_html(self, data: Dict[str, Any]) -> str:
        """Generate HTML using built-in template."""
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Fuzzy Inference Report - {data['project_name']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .section {{ margin-bottom: 30px; }}
                .section h2 {{ color: #333; border-bottom: 2px solid #007acc; padding-bottom: 5px; }}
                .section h3 {{ color: #555; margin-top: 20px; }}
                .table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .table th {{ background-color: #f2f2f2; }}
                .output {{ background-color: #e8f5e8; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                .rule {{ background-color: #f9f9f9; padding: 10px; border-left: 4px solid #007acc; margin: 5px 0; }}
                .metric {{ display: inline-block; margin: 5px 10px; padding: 5px 10px; background-color: #e1ecf4; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Fuzzy Inference Report</h1>
                <p><strong>Project:</strong> {data['project_name']}</p>
                <p><strong>Generated:</strong> {data['timestamp']}</p>
                <p><strong>Schema Version:</strong> {data['schema_version']}</p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <div class="metric">Total Rules: {data['summary']['total_rules']}</div>
                <div class="metric">Fired Rules: {data['summary']['fired_rules']}</div>
                <div class="metric">Max Activation: {data['summary']['max_activation']:.3f}</div>
                <div class="metric">Avg Activation: {data['summary']['avg_activation']:.3f}</div>
            </div>
            
            <div class="section">
                <h2>Input Values</h2>
                <table class="table">
                    <tr><th>Variable</th><th>Value</th></tr>
        """
        
        for var_name, value in data['input_values'].items():
            html += f"<tr><td>{var_name}</td><td>{value}</td></tr>"
        
        html += """
                </table>
            </div>
            
            <div class="section">
                <h2>Outputs</h2>
        """
        
        for var_name, value in data['outputs'].items():
            linguistic = data['linguistic_outputs'].get(var_name, {})
            best_term = linguistic.get('best_term', 'Unknown')
            confidence = linguistic.get('confidence', 0.0)
            
            html += f"""
                <div class="output">
                    <h3>{var_name}</h3>
                    <p><strong>Crisp Value:</strong> {value:.3f}</p>
                    <p><strong>Linguistic:</strong> {best_term} (confidence: {confidence:.3f})</p>
                </div>
            """
        
        html += """
            </div>
            
            <div class="section">
                <h2>Fired Rules</h2>
        """
        
        for rule in data['fired_rules']:
            html += f"""
                <div class="rule">
                    <h4>Rule {rule['id']}</h4>
                    <p><strong>Condition:</strong> {rule['condition']}</p>
                    <p><strong>Conclusion:</strong> {rule['conclusion']['variable']} is {rule['conclusion']['term']}</p>
                    <p><strong>Activation:</strong> {rule['activation_degree']:.3f}</p>
                    <p><strong>Weight:</strong> {rule['weight']:.3f}</p>
            """
            
            if rule.get('note'):
                html += f"<p><strong>Note:</strong> {rule['note']}</p>"
            
            html += "</div>"
        
        html += """
            </div>
            
            <div class="section">
                <h2>System Information</h2>
                <table class="table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Variables</td><td>{}</td></tr>
                    <tr><td>Input Variables</td><td>{}</td></tr>
                    <tr><td>Output Variables</td><td>{}</td></tr>
                    <tr><td>Total Rules</td><td>{}</td></tr>
                </table>
            </div>
        """.format(
            data['system_info']['total_variables'],
            data['system_info']['input_variables'],
            data['system_info']['output_variables'],
            data['system_info']['total_rules']
        )
        
        html += """
        </body>
        </html>
        """
        
        return html


def generate_report(result: FuzzyInferenceResult, config: FuzzySystemConfig,
                   format: str = "html", **kwargs) -> Union[str, bytes]:
    """
    Convenience function to generate reports.
    
    Args:
        result: Fuzzy inference result
        config: System configuration
        format: Report format ("html", "pdf", "docx", "json")
        **kwargs: Additional arguments for specific formats
        
    Returns:
        Generated report content
    """
    generator = FuzzyReportGenerator()
    
    if format.lower() == "html":
        return generator.generate_html_report(result, config, **kwargs)
    elif format.lower() == "pdf":
        return generator.generate_pdf_report(result, config, **kwargs)
    elif format.lower() == "docx":
        return generator.generate_docx_report(result, config, **kwargs)
    elif format.lower() == "json":
        return generator.generate_json_report(result, config, **kwargs)
    else:
        raise ValueError(f"Unsupported report format: {format}")
