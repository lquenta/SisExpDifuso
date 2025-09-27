"""
UI components for the fuzzy panel application.
"""

import panel as pn
import param
import numpy as np
from typing import Dict, Any, Optional, List
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.palettes import Category10

from core.schema import FuzzySystemConfig
from core.engine import FuzzyInferenceResult
from core.explain import FuzzyExplanation
from core.mfuncs import create_membership_function


class InputSliders(param.Parameterized):
    """Input sliders component for fuzzy variables."""
    
    config = param.ClassSelector(class_=FuzzySystemConfig)
    input_values = param.Dict(default={})
    
    def __init__(self, **params):
        super().__init__(**params)
        self.sliders = {}
        self._create_sliders()
    
    def _create_sliders(self):
        """Create sliders for input variables."""
        self.sliders = {}
        input_vars = self.config.get_input_variables()
        
        for var in input_vars:
            min_val, max_val = var.universe
            current_val = self.input_values.get(var.name, (min_val + max_val) / 2)
            
            slider = pn.widgets.FloatSlider(
                name=var.name,
                start=min_val,
                end=max_val,
                value=current_val,
                step=(max_val - min_val) / 100,
                width=350
            )
            
            self.sliders[var.name] = slider
            slider.param.watch(self._on_slider_change, 'value')
    
    def _on_slider_change(self, event):
        """Handle slider value changes."""
        # Update input values
        new_values = self.input_values.copy()
        new_values[event.obj.name] = event.new
        self.input_values = new_values
    
    def get_layout(self) -> pn.layout.Panel:
        """Get the sliders layout."""
        if not self.sliders:
            return pn.pane.HTML("<p>No hay variables de entrada definidas</p>")
        
        return pn.Column(*self.sliders.values(), sizing_mode='fixed', width=400)


class MembershipVisualization(param.Parameterized):
    """Membership function visualization component."""
    
    config = param.ClassSelector(class_=FuzzySystemConfig)
    input_values = param.Dict(default={})
    output_results = param.Parameter(default=None)
    
    def __init__(self, **params):
        super().__init__(**params)
        self.plots = {}
        self._create_plots()
    
    def _create_plots(self):
        """Create plots for all variables."""
        self.plots = {}
        
        for var in self.config.variables:
            plot = self._create_variable_plot(var)
            self.plots[var.name] = plot
    
    def _create_variable_plot(self, var) -> figure:
        """Create a plot for a single variable."""
        min_val, max_val = var.universe
        x = np.linspace(min_val, max_val, 1001)
        
        plot = figure(
            title=f"{var.name} ({var.kind})",
            x_axis_label="Value",
            y_axis_label="Membership Degree",
            width=400,
            height=300,
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )
        
        # Plot membership functions
        colors = Category10[10]
        for i, term in enumerate(var.terms):
            mf = create_membership_function(term.mf)
            y = mf(x)
            
            plot.line(x, y, line_width=2, color=colors[i % len(colors)], 
                     legend_label=term.label, alpha=0.7)
            # Use patch instead of area for compatibility
            plot.patch(x, y, alpha=0.2, color=colors[i % len(colors)])
        
        plot.legend.location = "top_right"
        plot.legend.click_policy = "hide"
        
        return plot
    
    def update_inputs(self, input_values: Dict[str, float]):
        """Update plots with current input values."""
        self.input_values = input_values
        
        for var_name, value in input_values.items():
            if var_name in self.plots:
                plot = self.plots[var_name]
                
                # Remove existing vertical line
                for renderer in plot.renderers:
                    if hasattr(renderer, 'data_source') and 'input_line' in renderer.data_source.data:
                        plot.renderers.remove(renderer)
                        break
                
                # Add new vertical line
                plot.line([value, value], [0, 1], line_width=3, color='red', 
                         alpha=0.8, legend_label='Current Value')
    
    def update_outputs(self, result: FuzzyInferenceResult):
        """Update plots with output results."""
        self.output_results = result
        
        for var_name, (x, membership) in result.aggregated_functions.items():
            if var_name in self.plots:
                plot = self.plots[var_name]
                
                # Remove existing aggregated function
                for renderer in plot.renderers:
                    if hasattr(renderer, 'data_source') and 'aggregated' in renderer.data_source.data:
                        plot.renderers.remove(renderer)
                        break
                
                # Add aggregated function
                plot.line(x, membership, line_width=3, color='black', 
                         alpha=0.8, legend_label='Aggregated')
                plot.patch(x, membership, alpha=0.3, color='black')
    
    def get_layout(self) -> pn.layout.Panel:
        """Get the visualization layout."""
        if not self.plots:
            return pn.pane.HTML("<p>No hay variables para visualizar</p>")
        
        # Create grid of plots
        plots_list = list(self.plots.values())
        n_cols = 2
        n_rows = (len(plots_list) + n_cols - 1) // n_cols
        
        rows = []
        for i in range(n_rows):
            row_plots = plots_list[i * n_cols:(i + 1) * n_cols]
            rows.append(pn.Row(*row_plots))
        
        return pn.Column(*rows, sizing_mode='stretch_width')


class RuleTable(param.Parameterized):
    """Rule table component."""
    
    config = param.ClassSelector(class_=FuzzySystemConfig)
    fired_rules = param.List(default=[])
    
    def __init__(self, **params):
        super().__init__(**params)
        self.table = None
        self._create_table()
    
    def _create_table(self):
        """Create the rule table."""
        import pandas as pd
        self.table = pn.widgets.DataFrame(
            value=pd.DataFrame(),
            height=400,
            sizing_mode='stretch_width'
        )
    
    def update_fired_rules(self, fired_rules: List[Dict[str, Any]]):
        """Update table with fired rules."""
        self.fired_rules = fired_rules
        
        # Prepare table data
        import pandas as pd
        table_data = []
        for rule in fired_rules:
            table_data.append({
                "ID de Regla": rule["id"],
                "Condición": rule["condition"],
                "Conclusión": f"{rule['conclusion']['variable']} es {rule['conclusion']['term']}",
                "Activación": f"{rule['activation_degree']:.3f}",
                "Peso": f"{rule['weight']:.3f}",
                "Nota": rule.get("note", "")
            })
        
        self.table.value = pd.DataFrame(table_data)
    
    def get_layout(self) -> pn.layout.Panel:
        """Get the table layout."""
        return pn.Column(
            pn.pane.HTML("<h3>Reglas Activadas</h3>"),
            self.table,
            sizing_mode='stretch_width'
        )


class OutputDisplay(param.Parameterized):
    """Output display component."""
    
    def __init__(self, **params):
        super().__init__(**params)
        self.output_cards = []
        self.error_display = pn.pane.HTML("")
    
    def update_results(self, result: FuzzyInferenceResult, explanation: FuzzyExplanation):
        """Update output display with results."""
        self.output_cards = []
        
        # Create output cards
        for var_name, crisp_value in result.outputs.items():
            # Get linguistic interpretation
            linguistic = self._get_linguistic_interpretation(var_name, crisp_value, result)
            
            card = pn.Card(
                pn.Column(
                    pn.pane.HTML(f"<h4>Valor Nítido: {crisp_value:.3f}</h4>"),
                    pn.pane.HTML(f"<p><strong>Lingüístico:</strong> {linguistic['best_term']}</p>"),
                    pn.pane.HTML(f"<p><strong>Confianza:</strong> {linguistic['confidence']:.3f}</p>"),
                    pn.pane.HTML(f"<p><strong>Todos los Términos:</strong></p>"),
                    pn.pane.HTML(self._format_all_terms(linguistic['all_terms']))
                ),
                title=f"Salida: {var_name}",
                collapsible=False,
                width=300
            )
            
            self.output_cards.append(card)
        
        # Clear error display
        self.error_display.object = ""
    
    def _get_linguistic_interpretation(self, var_name: str, crisp_value: float, 
                                     result: FuzzyInferenceResult) -> Dict[str, Any]:
        """Get linguistic interpretation of output."""
        # This would need access to the system configuration
        # For now, return placeholder
        return {
            "best_term": "Unknown",
            "confidence": 0.0,
            "all_terms": {}
        }
    
    def _format_all_terms(self, all_terms: Dict[str, float]) -> str:
        """Format all terms for display."""
        if not all_terms:
            return "<p>No hay información de términos disponible</p>"
        
        html = "<ul>"
        for term, confidence in all_terms.items():
            html += f"<li>{term}: {confidence:.3f}</li>"
        html += "</ul>"
        
        return html
    
    def show_error(self, error_message: str):
        """Show error message."""
        self.error_display.object = f"<div style='color: red; padding: 10px; border: 1px solid red; border-radius: 5px;'>{error_message}</div>"
        self.output_cards = []
    
    def get_layout(self) -> pn.layout.Panel:
        """Get the output display layout."""
        if self.output_cards:
            return pn.Column(
                pn.Row(*self.output_cards),
                sizing_mode='stretch_width'
            )
        else:
            return pn.Column(
                pn.pane.HTML("<p>No hay resultados para mostrar</p>"),
                self.error_display,
                sizing_mode='stretch_width'
            )


class ExplanationPanel(param.Parameterized):
    """Explanation panel component."""
    
    def __init__(self, **params):
        super().__init__(**params)
        self.explanation_display = pn.pane.HTML("")
    
    def update_explanation(self, explanation: FuzzyExplanation):
        """Update explanation display."""
        if explanation is None:
            self.explanation_display.object = "<p>No hay explicación disponible</p>"
            return
        
        # Get step-by-step explanation
        steps = explanation.get_step_by_step_explanation()
        
        # Format as HTML
        html = "<div style='max-height: 400px; overflow-y: auto;'>"
        html += "<h4>Explicación Paso a Paso</h4>"
        
        for step in steps:
            html += f"<div style='margin-bottom: 15px; padding: 10px; border: 1px solid #ddd; border-radius: 5px;'>"
            html += f"<h5>Paso {step['step']}: {step['title']}</h5>"
            html += f"<p>{step['description']}</p>"
            
            if step['details']:
                html += "<ul>"
                for detail in step['details']:
                    if isinstance(detail, dict) and 'description' in detail:
                        html += f"<li>{detail['description']}</li>"
                    elif isinstance(detail, dict) and 'terms' in detail:
                        html += f"<li>{detail['variable']}:"
                        for term in detail['terms']:
                            html += f" {term['description']}"
                        html += "</li>"
                html += "</ul>"
            
            html += "</div>"
        
        html += "</div>"
        
        self.explanation_display.object = html
    
    def get_layout(self) -> pn.layout.Panel:
        """Get the explanation panel layout."""
        return pn.Column(
            self.explanation_display,
            sizing_mode='stretch_width'
        )


class SystemConfigPanel(param.Parameterized):
    """System configuration panel component."""
    
    config = param.ClassSelector(class_=FuzzySystemConfig)
    config_loader = param.Parameter()
    
    def __init__(self, **params):
        super().__init__(**params)
        self.config_display = pn.pane.HTML("")
        self._update_display()
    
    def _update_display(self):
        """Update configuration display."""
        if self.config is None:
            self.config_display.object = "<p>No hay configuración cargada</p>"
            return
        
        # Format configuration as HTML
        html = "<div style='max-height: 500px; overflow-y: auto;'>"
        html += f"<h4>Proyecto: {self.config.project}</h4>"
        html += f"<p><strong>Versión del Esquema:</strong> {self.config.schema_version}</p>"
        
        # Variables
        html += "<h5>Variables</h5>"
        html += "<table border='1' style='border-collapse: collapse; width: 100%;'>"
        html += "<tr><th>Nombre</th><th>Tipo</th><th>Universo</th><th>Términos</th></tr>"
        
        for var in self.config.variables:
            terms_str = ", ".join([term.label for term in var.terms])
            html += f"<tr><td>{var.name}</td><td>{var.kind}</td><td>{var.universe}</td><td>{terms_str}</td></tr>"
        
        html += "</table>"
        
        # Rules
        html += "<h5>Reglas</h5>"
        html += "<table border='1' style='border-collapse: collapse; width: 100%;'>"
        html += "<tr><th>ID</th><th>Condición</th><th>Conclusión</th><th>Peso</th></tr>"
        
        for rule in self.config.rules:
            conclusion_str = ", ".join([f"{c.variable} is {c.term}" for c in rule.then_conclusions])
            html += f"<tr><td>{rule.id}</td><td>{rule.if_condition}</td><td>{conclusion_str}</td><td>{rule.weight}</td></tr>"
        
        html += "</table>"
        html += "</div>"
        
        self.config_display.object = html
    
    def get_layout(self) -> pn.layout.Panel:
        """Get the configuration panel layout."""
        return pn.Column(
            pn.pane.HTML("<h3>Configuración del Sistema</h3>"),
            self.config_display,
            sizing_mode='stretch_width'
        )
