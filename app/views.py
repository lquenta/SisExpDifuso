"""
Main views and layout for the fuzzy panel application.
"""

import panel as pn
import param
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path

from core.schema import FuzzySystemConfig, create_example_config
from core.engine import FuzzyInferenceEngine, create_inference_engine
from core.explain import FuzzyExplanation, create_explanation
from fuzzy_io.loader import FuzzyConfigLoader
from fuzzy_io.report import FuzzyReportGenerator
from app.components import (
    InputSliders, MembershipVisualization, RuleTable, 
    OutputDisplay, ExplanationPanel, SystemConfigPanel
)


class FuzzyPanelApp(param.Parameterized):
    """Main fuzzy panel application."""
    
    # Configuration
    config = param.ClassSelector(class_=FuzzySystemConfig, default=create_example_config())
    current_config_path = param.String(default="")
    
    # Input values
    input_values = param.Dict(default={})
    
    # Inference results
    inference_result = param.Parameter(default=None)
    explanation = param.Parameter(default=None)
    
    # UI state
    active_tab = param.String(default="inference")
    
    def __init__(self, **params):
        super().__init__(**params)
        
        # Initialize components
        self.config_loader = FuzzyConfigLoader()
        self.report_generator = FuzzyReportGenerator()
        self.inference_engine = None
        
        # Initialize input values
        self._initialize_input_values()
        
        # Create UI components
        self._create_components()
        
        # Update inference when inputs change
        self.param.watch(self._update_inference, ['input_values', 'config'])
    
    def _initialize_input_values(self):
        """Initialize input values to middle of universe ranges."""
        input_vars = self.config.get_input_variables()
        self.input_values = {}
        
        for var in input_vars:
            min_val, max_val = var.universe
            self.input_values[var.name] = (min_val + max_val) / 2
    
    def _create_components(self):
        """Create UI components."""
        # Input sliders
        self.input_sliders = InputSliders(
            config=self.config,
            input_values=self.input_values
        )
        self.input_sliders.param.watch(self._on_input_change, 'input_values')
        
        # Membership visualization
        self.membership_viz = MembershipVisualization(config=self.config)
        
        # Rule table
        self.rule_table = RuleTable(config=self.config)
        
        # Output display
        self.output_display = OutputDisplay()
        
        # Explanation panel
        self.explanation_panel = ExplanationPanel()
        
        # System configuration panel
        self.config_panel = SystemConfigPanel(
            config=self.config,
            config_loader=self.config_loader
        )
        self.config_panel.param.watch(self._on_config_change, 'config')
    
    def _on_input_change(self, event):
        """Handle input value changes."""
        self.input_values = event.new
        self._update_inference()
    
    def _on_config_change(self, event):
        """Handle configuration changes."""
        self.config = event.new
        self._update_inference()
        self._refresh_components()
    
    def _refresh_components(self):
        """Refresh all components with new configuration."""
        # Update input sliders
        self.input_sliders.config = self.config
        self.input_sliders.input_values = self.input_values
        
        # Update membership visualization
        self.membership_viz.config = self.config
        
        # Update rule table
        self.rule_table.config = self.config
        
        # Update inference engine
        self.inference_engine = create_inference_engine(self.config)
    
    def _update_inference(self, *events):
        """Update inference results."""
        try:
            # Create inference engine if not exists
            if self.inference_engine is None:
                self.inference_engine = create_inference_engine(self.config)
            
            # Run inference
            self.inference_result = self.inference_engine.infer(self.input_values)
            
            # Create explanation
            self.explanation = create_explanation(self.inference_result, self.config)
            
            # Update output display
            self.output_display.update_results(self.inference_result, self.explanation)
            
            # Update explanation panel
            self.explanation_panel.update_explanation(self.explanation)
            
            # Update membership visualization
            self.membership_viz.update_inputs(self.input_values)
            self.membership_viz.update_outputs(self.inference_result)
            
            # Update rule table
            self.rule_table.update_fired_rules(self.inference_result.fired_rules)
            
        except Exception as e:
            # Show error in output display
            self.output_display.show_error(str(e))
    
    def get_layout(self) -> pn.layout.Panel:
        """Get the main application layout."""
        # Create tabs
        tabs = pn.Tabs(
            ("Inferencia y Visualización", self._create_inference_tab()),
            ("Reglas", self._create_rules_tab()),
            ("Configuración", self._create_config_tab()),
            ("Exportar", self._create_export_tab()),
            dynamic=True,
            sizing_mode='stretch_width'
        )
        
        return pn.Column(
            pn.pane.HTML("<h1>Panel de Sistema Experto Difuso</h1>", margin=(10, 10, 0, 10)),
            tabs,
            sizing_mode='stretch_width',
            margin=(0, 10, 10, 10)
        )
    
    def _create_inference_tab(self) -> pn.layout.Panel:
        """Create the inference and visualization tab."""
        return pn.Column(
            pn.Row(
                pn.Column(
                    pn.pane.HTML("<h3>Variables de Entrada</h3>"),
                    self.input_sliders.get_layout(),
                    width=400
                ),
                pn.Column(
                    pn.pane.HTML("<h3>Resultados</h3>"),
                    self.output_display.get_layout(),
                    sizing_mode='stretch_width'
                ),
                sizing_mode='stretch_width'
            ),
            pn.pane.HTML("<h3>Visualización de Funciones de Pertenencia</h3>"),
            self.membership_viz.get_layout(),
            pn.pane.HTML("<h3>Explicación del Proceso</h3>"),
            self.explanation_panel.get_layout(),
            sizing_mode='stretch_width'
        )
    
    
    def _create_rules_tab(self) -> pn.layout.Panel:
        """Create the rules tab."""
        return pn.Column(
            pn.pane.HTML("<h3>Reglas Activas</h3>"),
            self.rule_table.get_layout(),
            sizing_mode='stretch_width'
        )
    
    def _create_config_tab(self) -> pn.layout.Panel:
        """Create the configuration tab."""
        return pn.Column(
            pn.pane.HTML("<h3>Configuración del Sistema</h3>"),
            self.config_panel.get_layout(),
            sizing_mode='stretch_width'
        )
    
    def _create_export_tab(self) -> pn.layout.Panel:
        """Create the export tab."""
        export_buttons = pn.Row(
            pn.widgets.Button(name="Exportar Reporte HTML", button_type="primary"),
            pn.widgets.Button(name="Exportar Reporte PDF", button_type="primary"),
            pn.widgets.Button(name="Exportar Configuración JSON", button_type="primary"),
            pn.widgets.Button(name="Guardar Configuración", button_type="success"),
            pn.widgets.Button(name="Cargar Configuración", button_type="warning")
        )
        
        # Connect export buttons
        export_buttons[0].on_click(self._export_html_report)
        export_buttons[1].on_click(self._export_pdf_report)
        export_buttons[2].on_click(self._export_json_config)
        export_buttons[3].on_click(self._save_configuration)
        export_buttons[4].on_click(self._load_configuration)
        
        return pn.Column(
            pn.pane.HTML("<h3>Opciones de Exportación</h3>"),
            export_buttons,
            pn.pane.HTML("<h3>Información del Sistema</h3>"),
            self._create_system_info_panel(),
            sizing_mode='stretch_width'
        )
    
    def _create_system_info_panel(self) -> pn.layout.Panel:
        """Create system information panel."""
        info_data = {
            "Proyecto": self.config.project,
            "Versión del Esquema": self.config.schema_version,
            "Total de Variables": len(self.config.variables),
            "Variables de Entrada": len(self.config.get_input_variables()),
            "Variables de Salida": len(self.config.get_output_variables()),
            "Total de Reglas": len(self.config.rules),
            "Configuración Lógica": f"AND: {self.config.logic.and_op}, OR: {self.config.logic.or_op}"
        }
        
        import pandas as pd
        info_table = pn.widgets.DataFrame(
            value=pd.DataFrame([{"Propiedad": k, "Valor": str(v)} for k, v in info_data.items()]),
            height=200
        )
        
        return info_table
    
    def _export_html_report(self, event):
        """Export HTML report."""
        if self.inference_result is None:
            pn.state.notifications.error("No inference results to export")
            return
        
        try:
            html_content = self.report_generator.generate_html_report(
                self.inference_result, self.config, self.explanation
            )
            
            # Save to file
            output_path = Path("fuzzy_report.html")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"✓ Reporte HTML guardado en {output_path}")
            
        except Exception as e:
            print(f"✗ Error exportando reporte HTML: {e}")
    
    def _export_pdf_report(self, event):
        """Export PDF report."""
        if self.inference_result is None:
            print("✗ No hay resultados de inferencia para exportar")
            return
        
        try:
            output_path = self.report_generator.generate_pdf_report(
                self.inference_result, self.config, self.explanation,
                output_path="fuzzy_report.pdf"
            )
            
            print(f"✓ Reporte PDF guardado en {output_path}")
            
        except Exception as e:
            print(f"✗ Error exportando reporte PDF: {e}")
    
    def _export_json_config(self, event):
        """Export JSON configuration."""
        try:
            json_content = self.config_loader.save_to_json(self.config)
            
            # Save to file
            output_path = Path("fuzzy_config.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_content)
            
            print(f"✓ Configuración JSON guardada en {output_path}")
            
        except Exception as e:
            print(f"✗ Error exportando configuración JSON: {e}")
    
    def _save_configuration(self, event):
        """Save configuration to file."""
        try:
            # For now, save to default location
            output_path = Path("saved_config.json")
            self.config_loader.save_to_file(self.config, output_path)
            
            print(f"✓ Configuración guardada en {output_path}")
            
        except Exception as e:
            print(f"✗ Error guardando configuración: {e}")
    
    def _load_configuration(self, event):
        """Load configuration from file."""
        try:
            # For now, load from examples
            example_path = Path("fuzzy_panel/examples/politica_arancelaria.json")
            if example_path.exists():
                self.config = self.config_loader.load_from_file(example_path)
                print("✓ Configuración cargada exitosamente")
            else:
                print("✗ Archivo de configuración de ejemplo no encontrado")
                
        except Exception as e:
            print(f"✗ Error cargando configuración: {e}")
