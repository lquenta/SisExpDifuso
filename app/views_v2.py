#!/usr/bin/env python3
"""
Nueva interfaz moderna para el Panel de Sistema Experto Difuso.
"""

import panel as pn
import param
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

from core.schema import FuzzySystemConfig, create_example_config
from core.engine import create_inference_engine, FuzzyInferenceResult
from core.explain import FuzzyExplanation
from fuzzy_io.loader import FuzzyConfigLoader
from fuzzy_io.report import FuzzyReportGenerator
from app.components_v2 import (
    ProjectBuilder, VariableBuilder, RuleBuilder, 
    InteractiveVisualization, ResultsDashboard, TemplateSelector
)


class ModernFuzzyPanel(param.Parameterized):
    """Panel moderno y funcional para sistemas expertos difusos."""
    
    # Parámetros principales
    current_config = param.ClassSelector(class_=FuzzySystemConfig)
    inference_result = param.Parameter(default=None)
    explanation = param.Parameter(default=None)
    
    # Componentes de la UI
    project_builder = param.Parameter()
    variable_builder = param.Parameter()
    rule_builder = param.Parameter()
    visualization = param.Parameter()
    results_dashboard = param.Parameter()
    template_selector = param.Parameter()
    
    # Utilidades
    config_loader = param.Parameter()
    report_generator = param.Parameter()
    
    def __init__(self, **params):
        super().__init__(**params)
        
        # Inicializar utilidades
        self.config_loader = FuzzyConfigLoader()
        self.report_generator = FuzzyReportGenerator()
        
        # Crear configuración inicial
        self.current_config = create_example_config()
        
        # Inicializar componentes
        self._create_components()
        
        # Crear layout principal
        self.main_layout = self._create_main_layout()
    
    def _create_components(self):
        """Crear todos los componentes de la UI."""
        self.template_selector = TemplateSelector()
        self.project_builder = ProjectBuilder(config=self.current_config)
        self.variable_builder = VariableBuilder(config=self.current_config)
        self.rule_builder = RuleBuilder(config=self.current_config)
        self.visualization = InteractiveVisualization(config=self.current_config)
        self.results_dashboard = ResultsDashboard()
        
        # Conectar callbacks
        self._connect_callbacks()
    
    def _connect_callbacks(self):
        """Conectar callbacks entre componentes."""
        # Cuando cambia la configuración, actualizar todos los componentes
        self.param.watch(self._on_config_change, 'current_config')
        
        # Callbacks de construcción
        self.project_builder.param.watch(self._on_project_change, 'project_data')
        self.variable_builder.param.watch(self._on_variables_change, 'variables_data')
        self.rule_builder.param.watch(self._on_rules_change, 'rules_data')
        
        # Callbacks de visualización
        self.visualization.param.watch(self._on_visualization_update, 'input_values')
    
    def _on_config_change(self, event):
        """Manejar cambios en la configuración."""
        config = event.new
        if config:
            # Actualizar todos los componentes
            self.variable_builder.config = config
            self.rule_builder.config = config
            self.visualization.config = config
    
    def _on_project_change(self, event):
        """Manejar cambios en el proyecto."""
        project_data = event.new
        if project_data and self.current_config:
            # Actualizar configuración del proyecto
            self.current_config.project = project_data.get('name', 'Nuevo Proyecto')
            self.current_config.description = project_data.get('description', '')
            self.current_config.author = project_data.get('author', '')
    
    def _on_variables_change(self, event):
        """Manejar cambios en las variables."""
        variables_data = event.new
        if variables_data and self.current_config:
            # Reconstruir configuración con nuevas variables
            self._rebuild_config_from_components()
    
    def _on_rules_change(self, event):
        """Manejar cambios en las reglas."""
        rules_data = event.new
        if rules_data and self.current_config:
            # Reconstruir configuración con nuevas reglas
            self._rebuild_config_from_components()
    
    def _on_visualization_update(self, event):
        """Manejar actualizaciones de visualización."""
        input_values = event.new
        if input_values and self.current_config:
            # Ejecutar inferencia
            self._run_inference(input_values)
    
    def _rebuild_config_from_components(self):
        """Reconstruir configuración desde los componentes."""
        try:
            # Obtener datos de los componentes
            project_data = self.project_builder.project_data
            variables_data = self.variable_builder.variables_data
            rules_data = self.rule_builder.rules_data
            
            # Crear nueva configuración
            new_config = FuzzySystemConfig(
                project=project_data.get('name', 'Nuevo Proyecto'),
                description=project_data.get('description', ''),
                author=project_data.get('author', ''),
                schema_version="1.0",
                variables=variables_data,
                rules=rules_data,
                logic=self.current_config.logic,
                defuzzification=self.current_config.defuzzification
            )
            
            self.current_config = new_config
            
        except Exception as e:
            print(f"Error reconstruyendo configuración: {e}")
    
    def _run_inference(self, input_values: Dict[str, float]):
        """Ejecutar inferencia difusa."""
        try:
            # Crear motor de inferencia
            engine = create_inference_engine(self.current_config)
            
            # Ejecutar inferencia
            result = engine.infer(input_values)
            
            # Generar explicación
            explanation = engine.explain_inference(input_values)
            
            # Actualizar resultados
            self.inference_result = result
            self.explanation = explanation
            
            # Actualizar dashboard
            self.results_dashboard.update_results(result, explanation)
            
        except Exception as e:
            print(f"Error en inferencia: {e}")
            self.results_dashboard.show_error(str(e))
    
    def _create_main_layout(self) -> pn.layout.Panel:
        """Crear el layout principal de la aplicación."""
        
        # Header con título y acciones principales
        header = self._create_header()
        
        # Pestañas principales
        tabs = pn.Tabs(
            ("🏗️ Constructor", self._create_builder_tab()),
            ("🎯 Inferencia", self._create_inference_tab()),
            ("📊 Análisis", self._create_analysis_tab()),
            ("⚙️ Configuración", self._create_settings_tab()),
            dynamic=True,
            sizing_mode='stretch_width'
        )
        
        # Layout principal
        main_layout = pn.Column(
            header,
            tabs,
            sizing_mode='stretch_width',
            margin=(0, 20, 20, 20)
        )
        
        return main_layout
    
    def _create_header(self) -> pn.layout.Panel:
        """Crear header de la aplicación."""
        # Título principal
        title = pn.pane.HTML(
            """
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
                <h1 style="margin: 0; font-size: 2.5em; font-weight: 300;">🧠 Sistema Experto Difuso</h1>
                <p style="margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9;">Constructor Visual Inteligente</p>
            </div>
            """,
            sizing_mode='stretch_width'
        )
        
        # Barra de acciones rápidas
        quick_actions = pn.Row(
            pn.widgets.Button(name="📁 Nuevo Proyecto", button_type="primary", width=150),
            pn.widgets.Button(name="💾 Guardar", button_type="success", width=120),
            pn.widgets.Button(name="📂 Cargar", button_type="warning", width=120),
            pn.widgets.Button(name="📋 Plantillas", button_type="light", width=120),
            pn.widgets.Button(name="📊 Exportar", button_type="light", width=120),
            sizing_mode='stretch_width',
            margin=(0, 0, 20, 0)
        )
        
        # Conectar botones
        quick_actions[0].on_click(self._new_project)
        quick_actions[1].on_click(self._save_project)
        quick_actions[2].on_click(self._load_project)
        quick_actions[3].on_click(self._show_templates)
        quick_actions[4].on_click(self._export_project)
        
        return pn.Column(title, quick_actions, sizing_mode='stretch_width')
    
    def _create_builder_tab(self) -> pn.layout.Panel:
        """Crear pestaña del constructor."""
        return pn.Column(
            pn.pane.HTML("<h2>🏗️ Constructor de Sistema Experto</h2>"),
            
            # Selector de plantillas
            pn.Card(
                self.template_selector.get_layout(),
                title="📋 Plantillas Predefinidas",
                collapsible=True,
                collapsed=False
            ),
            
            # Constructor de proyecto
            pn.Card(
                self.project_builder.get_layout(),
                title="📝 Información del Proyecto",
                collapsible=True,
                collapsed=False
            ),
            
            # Constructor de variables
            pn.Card(
                self.variable_builder.get_layout(),
                title="🔧 Variables del Sistema",
                collapsible=True,
                collapsed=False
            ),
            
            # Constructor de reglas
            pn.Card(
                self.rule_builder.get_layout(),
                title="📏 Reglas de Inferencia",
                collapsible=True,
                collapsed=False
            ),
            
            sizing_mode='stretch_width'
        )
    
    def _create_inference_tab(self) -> pn.layout.Panel:
        """Crear pestaña de inferencia."""
        return pn.Column(
            pn.pane.HTML("<h2>🎯 Inferencia y Visualización</h2>"),
            
            pn.Row(
                # Panel de entrada
                pn.Column(
                    pn.pane.HTML("<h3>📊 Valores de Entrada</h3>"),
                    self.visualization.get_input_panel(),
                    width=400
                ),
                
                # Panel de resultados
                pn.Column(
                    pn.pane.HTML("<h3>🎯 Resultados</h3>"),
                    self.results_dashboard.get_layout(),
                    sizing_mode='stretch_width'
                ),
                sizing_mode='stretch_width'
            ),
            
            # Visualización interactiva
            pn.pane.HTML("<h3>📈 Visualización Dinámica</h3>"),
            self.visualization.get_layout(),
            
            sizing_mode='stretch_width'
        )
    
    def _create_analysis_tab(self) -> pn.layout.Panel:
        """Crear pestaña de análisis."""
        return pn.Column(
            pn.pane.HTML("<h2>📊 Análisis y Reportes</h2>"),
            
            pn.Row(
                pn.widgets.Button(name="📈 Análisis de Sensibilidad", button_type="primary"),
                pn.widgets.Button(name="🔍 Validación de Reglas", button_type="success"),
                pn.widgets.Button(name="📋 Reporte Completo", button_type="warning"),
                sizing_mode='stretch_width'
            ),
            
            pn.pane.HTML("<p>Funciones de análisis avanzado (próximamente)</p>"),
            
            sizing_mode='stretch_width'
        )
    
    def _create_settings_tab(self) -> pn.layout.Panel:
        """Crear pestaña de configuración."""
        return pn.Column(
            pn.pane.HTML("<h2>⚙️ Configuración del Sistema</h2>"),
            
            pn.Card(
                pn.Column(
                    pn.widgets.Select(
                        name="Método de Defuzzificación",
                        options=["centroid", "bisector", "mean_of_maxima"],
                        value="centroid"
                    ),
                    pn.widgets.Select(
                        name="Operador AND",
                        options=["min", "product"],
                        value="min"
                    ),
                    pn.widgets.Select(
                        name="Operador OR",
                        options=["max", "sum"],
                        value="max"
                    ),
                ),
                title="🔧 Parámetros de Inferencia",
                collapsible=True
            ),
            
            pn.Card(
                pn.Column(
                    pn.widgets.TextInput(name="Nombre del Proyecto", value=self.current_config.project),
                    pn.widgets.TextAreaInput(name="Descripción", value=self.current_config.description, height=100),
                    pn.widgets.TextInput(name="Autor", value=self.current_config.author),
                ),
                title="📝 Información del Proyecto",
                collapsible=True
            ),
            
            sizing_mode='stretch_width'
        )
    
    # Métodos de acciones rápidas
    def _new_project(self, event):
        """Crear nuevo proyecto."""
        self.current_config = create_example_config()
        self._create_components()
        print("✅ Nuevo proyecto creado")
    
    def _save_project(self, event):
        """Guardar proyecto."""
        try:
            output_path = Path("mi_proyecto_difuso.json")
            self.config_loader.save_to_file(self.current_config, output_path)
            print(f"✅ Proyecto guardado en {output_path}")
        except Exception as e:
            print(f"❌ Error guardando proyecto: {e}")
    
    def _load_project(self, event):
        """Cargar proyecto."""
        try:
            # Por ahora cargar ejemplo
            example_path = Path("fuzzy_panel/examples/politica_arancelaria.json")
            if example_path.exists():
                self.current_config = self.config_loader.load_from_file(example_path)
                self._create_components()
                print("✅ Proyecto cargado")
            else:
                print("❌ Archivo no encontrado")
        except Exception as e:
            print(f"❌ Error cargando proyecto: {e}")
    
    def _show_templates(self, event):
        """Mostrar plantillas."""
        print("📋 Mostrando plantillas disponibles...")
    
    def _export_project(self, event):
        """Exportar proyecto."""
        try:
            # Generar reporte HTML
            if self.inference_result:
                output_path = self.report_generator.generate_html_report(
                    self.inference_result, self.current_config, self.explanation,
                    output_path="reporte_difuso.html"
                )
                print(f"✅ Reporte exportado a {output_path}")
            else:
                print("❌ No hay resultados para exportar")
        except Exception as e:
            print(f"❌ Error exportando: {e}")
    
    def get_layout(self) -> pn.layout.Panel:
        """Obtener el layout principal."""
        return self.main_layout


def create_modern_app():
    """Crear la aplicación moderna."""
    # Configurar Panel
    pn.extension('tabulator', sizing_mode='stretch_width')
    
    # Crear aplicación
    app = ModernFuzzyPanel()
    
    return app.get_layout()
