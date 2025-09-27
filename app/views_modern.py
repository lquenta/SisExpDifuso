#!/usr/bin/env python3
"""
Interfaz moderna simplificada para el Sistema Experto Difuso.
"""

import panel as pn
import param
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

from core.schema import FuzzySystemConfig, create_example_config
from core.engine import create_inference_engine, FuzzyInferenceResult
from core.explain import FuzzyExplanation
from fuzzy_io.loader import FuzzyConfigLoader
from core.mfuncs import create_membership_function
from bokeh.plotting import figure
from bokeh.palettes import Category10


class ModernFuzzyPanel(param.Parameterized):
    """Panel moderno y funcional para sistemas expertos difusos."""
    
    # Parámetros principales
    current_config = param.ClassSelector(class_=FuzzySystemConfig)
    inference_result = param.Parameter(default=None)
    explanation = param.Parameter(default=None)
    input_values = param.Dict(default={})
    
    def __init__(self, **params):
        super().__init__(**params)
        
        # Inicializar configuración - cargar desde archivo JSON
        from fuzzy_io.loader import FuzzyConfigLoader
        loader = FuzzyConfigLoader()
        try:
            # Cargar configuración completa desde el archivo JSON
            self.current_config = loader.load_from_file("examples/politica_arancelaria.json")
        except Exception as e:
            print(f"Error cargando configuración: {e}, usando configuración de ejemplo")
            self.current_config = create_example_config()
        
        # Inicializar valores de entrada
        self._initialize_input_values()
        
        # Crear componentes
        self._create_components()
        
        # Crear layout principal
        self.main_layout = self._create_main_layout()
    
    def _initialize_input_values(self):
        """Inicializar valores de entrada."""
        self.input_values = {}
        for var in self.current_config.get_input_variables():
            min_val, max_val = var.universe
            self.input_values[var.name] = (min_val + max_val) / 2
    
    def _create_components(self):
        """Crear componentes de la UI."""
        # Selector de plantillas
        self.template_selector = pn.widgets.Select(
            name="Plantilla",
            options=["Política Arancelaria", "Gestión Inventarios", "Riesgo Operativo"],
            value="Política Arancelaria",
            width=200
        )
        
        # Sliders de entrada
        self.input_sliders = {}
        for var in self.current_config.get_input_variables():
            min_val, max_val = var.universe
            current_val = self.input_values.get(var.name, (min_val + max_val) / 2)
            
            slider = pn.widgets.FloatSlider(
                name=f"{var.name}: {current_val:.1f}",
                start=min_val,
                end=max_val,
                value=current_val,
                step=(max_val - min_val) / 100,
                width=300
            )
            self.input_sliders[var.name] = slider
            slider.param.watch(lambda event, var_name=var.name: self._on_slider_change(event, var_name), 'value')
        
        # Botón de inferencia
        self.inference_button = pn.widgets.Button(
            name="🚀 Ejecutar Inferencia REAL",
            button_type="primary",
            width=250
        )
        self.inference_button.on_click(self._run_inference)
        
        # Área de resultados
        self.results_panel = pn.pane.HTML("", width=400, height=200)
        
        # Área de reglas
        self.rules_panel = pn.pane.HTML("", width=600, height=400)
        
        # Editor de reglas
        self.rule_id_input = pn.widgets.TextInput(name="ID de Regla", value="R4", width=100)
        self.rule_condition_input = pn.widgets.TextAreaInput(
            name="Condición (SI)", 
            value="Déficit is Medio AND Presión is Alta",
            height=100,
            width=400
        )
        self.rule_conclusion_var = pn.widgets.Select(
            name="Variable de Conclusión",
            options=[var.name for var in self.current_config.get_output_variables()],
            value="Tarifa",
            width=150
        )
        self.rule_conclusion_term = pn.widgets.Select(
            name="Término de Conclusión",
            options=["Baja", "Moderada", "Alta"],
            value="Moderada",
            width=150
        )
        
        # Botones de acción
        self.add_rule_button = pn.widgets.Button(
            name="+ Agregar Regla",
            button_type="success",
            width=120
        )
        self.add_rule_button.on_click(self._add_rule)
        
        self.edit_rule_button = pn.widgets.Button(
            name="✏️ Editar Regla",
            button_type="primary",
            width=120
        )
        self.edit_rule_button.on_click(self._edit_rule)
        
        self.delete_rule_button = pn.widgets.Button(
            name="🗑️ Eliminar Regla",
            button_type="danger",
            width=120
        )
        self.delete_rule_button.on_click(self._delete_rule)
        
        # Selector de regla para editar/eliminar
        self.rule_selector = pn.widgets.Select(
            name="Seleccionar Regla",
            options=[rule.id for rule in self.current_config.rules],
            value=self.current_config.rules[0].id if self.current_config.rules else None,
            width=150
        )
        self.rule_selector.param.watch(self._on_rule_selection_change, 'value')
        
        # Actualizar reglas al inicio
        self._update_rules_display()
    
    def _on_slider_change(self, event, var_name):
        """Manejar cambios en sliders."""
        self.input_values[var_name] = event.new
        # Actualizar nombre del slider
        slider = self.input_sliders[var_name]
        slider.name = f"{var_name}: {event.new:.1f}"
    
    def _run_inference(self, event):
        """Ejecutar inferencia difusa."""
        try:
            # Crear motor de inferencia
            engine = create_inference_engine(self.current_config)
            
            # Ejecutar inferencia
            result = engine.infer(self.input_values)
            
            # Actualizar resultados
            self.inference_result = result
            self._update_results_display(result)
            
            # Actualizar reglas activadas
            self._update_rules_display()
            
        except Exception as e:
            self.results_panel.object = f"""
            <div style="background-color: #ffebee; border: 1px solid #f44336; padding: 10px; border-radius: 5px;">
                <h4 style="color: #d32f2f; margin: 0;">❌ Error en Motor Difuso</h4>
                <p style="margin: 5px 0; color: #d32f2f;">{str(e)}</p>
            </div>
            """
    
    def _update_results_display(self, result: FuzzyInferenceResult):
        """Actualizar display de resultados."""
        html = """
        <div style="background-color: #e8f5e8; border: 1px solid #4caf50; padding: 15px; border-radius: 5px;">
            <h3 style="color: #2e7d32; margin: 0 0 10px 0;">🎯 Resultados de Inferencia</h3>
        """
        
        for var_name, value in result.outputs.items():
            html += f"""
            <div style="margin: 10px 0; padding: 10px; background-color: white; border-radius: 3px;">
                <strong>{var_name}:</strong> {value:.3f}
            </div>
            """
        
        html += """
            <div style="margin-top: 15px; font-size: 0.9em; color: #666;">
                <strong>Reglas activadas:</strong> {fired_count}<br>
                <strong>Confianza máxima:</strong> {max_activation:.3f}
            </div>
        </div>
        """.format(
            fired_count=len(result.fired_rules),
            max_activation=max([r["activation_degree"] for r in result.fired_rules]) if result.fired_rules else 0
        )
        
        self.results_panel.object = html
    
    def _update_rules_display(self):
        """Actualizar display de reglas."""
        html = """
        <div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px;">
            <h3 style="margin: 0 0 15px 0;">📋 Reglas del Sistema</h3>
            <table style="width: 100%; border-collapse: collapse; background-color: white; border-radius: 5px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <thead>
                    <tr style="background-color: #2196F3; color: white;">
                        <th style="padding: 12px; text-align: left; border-bottom: 2px solid #1976D2;">ID</th>
                        <th style="padding: 12px; text-align: left; border-bottom: 2px solid #1976D2;">Condición (SI)</th>
                        <th style="padding: 12px; text-align: left; border-bottom: 2px solid #1976D2;">Conclusión (ENTONCES)</th>
                        <th style="padding: 12px; text-align: center; border-bottom: 2px solid #1976D2;">Estado</th>
                        <th style="padding: 12px; text-align: center; border-bottom: 2px solid #1976D2;">Activación</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for i, rule in enumerate(self.current_config.rules):
            # Verificar si la regla está activa
            is_active = False
            activation_degree = 0.0
            
            if self.inference_result:
                for fired_rule in self.inference_result.fired_rules:
                    if fired_rule["id"] == rule.id:
                        is_active = True
                        activation_degree = fired_rule["activation_degree"]
                        break
            
            # Colores alternados para filas
            row_color = "#f8f9fa" if i % 2 == 0 else "white"
            status_color = "#4caf50" if is_active else "#757575"
            status_text = "✅ Activada" if is_active else "⚪ Inactiva"
            activation_text = f"{activation_degree:.3f}" if is_active else "0.000"
            
            html += f"""
                    <tr style="background-color: {row_color};">
                        <td style="padding: 12px; border-bottom: 1px solid #e0e0e0; font-weight: bold; color: #1976D2;">{rule.id}</td>
                        <td style="padding: 12px; border-bottom: 1px solid #e0e0e0;">{rule.if_condition}</td>
                        <td style="padding: 12px; border-bottom: 1px solid #e0e0e0; color: {status_color}; font-weight: bold;">→ {rule.then_conclusions[0].variable} es {rule.then_conclusions[0].term}</td>
                        <td style="padding: 12px; border-bottom: 1px solid #e0e0e0; text-align: center; color: {status_color};">{status_text}</td>
                        <td style="padding: 12px; border-bottom: 1px solid #e0e0e0; text-align: center; font-weight: bold; color: {status_color};">{activation_text}</td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        """
        self.rules_panel.object = html
    
    def _add_rule(self, event):
        """Agregar nueva regla."""
        try:
            # Crear nueva regla
            from core.schema import Rule, RuleConclusion
            
            new_rule = Rule(
                id=self.rule_id_input.value,
                if_condition=self.rule_condition_input.value,
                then_conclusions=[
                    RuleConclusion(
                        variable=self.rule_conclusion_var.value,
                        term=self.rule_conclusion_term.value
                    )
                ],
                weight=1.0,
                note="Regla agregada desde la interfaz"
            )
            
            # Agregar a la configuración
            self.current_config.rules.append(new_rule)
            
            # Actualizar selector de reglas
            self._update_rule_selector()
            
            # Actualizar display
            self._update_rules_display()
            
            # Limpiar campos
            self.rule_condition_input.value = ""
            
        except Exception as e:
            print(f"Error agregando regla: {e}")
    
    def _edit_rule(self, event):
        """Editar regla existente."""
        try:
            if not self.rule_selector.value:
                return
            
            # Buscar la regla a editar
            rule_to_edit = None
            for rule in self.current_config.rules:
                if rule.id == self.rule_selector.value:
                    rule_to_edit = rule
                    break
            
            if not rule_to_edit:
                return
            
            # Actualizar la regla
            from core.schema import RuleConclusion
            
            rule_to_edit.if_condition = self.rule_condition_input.value
            rule_to_edit.then_conclusions = [
                RuleConclusion(
                    variable=self.rule_conclusion_var.value,
                    term=self.rule_conclusion_term.value
                )
            ]
            rule_to_edit.note = "Regla editada desde la interfaz"
            
            # Actualizar display
            self._update_rules_display()
            
            # Limpiar campos
            self.rule_condition_input.value = ""
            
        except Exception as e:
            print(f"Error editando regla: {e}")
    
    def _delete_rule(self, event):
        """Eliminar regla existente."""
        try:
            if not self.rule_selector.value:
                return
            
            # Confirmar eliminación
            if len(self.current_config.rules) <= 1:
                print("No se puede eliminar la última regla del sistema")
                return
            
            # Eliminar la regla
            self.current_config.rules = [
                rule for rule in self.current_config.rules 
                if rule.id != self.rule_selector.value
            ]
            
            # Actualizar selector de reglas
            self._update_rule_selector()
            
            # Actualizar display
            self._update_rules_display()
            
            # Limpiar campos
            self.rule_condition_input.value = ""
            
        except Exception as e:
            print(f"Error eliminando regla: {e}")
    
    def _update_rule_selector(self):
        """Actualizar opciones del selector de reglas."""
        rule_ids = [rule.id for rule in self.current_config.rules]
        self.rule_selector.options = rule_ids
        if rule_ids:
            self.rule_selector.value = rule_ids[0]
        else:
            self.rule_selector.value = None
    
    def _on_rule_selection_change(self, event):
        """Manejar cambio de selección de regla."""
        if not event.new:
            return
        
        # Buscar la regla seleccionada
        selected_rule = None
        for rule in self.current_config.rules:
            if rule.id == event.new:
                selected_rule = rule
                break
        
        if selected_rule:
            # Cargar datos de la regla en los campos
            self.rule_id_input.value = selected_rule.id
            self.rule_condition_input.value = selected_rule.if_condition
            self.rule_conclusion_var.value = selected_rule.then_conclusions[0].variable
            self.rule_conclusion_term.value = selected_rule.then_conclusions[0].term
    
    def _create_main_layout(self) -> pn.layout.Panel:
        """Crear layout principal."""
        
        # Header
        header = pn.pane.HTML("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 20px; text-align: center; border-radius: 10px; margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 2.5em;">🧠 Sistema Experto Difuso</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.2em;">Motor Difuso Real + Editor de Reglas</p>
        </div>
        """, width=800)
        
        # Panel izquierdo - Controles
        left_panel = pn.Column(
            pn.pane.HTML("<h3>🎛️ Control de Entrada</h3>"),
            *self.input_sliders.values(),
            self.inference_button,
            pn.pane.HTML("<br>"),
            self.results_panel,
            width=450,
            sizing_mode='fixed'
        )
        
        # Panel derecho - Reglas y Editor
        right_panel = pn.Column(
            pn.pane.HTML("<h3>📝 Editor de Reglas</h3>"),
            pn.Row(
                self.rule_selector,
                pn.pane.HTML("<br>")
            ),
            pn.Row(
                self.rule_id_input,
                self.rule_conclusion_var,
                self.rule_conclusion_term
            ),
            self.rule_condition_input,
            pn.Row(
                self.add_rule_button,
                self.edit_rule_button,
                self.delete_rule_button
            ),
            pn.pane.HTML("<br>"),
            self.rules_panel,
            width=650,
            sizing_mode='fixed'
        )
        
        # Información del sistema
        info_panel = pn.pane.HTML(f"""
        <div style="background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin-top: 20px;">
            <h4 style="margin: 0 0 10px 0;">ℹ️ Información del Sistema</h4>
            <p><strong>Proyecto:</strong> {self.current_config.project}</p>
            <p><strong>Variables:</strong> {len(self.current_config.variables)}</p>
            <p><strong>Reglas:</strong> {len(self.current_config.rules)}</p>
            <p><strong>Motor:</strong> Difuso Real</p>
        </div>
        """)
        
        # Layout principal
        main_layout = pn.Column(
            header,
            pn.Row(left_panel, right_panel),
            info_panel,
            sizing_mode='stretch_width',
            margin=(0, 20, 20, 20)
        )
        
        return main_layout
    
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

