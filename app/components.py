#!/usr/bin/env python3
"""
Componentes modernos para la interfaz del Sistema Experto Difuso.
"""

import panel as pn
import param
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

from core.schema import FuzzySystemConfig, Variable, Term, Rule, RuleConclusion
from core.mfuncs import create_membership_function
from core.engine import create_inference_engine
from core.explain import FuzzyExplanation


class TemplateSelector(param.Parameterized):
    """Selector de plantillas predefinidas."""
    
    selected_template = param.String(default="")
    available_templates = param.List(default=[])
    
    def __init__(self, **params):
        super().__init__(**params)
        self.available_templates = [
            "Política Arancelaria",
            "Gestión de Inventarios", 
            "Riesgo Operativo",
            "Control de Calidad",
            "Sistema de Recomendación"
        ]
        self._create_selector()
    
    def _create_selector(self):
        """Crear selector de plantillas."""
        self.template_dropdown = pn.widgets.Select(
            name="Seleccionar Plantilla",
            options=[""] + self.available_templates,
            value="",
            width=300
        )
        
        self.load_button = pn.widgets.Button(
            name="Cargar Plantilla",
            button_type="primary",
            width=150
        )
        
        self.load_button.on_click(self._load_template)
    
    def _load_template(self, event):
        """Cargar plantilla seleccionada."""
        template = self.template_dropdown.value
        if template:
            print(f"📋 Cargando plantilla: {template}")
            # Aquí se cargaría la plantilla específica
    
    def get_layout(self) -> pn.layout.Panel:
        """Obtener layout del selector."""
        return pn.Row(
            self.template_dropdown,
            self.load_button,
            sizing_mode='fixed'
        )


class ProjectBuilder(param.Parameterized):
    """Constructor de información del proyecto."""
    
    config = param.ClassSelector(class_=FuzzySystemConfig)
    project_data = param.Dict(default={})
    
    def __init__(self, **params):
        super().__init__(**params)
        self._create_form()
    
    def _create_form(self):
        """Crear formulario del proyecto."""
        self.name_input = pn.widgets.TextInput(
            name="Nombre del Proyecto",
            value=self.config.project if self.config else "Mi Sistema Difuso",
            width=400
        )
        
        self.description_input = pn.widgets.TextAreaInput(
            name="Descripción",
            value=self.config.description if self.config else "",
            height=80,
            width=400
        )
        
        self.author_input = pn.widgets.TextInput(
            name="Autor",
            value=self.config.author if self.config else "",
            width=400
        )
        
        self.version_input = pn.widgets.TextInput(
            name="Versión",
            value="1.0",
            width=200
        )
        
        # Conectar callbacks
        self.name_input.param.watch(self._update_project_data, 'value')
        self.description_input.param.watch(self._update_project_data, 'value')
        self.author_input.param.watch(self._update_project_data, 'value')
        self.version_input.param.watch(self._update_project_data, 'value')
    
    def _update_project_data(self, event):
        """Actualizar datos del proyecto."""
        self.project_data = {
            'name': self.name_input.value,
            'description': self.description_input.value,
            'author': self.author_input.value,
            'version': self.version_input.value
        }
    
    def get_layout(self) -> pn.layout.Panel:
        """Obtener layout del constructor."""
        return pn.Column(
            self.name_input,
            self.description_input,
            pn.Row(self.author_input, self.version_input),
            sizing_mode='fixed'
        )


class VariableBuilder(param.Parameterized):
    """Constructor visual de variables."""
    
    config = param.ClassSelector(class_=FuzzySystemConfig)
    variables_data = param.List(default=[])
    current_variable = param.Parameter(default=None)
    
    def __init__(self, **params):
        super().__init__(**params)
        self.variable_editor = None
        self._create_interface()
    
    def _create_interface(self):
        """Crear interfaz del constructor."""
        # Lista de variables
        self.variables_list = pn.widgets.DataFrame(
            value=pd.DataFrame(),
            height=200,
            width=600
        )
        
        # Botones de acción
        self.add_button = pn.widgets.Button(
            name="➕ Agregar Variable",
            button_type="primary",
            width=150
        )
        
        self.edit_button = pn.widgets.Button(
            name="✏️ Editar",
            button_type="warning",
            width=100
        )
        
        self.delete_button = pn.widgets.Button(
            name="🗑️ Eliminar",
            button_type="danger",
            width=100
        )
        
        # Conectar callbacks
        self.add_button.on_click(self._add_variable)
        self.edit_button.on_click(self._edit_variable)
        self.delete_button.on_click(self._delete_variable)
        
        # Actualizar lista
        self._update_variables_list()
    
    def _update_variables_list(self):
        """Actualizar lista de variables."""
        if self.config and self.config.variables:
            data = []
            for var in self.config.variables:
                terms_str = ", ".join([term.label for term in var.terms])
                data.append({
                    "Nombre": var.name,
                    "Tipo": var.kind,
                    "Universo": f"[{var.universe[0]}, {var.universe[1]}]",
                    "Términos": terms_str
                })
            
            self.variables_list.value = pd.DataFrame(data)
    
    def _add_variable(self, event):
        """Agregar nueva variable."""
        self._show_variable_editor()
    
    def _edit_variable(self, event):
        """Editar variable seleccionada."""
        if not self.variables_list.selection:
            print("❌ Selecciona una variable para editar")
            return
        
        # Obtener variable seleccionada
        selected_idx = self.variables_list.selection[0]
        if selected_idx < len(self.config.variables):
            self.current_variable = self.config.variables[selected_idx]
            self._show_variable_editor()
    
    def _delete_variable(self, event):
        """Eliminar variable seleccionada."""
        if not self.variables_list.selection:
            print("❌ Selecciona una variable para eliminar")
            return
        
        selected_idx = self.variables_list.selection[0]
        if selected_idx < len(self.config.variables):
            del self.config.variables[selected_idx]
            self._update_variables_list()
            print("✅ Variable eliminada")
    
    def _show_variable_editor(self):
        """Mostrar editor de variable."""
        # Crear formulario de variable
        self.variable_editor = VariableEditor(
            variable=self.current_variable,
            on_save=self._save_variable
        )
        
        # Mostrar en modal o panel
        print("📝 Abriendo editor de variable...")
    
    def _save_variable(self, variable_data):
        """Guardar variable."""
        try:
            if self.current_variable:
                # Editar variable existente
                idx = self.config.variables.index(self.current_variable)
                self.config.variables[idx] = self._create_variable_from_data(variable_data)
            else:
                # Agregar nueva variable
                new_var = self._create_variable_from_data(variable_data)
                self.config.variables.append(new_var)
            
            self._update_variables_list()
            self.current_variable = None
            print("✅ Variable guardada")
            
        except Exception as e:
            print(f"❌ Error guardando variable: {e}")
    
    def _create_variable_from_data(self, data):
        """Crear variable desde datos del formulario."""
        # Crear términos
        terms = []
        for term_data in data['terms']:
            term = Term(
                label=term_data['label'],
                mf={
                    'type': 'triangular',
                    'params': term_data['params']
                }
            )
            terms.append(term)
        
        # Crear variable
        variable = Variable(
            name=data['name'],
            kind=data['kind'],
            universe=data['universe'],
            terms=terms
        )
        
        return variable
    
    def get_layout(self) -> pn.layout.Panel:
        """Obtener layout del constructor."""
        return pn.Column(
            pn.Row(
                self.add_button,
                self.edit_button,
                self.delete_button,
                sizing_mode='fixed'
            ),
            self.variables_list,
            sizing_mode='fixed'
        )


class VariableEditor(param.Parameterized):
    """Editor de variable individual."""
    
    variable = param.Parameter(default=None)
    on_save = param.Callable()
    
    def __init__(self, **params):
        super().__init__(**params)
        self._create_form()
    
    def _create_form(self):
        """Crear formulario de edición."""
        # Datos iniciales
        if self.variable:
            name = self.variable.name
            kind = self.variable.kind
            universe = self.variable.universe
        else:
            name = ""
            kind = "input"
            universe = [0, 100]
        
        self.name_input = pn.widgets.TextInput(
            name="Nombre de la Variable",
            value=name,
            width=300
        )
        
        self.kind_select = pn.widgets.Select(
            name="Tipo",
            options=["input", "output"],
            value=kind,
            width=200
        )
        
        self.universe_min = pn.widgets.NumberInput(
            name="Mínimo",
            value=universe[0],
            width=150
        )
        
        self.universe_max = pn.widgets.NumberInput(
            name="Máximo",
            value=universe[1],
            width=150
        )
        
        # Lista de términos
        self.terms_list = pn.widgets.DataFrame(
            value=pd.DataFrame(),
            height=150,
            width=500
        )
        
        # Botones de términos
        self.add_term_button = pn.widgets.Button(
            name="➕ Agregar Término",
            button_type="primary",
            width=150
        )
        
        self.save_button = pn.widgets.Button(
            name="💾 Guardar Variable",
            button_type="success",
            width=150
        )
        
        # Conectar callbacks
        self.save_button.on_click(self._save_variable)
        self.add_term_button.on_click(self._add_term)
        
        # Actualizar términos
        self._update_terms_list()
    
    def _update_terms_list(self):
        """Actualizar lista de términos."""
        if self.variable and self.variable.terms:
            data = []
            for term in self.variable.terms:
                params = term.mf.params
                data.append({
                    "Etiqueta": term.label,
                    "Tipo": term.mf.type,
                    "Parámetros": str(params)
                })
            
            self.terms_list.value = pd.DataFrame(data)
        else:
            self.terms_list.value = pd.DataFrame()
    
    def _add_term(self, event):
        """Agregar nuevo término."""
        print("📝 Agregando nuevo término...")
        # Aquí se abriría un editor de términos
    
    def _save_variable(self, event):
        """Guardar variable."""
        try:
            variable_data = {
                'name': self.name_input.value,
                'kind': self.kind_select.value,
                'universe': [self.universe_min.value, self.universe_max.value],
                'terms': []  # Por ahora vacío
            }
            
            if self.on_save:
                self.on_save(variable_data)
                
        except Exception as e:
            print(f"❌ Error guardando variable: {e}")
    
    def get_layout(self) -> pn.layout.Panel:
        """Obtener layout del editor."""
        return pn.Column(
            pn.Row(
                self.name_input,
                self.kind_select,
                sizing_mode='fixed'
            ),
            pn.Row(
                self.universe_min,
                self.universe_max,
                sizing_mode='fixed'
            ),
            pn.pane.HTML("<h4>Términos Lingüísticos</h4>"),
            self.terms_list,
            pn.Row(
                self.add_term_button,
                self.save_button,
                sizing_mode='fixed'
            ),
            sizing_mode='fixed'
        )


class RuleBuilder(param.Parameterized):
    """Constructor visual de reglas."""
    
    config = param.ClassSelector(class_=FuzzySystemConfig)
    rules_data = param.List(default=[])
    
    def __init__(self, **params):
        super().__init__(**params)
        self._create_interface()
    
    def _create_interface(self):
        """Crear interfaz del constructor."""
        # Lista de reglas
        self.rules_list = pn.widgets.DataFrame(
            value=pd.DataFrame(),
            height=300,
            width=800
        )
        
        # Constructor de reglas
        self.rule_constructor = RuleConstructor(config=self.config)
        
        # Botones de acción
        self.add_button = pn.widgets.Button(
            name="➕ Agregar Regla",
            button_type="primary",
            width=150
        )
        
        self.edit_button = pn.widgets.Button(
            name="✏️ Editar",
            button_type="warning",
            width=100
        )
        
        self.delete_button = pn.widgets.Button(
            name="🗑️ Eliminar",
            button_type="danger",
            width=100
        )
        
        # Conectar callbacks
        self.add_button.on_click(self._add_rule)
        self.edit_button.on_click(self._edit_rule)
        self.delete_button.on_click(self._delete_rule)
        
        # Actualizar lista
        self._update_rules_list()
    
    def _update_rules_list(self):
        """Actualizar lista de reglas."""
        if self.config and self.config.rules:
            data = []
            for rule in self.config.rules:
                conclusion_str = ", ".join([f"{c.variable} es {c.term}" for c in rule.then_conclusions])
                data.append({
                    "ID": rule.id,
                    "Si": rule.if_condition,
                    "Entonces": conclusion_str,
                    "Peso": rule.weight,
                    "Nota": rule.note or ""
                })
            
            self.rules_list.value = pd.DataFrame(data)
    
    def _add_rule(self, event):
        """Agregar nueva regla."""
        print("📝 Agregando nueva regla...")
    
    def _edit_rule(self, event):
        """Editar regla seleccionada."""
        if not self.rules_list.selection:
            print("❌ Selecciona una regla para editar")
            return
        
        print("✏️ Editando regla...")
    
    def _delete_rule(self, event):
        """Eliminar regla seleccionada."""
        if not self.rules_list.selection:
            print("❌ Selecciona una regla para eliminar")
            return
        
        selected_idx = self.rules_list.selection[0]
        if selected_idx < len(self.config.rules):
            del self.config.rules[selected_idx]
            self._update_rules_list()
            print("✅ Regla eliminada")
    
    def get_layout(self) -> pn.layout.Panel:
        """Obtener layout del constructor."""
        return pn.Column(
            pn.Row(
                self.add_button,
                self.edit_button,
                self.delete_button,
                sizing_mode='fixed'
            ),
            self.rules_list,
            pn.pane.HTML("<h4>Constructor de Reglas</h4>"),
            self.rule_constructor.get_layout(),
            sizing_mode='fixed'
        )


class RuleConstructor(param.Parameterized):
    """Constructor visual de reglas individuales."""
    
    config = param.ClassSelector(class_=FuzzySystemConfig)
    
    def __init__(self, **params):
        super().__init__(**params)
        self._create_interface()
    
    def _create_interface(self):
        """Crear interfaz del constructor."""
        # Variables disponibles
        input_vars = self.config.get_input_variables() if self.config else []
        output_vars = self.config.get_output_variables() if self.config else []
        
        # Constructor de condición
        self.condition_builder = pn.pane.HTML(
            "<p>Constructor de condiciones (próximamente)</p>",
            width=400
        )
        
        # Constructor de conclusión
        self.conclusion_builder = pn.pane.HTML(
            "<p>Constructor de conclusiones (próximamente)</p>",
            width=400
        )
        
        # Botón de guardar
        self.save_rule_button = pn.widgets.Button(
            name="💾 Guardar Regla",
            button_type="success",
            width=150
        )
    
    def get_layout(self) -> pn.layout.Panel:
        """Obtener layout del constructor."""
        return pn.Column(
            pn.Row(
                self.condition_builder,
                self.conclusion_builder,
                sizing_mode='fixed'
            ),
            self.save_rule_button,
            sizing_mode='fixed'
        )


class InteractiveVisualization(param.Parameterized):
    """Visualización interactiva mejorada."""
    
    config = param.ClassSelector(class_=FuzzySystemConfig)
    input_values = param.Dict(default={})
    plots = param.Dict(default={})
    
    def __init__(self, **params):
        super().__init__(**params)
        self._create_visualization()
    
    def _create_visualization(self):
        """Crear visualización."""
        self.plots = {}
        self.input_sliders = {}
        
        if self.config:
            self._create_input_sliders()
            self._create_membership_plots()
    
    def _create_input_sliders(self):
        """Crear sliders de entrada."""
        input_vars = self.config.get_input_variables()
        
        for var in input_vars:
            min_val, max_val = var.universe
            current_val = self.input_values.get(var.name, (min_val + max_val) / 2)
            
            slider = pn.widgets.FloatSlider(
                name=f"📊 {var.name}",
                start=min_val,
                end=max_val,
                value=current_val,
                step=(max_val - min_val) / 100,
                width=350,
                bar_color='#1f77b4'
            )
            
            self.input_sliders[var.name] = slider
            slider.param.watch(self._on_slider_change, 'value')
    
    def _create_membership_plots(self):
        """Crear gráficos de funciones de pertenencia."""
        from bokeh.plotting import figure
        from bokeh.palettes import Category10
        
        for var in self.config.variables:
            min_val, max_val = var.universe
            x = np.linspace(min_val, max_val, 1001)
            
            plot = figure(
                title=f"🎯 {var.name} ({var.kind})",
                x_axis_label="Valor",
                y_axis_label="Grado de Pertenencia",
                width=400,
                height=300,
                tools="pan,wheel_zoom,box_zoom,reset,save",
                toolbar_location="above"
            )
            
            # Colores
            colors = Category10[10]
            
            # Plotear funciones de pertenencia
            for i, term in enumerate(var.terms):
                mf = create_membership_function(term.mf)
                y = mf(x)
                
                plot.line(x, y, line_width=3, color=colors[i % len(colors)], 
                         legend_label=term.label, alpha=0.8)
                plot.patch(x, y, alpha=0.2, color=colors[i % len(colors)])
            
            plot.legend.location = "top_right"
            plot.legend.click_policy = "hide"
            
            self.plots[var.name] = plot
    
    def _on_slider_change(self, event):
        """Manejar cambios en sliders."""
        new_values = self.input_values.copy()
        new_values[event.obj.name] = event.new
        self.input_values = new_values
        
        # Actualizar visualización
        self._update_visualization()
    
    def _update_visualization(self):
        """Actualizar visualización con valores actuales."""
        # Aquí se actualizarían las líneas verticales en los gráficos
        # para mostrar los valores actuales de entrada
        pass
    
    def get_input_panel(self) -> pn.layout.Panel:
        """Obtener panel de entrada."""
        if not self.input_sliders:
            return pn.pane.HTML("<p>No hay variables de entrada definidas</p>")
        
        return pn.Column(*self.input_sliders.values(), sizing_mode='fixed')
    
    def get_layout(self) -> pn.layout.Panel:
        """Obtener layout de visualización."""
        if not self.plots:
            return pn.pane.HTML("<p>No hay variables para visualizar</p>")
        
        # Crear grid de gráficos
        plots_list = list(self.plots.values())
        n_cols = 2
        n_rows = (len(plots_list) + n_cols - 1) // n_cols
        
        grid = []
        for i in range(n_rows):
            row_plots = plots_list[i*n_cols:(i+1)*n_cols]
            if len(row_plots) < n_cols:
                row_plots.extend([None] * (n_cols - len(row_plots)))
            grid.append(pn.Row(*[p for p in row_plots if p is not None], sizing_mode='stretch_width'))
        
        return pn.Column(*grid, sizing_mode='stretch_width')


class ResultsDashboard(param.Parameterized):
    """Dashboard de resultados mejorado."""
    
    results = param.Parameter(default=None)
    explanation = param.Parameter(default=None)
    
    def __init__(self, **params):
        super().__init__(**params)
        self._create_dashboard()
    
    def _create_dashboard(self):
        """Crear dashboard."""
        self.results_cards = []
        self.explanation_display = pn.pane.HTML("")
        self.error_display = pn.pane.HTML("")
    
    def update_results(self, result, explanation):
        """Actualizar resultados."""
        self.results = result
        self.explanation = explanation
        
        # Crear tarjetas de resultados
        self._create_results_cards()
        
        # Actualizar explicación
        self._update_explanation()
    
    def _create_results_cards(self):
        """Crear tarjetas de resultados."""
        self.results_cards = []
        
        if not self.results:
            return
        
        for var_name, crisp_value in self.results.outputs.items():
            # Crear tarjeta de resultado
            card = pn.Card(
                pn.Column(
                    pn.pane.HTML(f"<h2 style='color: #2E8B57; margin: 0;'>{crisp_value:.3f}</h2>"),
                    pn.pane.HTML(f"<p style='margin: 5px 0;'><strong>Variable:</strong> {var_name}</p>"),
                    pn.pane.HTML(f"<p style='margin: 5px 0;'><strong>Interpretación:</strong> Valor nítido</p>"),
                ),
                title=f"🎯 {var_name}",
                collapsible=False,
                width=200,
                margin=(5, 5, 5, 5)
            )
            
            self.results_cards.append(card)
    
    def _update_explanation(self):
        """Actualizar explicación."""
        if not self.explanation:
            self.explanation_display.object = "<p>No hay explicación disponible</p>"
            return
        
        # Crear explicación HTML
        html = "<div style='background: #f8f9fa; padding: 15px; border-radius: 8px;'>"
        html += "<h4>📋 Explicación del Proceso</h4>"
        
        if hasattr(self.explanation, 'get_step_by_step_explanation'):
            steps = self.explanation.get_step_by_step_explanation()
            for step in steps:
                html += f"<div style='margin: 10px 0; padding: 10px; background: white; border-radius: 5px;'>"
                html += f"<h5>Paso {step['step']}: {step['title']}</h5>"
                html += f"<p>{step['description']}</p>"
                html += "</div>"
        else:
            html += "<p>Explicación detallada no disponible</p>"
        
        html += "</div>"
        self.explanation_display.object = html
    
    def show_error(self, error_message):
        """Mostrar error."""
        self.error_display.object = f"""
        <div style='background: #f8d7da; color: #721c24; padding: 15px; border-radius: 8px; border: 1px solid #f5c6cb;'>
            <h4>❌ Error</h4>
            <p>{error_message}</p>
        </div>
        """
        self.results_cards = []
    
    def get_layout(self) -> pn.layout.Panel:
        """Obtener layout del dashboard."""
        if self.results_cards:
            return pn.Column(
                pn.Row(*self.results_cards, sizing_mode='stretch_width'),
                self.explanation_display,
                sizing_mode='stretch_width'
            )
        else:
            return pn.Column(
                pn.pane.HTML("<p>No hay resultados para mostrar</p>"),
                self.error_display,
                sizing_mode='stretch_width'
            )
