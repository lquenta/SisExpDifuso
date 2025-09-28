#!/usr/bin/env python3
"""
Sistema Experto Difuso Completo - Todo en un solo archivo
Incluye: Esquemas, Motor de Inferencia, UI y Aplicación Principal
"""

import panel as pn
import param
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Literal, Tuple
from pydantic import BaseModel, Field, validator
from abc import ABC, abstractmethod

# =============================================================================
# ESQUEMAS Y MODELOS DE DATOS
# =============================================================================

class TriangularMF(BaseModel):
    """Función de pertenencia triangular."""
    type: Literal["tri"] = "tri"
    a: float = Field(..., description="Vértice izquierdo")
    b: float = Field(..., description="Vértice pico")
    c: float = Field(..., description="Vértice derecho")
    
    @validator('a', 'b', 'c')
    def validate_vertices(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError("Los vértices deben ser numéricos")
        return float(v)

class Term(BaseModel):
    """Término difuso con etiqueta y función de pertenencia."""
    label: str = Field(..., description="Etiqueta lingüística")
    mf: TriangularMF = Field(..., description="Parámetros de función de pertenencia")

class Variable(BaseModel):
    """Variable difusa (entrada o salida)."""
    name: str = Field(..., description="Nombre de la variable")
    kind: Literal["input", "output"] = Field(..., description="Tipo de variable")
    universe: List[float] = Field(..., description="Universo de discurso [min, max]")
    terms: List[Term] = Field(..., description="Términos difusos")
    defuzz: Optional[Literal["centroid", "bisector", "mean_of_maxima"]] = Field(
        default="centroid", description="Método de defuzzificación"
    )

class LogicConfig(BaseModel):
    """Configuración de operadores lógicos."""
    and_op: Literal["min", "prod"] = Field(default="min", description="Operador AND")
    or_op: Literal["max", "prob_or"] = Field(default="max", description="Operador OR")
    implication: Literal["min"] = Field(default="min", description="Método de implicación")
    aggregation: Literal["max"] = Field(default="max", description="Método de agregación")
    defuzz_default: Literal["centroid", "bisector", "mean_of_maxima"] = Field(
        default="centroid", description="Defuzzificación por defecto"
    )

class RuleConclusion(BaseModel):
    """Conclusión de una regla."""
    variable: str = Field(..., description="Variable de salida")
    term: str = Field(..., description="Término de salida")

class Rule(BaseModel):
    """Regla difusa."""
    id: str = Field(..., description="ID único de la regla")
    if_condition: str = Field(..., alias="if", description="Condición antecedente")
    then_conclusions: List[RuleConclusion] = Field(..., alias="then", description="Conclusiones")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Peso de la regla")
    note: Optional[str] = Field(default=None, description="Nota o descripción")
    
    class Config:
        populate_by_name = True

class FuzzySystemConfig(BaseModel):
    """Configuración completa del sistema difuso."""
    schema_version: str = Field(default="1.0", description="Versión del esquema")
    project: str = Field(..., description="Nombre del proyecto")
    description: str = Field(default="", description="Descripción del proyecto")
    author: str = Field(default="", description="Autor del proyecto")
    logic: LogicConfig = Field(default_factory=LogicConfig, description="Operadores lógicos")
    variables: List[Variable] = Field(..., description="Variables del sistema")
    rules: List[Rule] = Field(..., description="Reglas difusas")
    
    def get_variable(self, name: str) -> Optional[Variable]:
        """Obtener variable por nombre."""
        for var in self.variables:
            if var.name == name:
                return var
        return None
    
    def get_input_variables(self) -> List[Variable]:
        """Obtener todas las variables de entrada."""
        return [var for var in self.variables if var.kind == "input"]
    
    def get_output_variables(self) -> List[Variable]:
        """Obtener todas las variables de salida."""
        return [var for var in self.variables if var.kind == "output"]

# =============================================================================
# FUNCIONES DE PERTENENCIA
# =============================================================================

class MembershipFunctionBase(ABC):
    """Clase base para funciones de pertenencia."""
    
    @abstractmethod
    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluar función de pertenencia en punto(s) x."""
        pass

class TriangularMembershipFunction(MembershipFunctionBase):
    """Función de pertenencia triangular."""
    
    def __init__(self, a: float, b: float, c: float):
        if not (a <= b <= c):
            raise ValueError("Los vértices triangulares deben satisfacer a <= b <= c")
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
    
    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        x = np.asarray(x)
        
        if self.a == self.b == self.c:
            return np.where(x == self.a, 1.0, 0.0)
        
        # Lado izquierdo: (x-a)/(b-a) para a <= x <= b
        left = np.where(
            (self.a < self.b) & (x >= self.a) & (x <= self.b),
            (x - self.a) / (self.b - self.a),
            0.0
        )
        
        # Lado derecho: (c-x)/(c-b) para b <= x <= c
        right = np.where(
            (self.b < self.c) & (x >= self.b) & (x <= self.c),
            (self.c - x) / (self.c - self.b),
            0.0
        )
        
        # Pico en b
        peak = np.where(x == self.b, 1.0, 0.0)
        
        result = np.maximum(np.maximum(left, right), peak)
        result = np.maximum(result, 0.0)
        
        return result if isinstance(x, np.ndarray) else float(result)

def create_membership_function(mf_config: TriangularMF) -> MembershipFunctionBase:
    """Crear función de pertenencia desde configuración."""
    return TriangularMembershipFunction(mf_config.a, mf_config.b, mf_config.c)

# =============================================================================
# MOTOR DE INFERENCIA DIFUSA
# =============================================================================

class FuzzyInferenceResult:
    """Resultado del proceso de inferencia difusa."""
    
    def __init__(self, outputs: Dict[str, float], 
                 membership_values: Dict[Tuple[str, str], float],
                 fired_rules: List[Dict[str, Any]],
                 aggregated_functions: Dict[str, Tuple[np.ndarray, np.ndarray]],
                 explanation: Dict[str, Any]):
        self.outputs = outputs
        self.membership_values = membership_values
        self.fired_rules = fired_rules
        self.aggregated_functions = aggregated_functions
        self.explanation = explanation

class FuzzyInferenceEngine:
    """Motor de inferencia difusa Mamdani."""
    
    def __init__(self, config: FuzzySystemConfig, resolution: int = 1001):
        self.config = config
        self.resolution = resolution
        
        # Pre-compilar funciones de pertenencia
        self.membership_functions: Dict[Tuple[str, str], MembershipFunctionBase] = {}
        self._compile_membership_functions()
        
        # Crear defuzzificadores
        self.defuzzifiers: Dict[str, str] = {}
        self._setup_defuzzifiers()
    
    def _compile_membership_functions(self):
        """Pre-compilar todas las funciones de pertenencia."""
        for var in self.config.variables:
            for term in var.terms:
                key = (var.name, term.label)
                self.membership_functions[key] = create_membership_function(term.mf)
    
    def _setup_defuzzifiers(self):
        """Configurar defuzzificadores para cada variable de salida."""
        for var in self.config.get_output_variables():
            self.defuzzifiers[var.name] = var.defuzz or self.config.logic.defuzz_default
    
    def fuzzify(self, input_values: Dict[str, float]) -> Dict[Tuple[str, str], float]:
        """Fuzzificar valores de entrada a grados de pertenencia."""
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
        """Evaluar todas las reglas y retornar reglas activadas."""
        fired_rules = []
        
        for rule in self.config.rules:
            # Evaluar condición (simplificado para este ejemplo)
            activation_degree = self._evaluate_condition(rule.if_condition, membership_values)
            activation_degree *= rule.weight
            
            if activation_degree > 0:
                fired_rules.append({
                    "id": rule.id,
                    "condition": rule.if_condition,
                    "conclusion": {
                        "variable": rule.then_conclusions[0].variable,
                        "term": rule.then_conclusions[0].term
                    },
                    "activation_degree": activation_degree,
                    "weight": rule.weight,
                    "note": rule.note
                })
        
        return fired_rules
    
    def _evaluate_condition(self, condition: str, membership_values: Dict[Tuple[str, str], float]) -> float:
        """Evaluar condición de regla con soporte completo para AND, OR y NOT."""
        condition = condition.strip()
        
        # Manejar NOT
        if condition.startswith("NOT "):
            not_condition = condition[4:].strip()
            return 1.0 - self._evaluate_condition(not_condition, membership_values)
        
        # Manejar paréntesis para agrupar operaciones
        if "(" in condition and ")" in condition:
            return self._evaluate_grouped_condition(condition, membership_values)
        
        # Manejar AND
        if " AND " in condition:
            parts = condition.split(" AND ")
            if self.config.logic.and_op == "min":
                result = 1.0
                for part in parts:
                    part = part.strip()
                    result = min(result, self._evaluate_condition(part, membership_values))
                return result
            elif self.config.logic.and_op == "prod":
                result = 1.0
                for part in parts:
                    part = part.strip()
                    result *= self._evaluate_condition(part, membership_values)
                return result
        
        # Manejar OR
        if " OR " in condition:
            parts = condition.split(" OR ")
            if self.config.logic.or_op == "max":
                result = 0.0
                for part in parts:
                    part = part.strip()
                    result = max(result, self._evaluate_condition(part, membership_values))
                return result
            elif self.config.logic.or_op == "prob_or":
                result = 0.0
                for part in parts:
                    part = part.strip()
                    part_value = self._evaluate_condition(part, membership_values)
                    result = result + part_value - (result * part_value)
                return result
        
        # Condición simple: "Variable is Term"
        if " is " in condition:
            var_name, term_name = condition.split(" is ")
            var_name = var_name.strip()
            term_name = term_name.strip()
            key = (var_name, term_name)
            return membership_values.get(key, 0.0)
        
        return 0.0
    
    def _evaluate_grouped_condition(self, condition: str, membership_values: Dict[Tuple[str, str], float]) -> float:
        """Evaluar condiciones con paréntesis para agrupar operaciones."""
        import re
        
        # Encontrar el primer paréntesis más interno
        pattern = r'\(([^()]+)\)'
        match = re.search(pattern, condition)
        
        if match:
            inner_condition = match.group(1)
            inner_result = self._evaluate_condition(inner_condition, membership_values)
            
            # Reemplazar la expresión entre paréntesis con el resultado
            new_condition = condition.replace(f"({inner_condition})", str(inner_result))
            
            # Continuar evaluando recursivamente
            return self._evaluate_condition(new_condition, membership_values)
        
        # Si no hay paréntesis, evaluar normalmente
        return self._evaluate_condition(condition, membership_values)
    
    def aggregate_outputs(self, fired_rules: List[Dict[str, Any]]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Agregar funciones de pertenencia implícitas."""
        output_vars = self.config.get_output_variables()
        aggregated_functions = {}
        
        for var in output_vars:
            x = np.linspace(var.universe[0], var.universe[1], self.resolution)
            aggregated_membership = np.zeros_like(x)
            
            for rule in fired_rules:
                conclusion = rule["conclusion"]
                if conclusion["variable"] == var.name:
                    term_key = (var.name, conclusion["term"])
                    mf = self.membership_functions[term_key]
                    
                    # Aplicar implicación
                    consequent_membership = mf(x)
                    implied_membership = np.minimum(rule["activation_degree"], consequent_membership)
                    
                    # Agregar usando máximo
                    aggregated_membership = np.maximum(aggregated_membership, implied_membership)
            
            aggregated_functions[var.name] = (x, aggregated_membership)
        
        return aggregated_functions
    
    def defuzzify_outputs(self, aggregated_functions: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
        """Defuzzificar funciones agregadas a valores nítidos."""
        outputs = {}
        
        for var_name, (x, membership) in aggregated_functions.items():
            # Método centroid (centroide)
            if np.sum(membership) > 0:
                centroid = np.sum(x * membership) / np.sum(membership)
            else:
                centroid = (x[0] + x[-1]) / 2
            
            outputs[var_name] = centroid
        
        return outputs
    
    def infer(self, input_values: Dict[str, float]) -> FuzzyInferenceResult:
        """Realizar proceso completo de inferencia difusa."""
        # Validar entradas
        self._validate_inputs(input_values)
        
        # Paso 1: Fuzzificación
        membership_values = self.fuzzify(input_values)
        
        # Paso 2: Evaluación de reglas
        fired_rules = self.evaluate_rules(membership_values)
        
        # Paso 3: Agregación
        aggregated_functions = self.aggregate_outputs(fired_rules)
        
        # Paso 4: Defuzzificación
        outputs = self.defuzzify_outputs(aggregated_functions)
        
        # Paso 5: Crear explicación
        explanation = {
            "input_values": input_values,
            "fuzzification": {},
            "rule_evaluation": {},
            "outputs": outputs,
            "summary": {
                "total_rules": len(self.config.rules),
                "fired_rules": len(fired_rules),
                "max_activation": max([r["activation_degree"] for r in fired_rules]) if fired_rules else 0
            }
        }
        
        return FuzzyInferenceResult(
            outputs=outputs,
            membership_values=membership_values,
            fired_rules=fired_rules,
            aggregated_functions=aggregated_functions,
            explanation=explanation
        )
    
    def _validate_inputs(self, input_values: Dict[str, float]):
        """Validar valores de entrada."""
        input_vars = self.config.get_input_variables()
        input_var_names = {var.name for var in input_vars}
        
        for var_name in input_values:
            if var_name not in input_var_names:
                raise ValueError(f"Variable de entrada desconocida: {var_name}")
        
        for var in input_vars:
            if var.name not in input_values:
                raise ValueError(f"Variable de entrada requerida faltante: {var.name}")

# =============================================================================
# CONFIGURACIÓN DE EJEMPLO
# =============================================================================

def create_example_config() -> FuzzySystemConfig:
    """Crear configuración de ejemplo."""
    return FuzzySystemConfig(
        schema_version="1.0",
        project="Politica_Arancelaria",
        description="Sistema experto para determinar política arancelaria",
        author="Sistema Experto Difuso",
        logic=LogicConfig(),
        variables=[
            Variable(
                name="Déficit",
                kind="input",
                universe=[0.0, 10.0],
                terms=[
                    Term(label="Bajo", mf=TriangularMF(a=0.0, b=0.0, c=4.0)),
                    Term(label="Medio", mf=TriangularMF(a=2.5, b=5.0, c=7.5)),
                    Term(label="Alto", mf=TriangularMF(a=6.0, b=10.0, c=10.0))
                ]
            ),
            Variable(
                name="Presión",
                kind="input",
                universe=[0.0, 100.0],
                terms=[
                    Term(label="Baja", mf=TriangularMF(a=0.0, b=0.0, c=40.0)),
                    Term(label="Media", mf=TriangularMF(a=20.0, b=50.0, c=80.0)),
                    Term(label="Alta", mf=TriangularMF(a=60.0, b=100.0, c=100.0))
                ]
            ),
            Variable(
                name="Tarifa",
                kind="output",
                universe=[0.0, 100.0],
                defuzz="centroid",
                terms=[
                    Term(label="Baja", mf=TriangularMF(a=0.0, b=0.0, c=30.0)),
                    Term(label="Moderada", mf=TriangularMF(a=20.0, b=50.0, c=80.0)),
                    Term(label="Alta", mf=TriangularMF(a=70.0, b=100.0, c=100.0))
                ]
            )
        ],
        rules=[
            Rule(
                id="R1",
                if_condition="Déficit is Bajo AND Presión is Baja",
                then_conclusions=[RuleConclusion(variable="Tarifa", term="Baja")],
                weight=1.0,
                note="Situación estable, tarifa baja"
            ),
            Rule(
                id="R2",
                if_condition="Déficit is Medio AND Presión is Alta",
                then_conclusions=[RuleConclusion(variable="Tarifa", term="Alta")],
                weight=1.0,
                note="Déficit medio con alta presión requiere tarifa alta"
            ),
            Rule(
                id="R3",
                if_condition="Déficit is Alto AND Presión is Alta",
                then_conclusions=[RuleConclusion(variable="Tarifa", term="Alta")],
                weight=1.0,
                note="Situación crítica, tarifa alta necesaria"
            ),
            Rule(
                id="R4",
                if_condition="Déficit is Alto OR Presión is Alta",
                then_conclusions=[RuleConclusion(variable="Tarifa", term="Moderada")],
                weight=0.8,
                note="Cualquier condición extrema requiere tarifa moderada"
            ),
            Rule(
                id="R5",
                if_condition="NOT Déficit is Bajo",
                then_conclusions=[RuleConclusion(variable="Tarifa", term="Moderada")],
                weight=0.7,
                note="Si el déficit no es bajo, aplicar tarifa moderada"
            ),
            Rule(
                id="R6",
                if_condition="(Déficit is Medio OR Déficit is Alto) AND NOT Presión is Baja",
                then_conclusions=[RuleConclusion(variable="Tarifa", term="Alta")],
                weight=0.9,
                note="Déficit medio/alto con presión no baja = tarifa alta"
            )
        ]
    )

# =============================================================================
# INTERFAZ DE USUARIO
# =============================================================================

class FuzzySystemApp(param.Parameterized):
    """Aplicación principal del sistema experto difuso."""
    
    # Parámetros principales
    current_config = param.ClassSelector(class_=FuzzySystemConfig)
    inference_result = param.Parameter(default=None)
    input_values = param.Dict(default={})
    
    def __init__(self, **params):
        super().__init__(**params)
        
        # Crear configuración inicial
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
            name="🚀 Ejecutar Inferencia",
            button_type="primary",
            width=250
        )
        self.inference_button.on_click(self._run_inference)
        
        # Editor de variables
        self._create_variable_editor()
        
        # Editor de reglas
        self._create_rule_editor()
        
        # Área de resultados
        self.results_panel = pn.pane.HTML("", width=400, height=200)
        
        # Área de reglas
        self.rules_panel = pn.pane.HTML("", width=600, height=400)
        
        # Área de gráficas dinámicas
        self.plots_panel = pn.pane.HTML("", width=600, height=500)
        
        # Actualizar display inicial
        self._update_rules_display()
        self._update_plots_display()
    
    
    
    def _recreate_complete_layout(self):
        """Recrear completamente el layout con el nuevo tema."""
        try:
            # Actualizar todos los displays
            self._update_all_displays()
            
            # Recrear el layout principal
            new_layout = self._create_main_layout()
            
            # Reemplazar el layout actual
            if hasattr(self, '_main_layout'):
                # Guardar el nuevo layout
                self._main_layout = new_layout
                
                # Forzar actualización del servidor
                import panel as pn
                if hasattr(pn.state, 'curdoc') and pn.state.curdoc():
                    # Limpiar y reemplazar el contenido
                    doc = pn.state.curdoc()
                    doc.clear()
                    doc.add_root(new_layout)
            
            print("🔄 Layout completamente recreado con el nuevo tema")
        except Exception as e:
            print(f"❌ Error recreando layout completo: {e}")
    
    def _force_update_all_displays(self):
        """Forzar actualización visual de todos los displays."""
        try:
            # Actualizar displays existentes
            if hasattr(self, 'rules_panel'):
                self._update_rules_display()
                
            if hasattr(self, 'plots_panel'):
                self._update_plots_display()
                
            if hasattr(self, 'results_panel') and hasattr(self, '_last_result'):
                self._update_results_display(self._last_result)
            
            # Actualizar también la guía de ayuda si existe
            if hasattr(self, '_help_guide'):
                help_html = self._update_help_guide()
                self._help_guide.object = help_html
            
            # Forzar actualización del servidor
            import panel as pn
            if hasattr(pn.state, 'curdoc') and pn.state.curdoc():
                pn.state.curdoc().add_next_tick_callback(self._refresh_layout)
            
            print("✅ Todos los displays actualizados visualmente con el nuevo tema")
        except Exception as e:
            print(f"❌ Error forzando actualización de displays: {e}")
    
    def _refresh_layout(self):
        """Refrescar el layout completo."""
        try:
            # Recrear el layout principal con el nuevo tema
            new_layout = self._create_main_layout()
            
            # Reemplazar el layout actual
            if hasattr(self, '_main_layout'):
                # Limpiar el layout actual
                self._main_layout.clear()
                # Agregar el nuevo contenido
                for obj in new_layout:
                    self._main_layout.append(obj)
            
            print("🔄 Layout refrescado con el nuevo tema")
        except Exception as e:
            print(f"❌ Error refrescando layout: {e}")
    
    def _update_help_guide(self):
        """Actualizar la guía de ayuda."""
        html = """
        <div style="background-color: #f8f9fa; border: 1px solid #dee2e6; padding: 15px; border-radius: 8px; font-size: 0.9em;">
            <h4 style="margin: 0 0 15px 0;">📖 Guía de Parámetros</h4>
            
            <div style="margin-bottom: 15px;">
                <h5 style="margin: 0 0 8px 0; color: #6c757d;">🔧 Variables:</h5>
                <ul style="margin: 0; padding-left: 20px;">
                    <li><strong>Nombre:</strong> Identificador único de la variable</li>
                    <li><strong>Tipo:</strong> input (entrada) o output (salida)</li>
                    <li><strong>Universo:</strong> Rango mínimo y máximo de valores</li>
                    <li><strong>Términos:</strong> Etiquetas lingüísticas (Bajo, Medio, Alto)</li>
                </ul>
            </div>
            
            <div style="margin-bottom: 15px;">
                <h5 style="margin: 0 0 8px 0; color: #6c757d;">📐 Funciones de Pertenencia:</h5>
                <ul style="margin: 0; padding-left: 20px;">
                    <li><strong>A, B, C:</strong> Parámetros triangulares</li>
                    <li><strong>B:</strong> Punto de máxima pertenencia (1.0)</li>
                    <li><strong>A, C:</strong> Puntos de pertenencia cero (0.0)</li>
                    <li><strong>Orden:</strong> A ≤ B ≤ C</li>
                </ul>
            </div>
            
            <div style="margin-bottom: 15px;">
                <h5 style="margin: 0 0 8px 0; color: #6c757d;">📝 Reglas:</h5>
                <ul style="margin: 0; padding-left: 20px;">
                    <li><strong>ID:</strong> Identificador único (ej: R1, R2)</li>
                    <li><strong>Condición:</strong> Lógica difusa (ej: "X is Alto AND Y is Bajo")</li>
                    <li><strong>Conclusión:</strong> Variable y término de salida</li>
                    <li><strong>Operadores:</strong> AND, OR, NOT</li>
                </ul>
            </div>
            
            <div style="margin-bottom: 15px;">
                <h5 style="margin: 0 0 8px 0; color: #6c757d;">🎯 Interpretación:</h5>
                <ul style="margin: 0; padding-left: 20px;">
                    <li><strong>Valor Nítido:</strong> Resultado numérico final</li>
                    <li><strong>Mejor Término:</strong> Etiqueta más apropiada</li>
                    <li><strong>Confianza:</strong> Grado de certeza (0-1)</li>
                    <li><strong>Reglas Activadas:</strong> Reglas que se ejecutaron</li>
                </ul>
            </div>
            
            <div style="background-color: #e9ecef; padding: 10px; border-radius: 5px; margin-top: 15px;">
                <strong>💡 Tip:</strong> 
                <span style="color: #6c757d;">Las funciones triangulares deben solaparse para transiciones suaves entre términos.</span>
            </div>
        </div>
        """
        
        return html
    
    def _recreate_layout(self):
        """Recrear el layout completo con el nuevo tema."""
        try:
            # Actualizar todos los displays
            self._update_all_displays()
            print("✅ Layout actualizado con el nuevo tema")
        except Exception as e:
            print(f"❌ Error recreando layout: {e}")
    
    def _update_all_displays(self):
        """Actualizar todos los displays con el nuevo tema."""
        try:
            # Actualizar displays existentes
            if hasattr(self, 'rules_panel'):
                self._update_rules_display()
            if hasattr(self, 'plots_panel'):
                self._update_plots_display()
            if hasattr(self, 'results_panel') and hasattr(self, '_last_result'):
                self._update_results_display(self._last_result)
            
            print("✅ Todos los displays actualizados con el nuevo tema")
        except Exception as e:
            print(f"❌ Error actualizando displays: {e}")
    
    def _create_variable_editor(self):
        """Crear editor de variables."""
        # Selector de variable para editar
        self.variable_selector = pn.widgets.Select(
            name="Seleccionar Variable",
            options=[var.name for var in self.current_config.variables],
            value=self.current_config.variables[0].name if self.current_config.variables else None,
            width=200
        )
        self.variable_selector.param.watch(self._on_variable_selection_change, 'value')
        
        # Campos del editor de variables
        self.var_name_input = pn.widgets.TextInput(
            name="Nombre de Variable",
            value="",
            width=200
        )
        
        self.var_type_select = pn.widgets.Select(
            name="Tipo",
            options=["input", "output"],
            value="input",
            width=150
        )
        
        self.var_universe_min = pn.widgets.NumberInput(
            name="Mínimo",
            value=0.0,
            width=100
        )
        
        self.var_universe_max = pn.widgets.NumberInput(
            name="Máximo",
            value=100.0,
            width=100
        )
        
        # Editor de términos
        self.term_low_input = pn.widgets.TextInput(
            name="Término Bajo",
            value="Bajo",
            width=120
        )
        
        self.term_medium_input = pn.widgets.TextInput(
            name="Término Medio",
            value="Medio",
            width=120
        )
        
        self.term_high_input = pn.widgets.TextInput(
            name="Término Alto",
            value="Alto",
            width=120
        )
        
        # Parámetros de funciones de pertenencia
        self.mf_low_a = pn.widgets.NumberInput(name="Bajo A", value=0.0, width=80)
        self.mf_low_b = pn.widgets.NumberInput(name="Bajo B", value=0.0, width=80)
        self.mf_low_c = pn.widgets.NumberInput(name="Bajo C", value=30.0, width=80)
        
        self.mf_medium_a = pn.widgets.NumberInput(name="Medio A", value=20.0, width=80)
        self.mf_medium_b = pn.widgets.NumberInput(name="Medio B", value=50.0, width=80)
        self.mf_medium_c = pn.widgets.NumberInput(name="Medio C", value=80.0, width=80)
        
        self.mf_high_a = pn.widgets.NumberInput(name="Alto A", value=70.0, width=80)
        self.mf_high_b = pn.widgets.NumberInput(name="Alto B", value=100.0, width=80)
        self.mf_high_c = pn.widgets.NumberInput(name="Alto C", value=100.0, width=80)
        
        # Botones de acción para variables
        self.add_variable_button = pn.widgets.Button(
            name="+ Agregar Variable",
            button_type="success",
            width=150
        )
        self.add_variable_button.on_click(self._add_variable)
        
        self.edit_variable_button = pn.widgets.Button(
            name="✏️ Editar Variable",
            button_type="primary",
            width=150
        )
        self.edit_variable_button.on_click(self._edit_variable)
        
        self.delete_variable_button = pn.widgets.Button(
            name="🗑️ Eliminar Variable",
            button_type="danger",
            width=150
        )
        self.delete_variable_button.on_click(self._delete_variable)
        
        # Actualizar selector inicial
        self._update_variable_selector()
    
    def _create_rule_editor(self):
        """Crear editor de reglas."""
        # Campos del editor de reglas
        self.rule_id_input = pn.widgets.TextInput(
            name="ID de Regla", 
            value="R4", 
            width=100
        )
        
        self.rule_condition_input = pn.widgets.TextAreaInput(
            name="Condición (SI)", 
            value="Déficit is Medio AND Presión is Alta",
            height=100,
            width=400
        )
        
        # Obtener variables de salida disponibles
        output_vars = [var.name for var in self.current_config.get_output_variables()]
        self.rule_conclusion_var = pn.widgets.Select(
            name="Variable de Conclusión",
            options=output_vars,
            value=output_vars[0] if output_vars else "",
            width=150
        )
        
        # Obtener términos disponibles para la variable seleccionada
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
    
    def _on_slider_change(self, event, var_name):
        """Manejar cambios en sliders."""
        self.input_values[var_name] = event.new
        slider = self.input_sliders[var_name]
        slider.name = f"{var_name}: {event.new:.1f}"
        
        # Actualizar gráficas dinámicamente
        self._update_plots_display()
    
    def _run_inference(self, event):
        """Ejecutar inferencia difusa."""
        try:
            # Crear motor de inferencia
            engine = FuzzyInferenceEngine(self.current_config)
            
            # Ejecutar inferencia
            result = engine.infer(self.input_values)
            
            # Actualizar resultados
            self.inference_result = result
            self._update_results_display(result)
            self._update_rules_display()
            self._update_plots_display()  # Agregar actualización de gráficas
            
        except Exception as e:
            self.results_panel.object = f"""
            <div style="background-color: #ffebee; border: 1px solid #f44336; padding: 10px; border-radius: 5px;">
                <h4 style="color: #d32f2f; margin: 0;">❌ Error en Motor Difuso</h4>
                <p style="margin: 5px 0; color: #d32f2f;">{str(e)}</p>
            </div>
            """
    
    def _update_results_display(self, result: FuzzyInferenceResult):
        """Actualizar display de resultados con interpretación lingüística."""
        if result is None:
            self.results_panel.object = ""
            return
            
        self._last_result = result
        
        html = """
        <div style="background-color: #e8f5e8; border: 1px solid #4caf50; padding: 15px; border-radius: 5px;">
            <h3 style="color: #2e7d32; margin: 0 0 10px 0;">🎯 Resultados de Inferencia</h3>
        """
        
        for var_name, value in result.outputs.items():
            # Obtener interpretación lingüística
            linguistic = self._get_linguistic_interpretation(var_name, value)
            
            html += f"""
            <div style="margin: 10px 0; padding: 15px; background-color: white; border-radius: 5px; border-left: 4px solid #4caf50;">
                <h4 style="margin: 0 0 8px 0; color: #2e7d32;">{var_name}</h4>
                <div style="font-size: 1.2em; font-weight: bold; color: #1976D2; margin-bottom: 8px;">
                    Valor Nítido: {value:.3f}
                </div>
                <div style="margin-bottom: 5px;">
                    <strong>Interpretación Lingüística:</strong> 
                    <span style="color: #FF6B35; font-weight: bold;">{linguistic['best_term']}</span>
                    <span style="color: #666; font-size: 0.9em;">(Confianza: {linguistic['confidence']:.3f})</span>
                </div>
                <div style="font-size: 0.9em; color: #666;">
                    <strong>Todos los términos:</strong>
                    <ul style="margin: 5px 0; padding-left: 20px;">
            """
            
            for term, confidence in linguistic['all_terms'].items():
                html += f"<li>{term}: {confidence:.3f}</li>"
            
            html += """
                    </ul>
                </div>
            </div>
            """
        
        html += f"""
            <div style="margin-top: 15px; font-size: 0.9em; color: #666; background-color: #f0f8f0; padding: 10px; border-radius: 3px;">
                <strong>Estadísticas:</strong><br>
                • Reglas activadas: {len(result.fired_rules)}<br>
                • Confianza máxima: {max([r["activation_degree"] for r in result.fired_rules]) if result.fired_rules else 0:.3f}<br>
                • Total de reglas: {len(self.current_config.rules)}
            </div>
        </div>
        """
        
        self.results_panel.object = html
    
    def _get_linguistic_interpretation(self, var_name: str, crisp_value: float) -> Dict[str, Any]:
        """Obtener interpretación lingüística del resultado."""
        var = self.current_config.get_variable(var_name)
        if var is None:
            return {"best_term": "Desconocido", "confidence": 0.0, "all_terms": {}}
        
        # Crear motor temporal para evaluar funciones de pertenencia
        engine = FuzzyInferenceEngine(self.current_config)
        
        best_term = None
        best_confidence = 0.0
        all_terms = {}
        
        for term in var.terms:
            key = (var_name, term.label)
            mf = engine.membership_functions[key]
            confidence = float(mf(crisp_value))
            all_terms[term.label] = confidence
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_term = term.label
        
        return {
            "best_term": best_term or "Desconocido",
            "confidence": best_confidence,
            "all_terms": all_terms
        }
    
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
            status_color = "#28a745" if is_active else "#6c757d"
            status_text = "✅ Activada" if is_active else "⚪ Inactiva"
            activation_text = f"{activation_degree:.3f}" if is_active else "0.000"
            
            html += f"""
                    <tr style="background-color: {row_color};">
                        <td style="padding: 12px; border-bottom: 1px solid #dee2e6; font-weight: bold; color: #007bff;">{rule.id}</td>
                        <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">{rule.if_condition}</td>
                        <td style="padding: 12px; border-bottom: 1px solid #dee2e6; font-weight: bold;">→ {rule.then_conclusions[0].variable} es {rule.then_conclusions[0].term}</td>
                        <td style="padding: 12px; border-bottom: 1px solid #dee2e6; text-align: center; color: {status_color};">{status_text}</td>
                        <td style="padding: 12px; border-bottom: 1px solid #dee2e6; text-align: center; font-weight: bold; color: {status_color};">{activation_text}</td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        """
        self.rules_panel.object = html
    
    def _update_plots_display(self):
        """Actualizar display de gráficas dinámicas."""
        try:
            # Crear motor temporal para evaluar funciones de pertenencia
            engine = FuzzyInferenceEngine(self.current_config)
            
            html = """
            <div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px;">
                <h3 style="margin: 0 0 15px 0;">📊 Visualización Dinámica</h3>
            """
            
            # Generar gráficas para cada variable
            for var in self.current_config.variables:
                min_val, max_val = var.universe
                
                # Determinar el valor actual según el tipo de variable
                if var.kind == "input":
                    current_val = self.input_values.get(var.name, (min_val + max_val) / 2)
                else:  # variable de salida
                    if self.inference_result and var.name in self.inference_result.outputs:
                        current_val = self.inference_result.outputs[var.name]
                    else:
                        current_val = (min_val + max_val) / 2
                
                html += f"""
                <div style="margin-bottom: 25px; padding: 15px; background-color: white; border-radius: 5px; border: 1px solid #dee2e6;">
                    <h4 style="margin: 0 0 10px 0;">{var.name} (Valor actual: {current_val:.2f})</h4>
                    <div style="position: relative; height: 200px; border: 1px solid #dee2e6; background-color: #f8f9fa;">
                """
                
                # Generar SVG para las funciones de pertenencia
                svg_width = 500
                svg_height = 180
                margin = 20
                
                html += f"""
                <svg width="{svg_width}" height="{svg_height}" style="position: absolute; top: 10px; left: 10px;">
                """
                
                # Dibujar ejes
                html += f"""
                    <line x1="{margin}" y1="{svg_height - margin}" x2="{svg_width - margin}" y2="{svg_height - margin}" stroke="#6c757d" stroke-width="2"/>
                    <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{svg_height - margin}" stroke="#6c757d" stroke-width="2"/>
                """
                
                # Etiquetas de ejes
                html += f"""
                    <text x="{svg_width//2}" y="{svg_height - 5}" text-anchor="middle" font-size="12" fill="#6c757d">Valor</text>
                    <text x="10" y="{svg_height//2}" text-anchor="middle" font-size="12" fill="#6c757d" transform="rotate(-90, 10, {svg_height//2})">Pertenencia</text>
                """
                
                # Dibujar funciones de pertenencia para cada término
                colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
                
                for i, term in enumerate(var.terms):
                    color = colors[i % len(colors)]
                    key = (var.name, term.label)
                    mf = engine.membership_functions[key]
                    
                    # Generar puntos para la función triangular
                    points = []
                    for x in range(margin, svg_width - margin, 2):
                        val = min_val + (x - margin) * (max_val - min_val) / (svg_width - 2 * margin)
                        membership = float(mf(val))
                        y = svg_height - margin - membership * (svg_height - 2 * margin)
                        points.append(f"{x},{y}")
                    
                    # Dibujar línea de la función
                    if points:
                        html += f"""
                        <polyline points="{' '.join(points)}" fill="none" stroke="{color}" stroke-width="2"/>
                        """
                    
                    # Etiqueta del término
                    term_x = margin + (term.mf.b - min_val) * (svg_width - 2 * margin) / (max_val - min_val)
                    term_y = svg_height - margin - 0.8 * (svg_height - 2 * margin)
                    html += f"""
                    <text x="{term_x}" y="{term_y}" text-anchor="middle" font-size="10" fill="{color}" font-weight="bold">{term.label}</text>
                    """
                
                # Dibujar línea vertical para el valor actual
                current_x = margin + (current_val - min_val) * (svg_width - 2 * margin) / (max_val - min_val)
                line_color = "#28a745" if var.kind == "output" else "#dc3545"  # Verde para salida, rojo para entrada
                line_label = "Resultado" if var.kind == "output" else "Actual"
                html += f"""
                <line x1="{current_x}" y1="{margin}" x2="{current_x}" y2="{svg_height - margin}" stroke="{line_color}" stroke-width="3" stroke-dasharray="5,5"/>
                <text x="{current_x}" y="{margin - 5}" text-anchor="middle" font-size="10" fill="{line_color}" font-weight="bold">{line_label}</text>
                """
                
                # Mostrar valores de pertenencia actuales
                html += f"""
                <div style="position: absolute; top: 10px; right: 10px; background-color: white; padding: 8px; border-radius: 3px; font-size: 11px; border: 1px solid #dee2e6;">
                """
                
                for term in var.terms:
                    key = (var.name, term.label)
                    mf = engine.membership_functions[key]
                    membership = float(mf(current_val))
                    html += f"""
                    <div style="color: {colors[var.terms.index(term) % len(colors)]}; font-weight: bold;">
                        {term.label}: {membership:.3f}
                    </div>
                    """
                
                html += """
                </div>
                """
                
                html += """
                    </svg>
                    </div>
                </div>
                """
            
            html += """
            </div>
            """
            
            self.plots_panel.object = html
            
        except Exception as e:
            self.plots_panel.object = f"""
            <div style="background-color: #f8d7da; border: 1px solid #f5c6cb; padding: 15px; border-radius: 5px;">
                <h4 style="color: #721c24; margin: 0;">❌ Error en Gráficas</h4>
                <p style="margin: 5px 0; color: #721c24;">{str(e)}</p>
            </div>
            """
    
    def _add_rule(self, event):
        """Agregar nueva regla."""
        try:
            # Crear nueva regla
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
            
            print("✅ Regla agregada exitosamente")
            
        except Exception as e:
            print(f"❌ Error agregando regla: {e}")
    
    def _edit_rule(self, event):
        """Editar regla existente."""
        try:
            if not self.rule_selector.value:
                print("❌ Selecciona una regla para editar")
                return
            
            # Buscar la regla a editar
            rule_to_edit = None
            for rule in self.current_config.rules:
                if rule.id == self.rule_selector.value:
                    rule_to_edit = rule
                    break
            
            if not rule_to_edit:
                print("❌ Regla no encontrada")
                return
            
            # Actualizar la regla
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
            
            print("✅ Regla editada exitosamente")
            
        except Exception as e:
            print(f"❌ Error editando regla: {e}")
    
    def _delete_rule(self, event):
        """Eliminar regla existente."""
        try:
            if not self.rule_selector.value:
                print("❌ Selecciona una regla para eliminar")
                return
            
            # Confirmar eliminación
            if len(self.current_config.rules) <= 1:
                print("❌ No se puede eliminar la última regla del sistema")
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
            
            print("✅ Regla eliminada exitosamente")
            
        except Exception as e:
            print(f"❌ Error eliminando regla: {e}")
    
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
    
    def _add_variable(self, event):
        """Agregar nueva variable."""
        try:
            # Crear nueva variable
            new_variable = Variable(
                name=self.var_name_input.value,
                kind=self.var_type_select.value,
                universe=[self.var_universe_min.value, self.var_universe_max.value],
                terms=[
                    Term(label=self.term_low_input.value, mf=TriangularMF(
                        a=self.mf_low_a.value, b=self.mf_low_b.value, c=self.mf_low_c.value
                    )),
                    Term(label=self.term_medium_input.value, mf=TriangularMF(
                        a=self.mf_medium_a.value, b=self.mf_medium_b.value, c=self.mf_medium_c.value
                    )),
                    Term(label=self.term_high_input.value, mf=TriangularMF(
                        a=self.mf_high_a.value, b=self.mf_high_b.value, c=self.mf_high_c.value
                    ))
                ]
            )
            
            # Agregar a la configuración
            self.current_config.variables.append(new_variable)
            
            # Actualizar selectores
            self._update_variable_selector()
            self._update_rule_selectors()
            self._recreate_input_sliders()
            
            # Actualizar gráficas
            self._update_plots_display()
            
            # Limpiar campos
            self._clear_variable_fields()
            
            print("✅ Variable agregada exitosamente")
            
        except Exception as e:
            print(f"❌ Error agregando variable: {e}")
    
    def _edit_variable(self, event):
        """Editar variable existente."""
        try:
            if not self.variable_selector.value:
                print("❌ Selecciona una variable para editar")
                return
            
            # Buscar la variable a editar
            var_to_edit = None
            for var in self.current_config.variables:
                if var.name == self.variable_selector.value:
                    var_to_edit = var
                    break
            
            if not var_to_edit:
                print("❌ Variable no encontrada")
                return
            
            # Actualizar la variable
            var_to_edit.name = self.var_name_input.value
            var_to_edit.kind = self.var_type_select.value
            var_to_edit.universe = [self.var_universe_min.value, self.var_universe_max.value]
            
            # Actualizar términos
            var_to_edit.terms = [
                Term(label=self.term_low_input.value, mf=TriangularMF(
                    a=self.mf_low_a.value, b=self.mf_low_b.value, c=self.mf_low_c.value
                )),
                Term(label=self.term_medium_input.value, mf=TriangularMF(
                    a=self.mf_medium_a.value, b=self.mf_medium_b.value, c=self.mf_medium_c.value
                )),
                Term(label=self.term_high_input.value, mf=TriangularMF(
                    a=self.mf_high_a.value, b=self.mf_high_b.value, c=self.mf_high_c.value
                ))
            ]
            
            # Actualizar selectores
            self._update_variable_selector()
            self._update_rule_selectors()
            self._recreate_input_sliders()
            
            # Actualizar gráficas
            self._update_plots_display()
            
            print("✅ Variable editada exitosamente")
            
        except Exception as e:
            print(f"❌ Error editando variable: {e}")
    
    def _delete_variable(self, event):
        """Eliminar variable existente."""
        try:
            if not self.variable_selector.value:
                print("❌ Selecciona una variable para eliminar")
                return
            
            # Confirmar eliminación
            if len(self.current_config.variables) <= 1:
                print("❌ No se puede eliminar la última variable del sistema")
                return
            
            # Eliminar la variable
            self.current_config.variables = [
                var for var in self.current_config.variables 
                if var.name != self.variable_selector.value
            ]
            
            # Actualizar selectores
            self._update_variable_selector()
            self._update_rule_selectors()
            self._recreate_input_sliders()
            
            # Actualizar gráficas
            self._update_plots_display()
            
            # Limpiar campos
            self._clear_variable_fields()
            
            print("✅ Variable eliminada exitosamente")
            
        except Exception as e:
            print(f"❌ Error eliminando variable: {e}")
    
    def _update_variable_selector(self):
        """Actualizar opciones del selector de variables."""
        var_names = [var.name for var in self.current_config.variables]
        self.variable_selector.options = var_names
        if var_names:
            self.variable_selector.value = var_names[0]
        else:
            self.variable_selector.value = None
    
    def _update_rule_selectors(self):
        """Actualizar selectores de reglas con nuevas variables."""
        # Actualizar selector de variable de conclusión
        output_vars = [var.name for var in self.current_config.get_output_variables()]
        self.rule_conclusion_var.options = output_vars
        if output_vars:
            self.rule_conclusion_var.value = output_vars[0]
        
        # Actualizar términos disponibles
        if self.rule_conclusion_var.value:
            var = self.current_config.get_variable(self.rule_conclusion_var.value)
            if var:
                term_options = [term.label for term in var.terms]
                self.rule_conclusion_term.options = term_options
                if term_options:
                    self.rule_conclusion_term.value = term_options[0]
    
    def _recreate_input_sliders(self):
        """Recrear sliders de entrada con nuevas variables."""
        # Limpiar sliders existentes
        self.input_sliders = {}
        
        # Crear nuevos sliders
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
    
    def _on_variable_selection_change(self, event):
        """Manejar cambio de selección de variable."""
        if not event.new:
            return
        
        # Buscar la variable seleccionada
        selected_var = None
        for var in self.current_config.variables:
            if var.name == event.new:
                selected_var = var
                break
        
        if selected_var:
            # Cargar datos de la variable en los campos
            self.var_name_input.value = selected_var.name
            self.var_type_select.value = selected_var.kind
            self.var_universe_min.value = selected_var.universe[0]
            self.var_universe_max.value = selected_var.universe[1]
            
            # Cargar términos (asumiendo que siempre hay 3 términos)
            if len(selected_var.terms) >= 3:
                self.term_low_input.value = selected_var.terms[0].label
                self.term_medium_input.value = selected_var.terms[1].label
                self.term_high_input.value = selected_var.terms[2].label
                
                # Cargar parámetros de funciones de pertenencia
                self.mf_low_a.value = selected_var.terms[0].mf.a
                self.mf_low_b.value = selected_var.terms[0].mf.b
                self.mf_low_c.value = selected_var.terms[0].mf.c
                
                self.mf_medium_a.value = selected_var.terms[1].mf.a
                self.mf_medium_b.value = selected_var.terms[1].mf.b
                self.mf_medium_c.value = selected_var.terms[1].mf.c
                
                self.mf_high_a.value = selected_var.terms[2].mf.a
                self.mf_high_b.value = selected_var.terms[2].mf.b
                self.mf_high_c.value = selected_var.terms[2].mf.c
    
    def _clear_variable_fields(self):
        """Limpiar campos del editor de variables."""
        self.var_name_input.value = ""
        self.var_type_select.value = "input"
        self.var_universe_min.value = 0.0
        self.var_universe_max.value = 100.0
        self.term_low_input.value = "Bajo"
        self.term_medium_input.value = "Medio"
        self.term_high_input.value = "Alto"
        self.mf_low_a.value = 0.0
        self.mf_low_b.value = 0.0
        self.mf_low_c.value = 30.0
        self.mf_medium_a.value = 20.0
        self.mf_medium_b.value = 50.0
        self.mf_medium_c.value = 80.0
        self.mf_high_a.value = 70.0
        self.mf_high_b.value = 100.0
        self.mf_high_c.value = 100.0
    
    def _create_main_layout(self) -> pn.layout.Panel:
        """Crear layout principal."""
        
        # Header
        header = pn.pane.HTML("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 20px; text-align: center; border-radius: 10px; margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 2.5em;">🧠 Sistema Experto Difuso</h1>
        </div>
        """, width=800)
        
        # Panel izquierdo - Control de Entrada y Resultados
        left_panel = pn.Column(
            pn.pane.HTML("<h3>🎛️ Control de Entrada</h3>"),
            *self.input_sliders.values(),
            self.inference_button,
            pn.pane.HTML("<br>"),
            self.results_panel,
            width=450,
            sizing_mode='fixed',
            margin=(0, 10, 0, 0)
        )
        
        # Panel derecho - Editor de Variables y Reglas en pestañas
        variable_editor = pn.Column(
            pn.pane.HTML("<h3>🔧 Editor de Variables</h3>"),
            self.variable_selector,
            pn.Row(
                self.var_name_input,
                self.var_type_select
            ),
            pn.Row(
                self.var_universe_min,
                self.var_universe_max
            ),
            pn.pane.HTML("<h4>Términos Lingüísticos</h4>"),
            pn.Row(
                self.term_low_input,
                self.term_medium_input,
                self.term_high_input
            ),
            pn.pane.HTML("<h4>Parámetros de Funciones de Pertenencia</h4>"),
            pn.pane.HTML("<strong>Término Bajo:</strong>"),
            pn.Row(
                self.mf_low_a,
                self.mf_low_b,
                self.mf_low_c
            ),
            pn.pane.HTML("<strong>Término Medio:</strong>"),
            pn.Row(
                self.mf_medium_a,
                self.mf_medium_b,
                self.mf_medium_c
            ),
            pn.pane.HTML("<strong>Término Alto:</strong>"),
            pn.Row(
                self.mf_high_a,
                self.mf_high_b,
                self.mf_high_c
            ),
            pn.Row(
                self.add_variable_button,
                self.edit_variable_button,
                self.delete_variable_button
            ),
            width=600,
            sizing_mode='fixed'
        )
        
        # Panel de reglas
        rules_editor = pn.Column(
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
            pn.pane.HTML("<h3>📋 Reglas del Sistema</h3>"),
            self.rules_panel,
            width=600,
            sizing_mode='fixed'
        )
        
        # Guía de ayuda
        help_guide_html = self._update_help_guide()
        help_guide = pn.pane.HTML(help_guide_html, width=300, sizing_mode='fixed')
        
        # Guardar referencia para actualizaciones de tema
        self._help_guide = help_guide
        
        # Panel de gráficas
        plots_editor = pn.Column(
            self.plots_panel,
            width=650,
            sizing_mode='fixed'
        )
        
        # Crear pestañas para el panel derecho
        right_panel = pn.Tabs(
            ("🔧 Variables", variable_editor),
            ("📝 Reglas", rules_editor),
            ("📊 Gráficas", plots_editor),
            width=650,
            sizing_mode='fixed',
            margin=(0, 0, 0, 10)
        )
        
        # Panel de ayuda
        help_panel = pn.Column(
            help_guide,
            width=320,
            sizing_mode='fixed',
            margin=(0, 0, 0, 10)
        )
        
        # Información del sistema
        info_panel = pn.pane.HTML(f"""
        <div style="background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin-top: 20px;">
            <h4 style="margin: 0 0 10px 0;">ℹ️ Información del Sistema</h4>
            <p><strong>Proyecto:</strong> {self.current_config.project}</p>
            <p><strong>Variables:</strong> {len(self.current_config.variables)}</p>
            <p><strong>Reglas:</strong> {len(self.current_config.rules)}</p>
            <p><strong>Motor:</strong> Difuso Completo</p>
        </div>
        """)
        
        # Layout principal
        main_layout = pn.Column(
            header,
            pn.Row(left_panel, right_panel, help_panel, margin=(0, 20)),
            info_panel,
            width=1500,
            sizing_mode='fixed',
            margin=(0, 20, 20, 20)
        )
        
        # Guardar referencia para actualizaciones de tema
        self._main_layout = main_layout
        
        return main_layout
    
    def get_layout(self) -> pn.layout.Panel:
        """Obtener el layout principal."""
        return self.main_layout

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """Función principal para ejecutar la aplicación."""
    # Configurar Panel
    pn.extension('tabulator', sizing_mode='stretch_width')
    
    # Crear aplicación
    app = FuzzySystemApp()
    
    # Servir la aplicación
    pn.serve(
        app.get_layout(), 
        show=True, 
        port=5011,  # Puerto único
        allow_websocket_origin=["*"],
        title="Sistema Experto Difuso - Completo"
    )

if __name__ == "__main__":
    main()
