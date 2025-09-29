#!/usr/bin/env python3
"""
Sistema Difuso Híbrido: Manual + Automático (Excel + Clustering)
Integra la funcionalidad de clustering automático con el sistema difuso existente
Mantiene compatibilidad con parametrización manual
"""

import os
import sys
import panel as pn
import param
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
import warnings
warnings.filterwarnings('ignore')

# Importar matplotlib con manejo de errores
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Advertencia: matplotlib no esta instalado. Instale con: pip install matplotlib")

# Importar skfuzzy con manejo de errores
try:
    import skfuzzy as fuzz
    SKFUZZY_AVAILABLE = True
except ImportError:
    SKFUZZY_AVAILABLE = False
    print("Advertencia: scikit-fuzzy no esta instalado. Instale con: pip install scikit-fuzzy")

# Importar componentes del sistema original
from fuzzy_system_complete import (
    TriangularMF, Variable, Term, Rule, RuleConclusion, 
    LogicConfig, FuzzySystemConfig, FuzzyInferenceEngine
)

# Configurar Panel
pn.extension('tabulator', sizing_mode='stretch_width')

class ClusteringConfig(BaseModel):
    """Configuración para clustering automático."""
    enabled: bool = False
    n_clusters: int = Field(default=2, ge=2, le=10)
    fuzziness: float = Field(default=2.0, ge=1.1, le=10.0)
    max_iterations: int = Field(default=1000, ge=100, le=5000)
    convergence_error: float = Field(default=0.005, ge=0.001, le=0.1)
    validation_enabled: bool = True

class DataSourceConfig(BaseModel):
    """Configuración para fuente de datos."""
    source_type: str = Field(default="manual")  # "manual", "excel", "hybrid"
    file_path: Optional[str] = None
    sheet_name: Optional[str] = None
    column_mapping: Dict[str, str] = Field(default_factory=dict)
    current_row: int = 0

class HybridSystemConfig(FuzzySystemConfig):
    """Configuración extendida para sistema híbrido."""
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    data_source: DataSourceConfig = Field(default_factory=DataSourceConfig)
    
    def get_variable(self, name: str) -> Optional[Variable]:
        """Obtener variable por nombre."""
        for var in self.variables:
            if var.name == name:
                return var
        return None

class ClusteringValidator:
    """Validador de calidad de clustering."""
    
    @staticmethod
    def calculate_partition_coefficient(u: np.ndarray) -> float:
        """Calcular coeficiente de partición (PC)."""
        return np.sum(u ** 2) / u.size
    
    @staticmethod
    def calculate_partition_entropy(u: np.ndarray) -> float:
        """Calcular entropía de partición (PE)."""
        u_safe = np.where(u == 0, 1e-10, u)  # Evitar log(0)
        return -np.sum(u * np.log(u_safe)) / u.size
    
    @staticmethod
    def calculate_xie_beni_index(data: np.ndarray, centers: np.ndarray, u: np.ndarray) -> float:
        """Calcular índice de Xie-Beni."""
        m = 2.0  # Parámetro de fuzziness
        n, c = u.shape
        
        # Calcular suma de distancias intra-cluster
        sum_intra = 0
        for i in range(c):
            for k in range(n):
                sum_intra += (u[k, i] ** m) * np.linalg.norm(data[k] - centers[i]) ** 2
        
        # Calcular distancia mínima inter-cluster
        min_inter = float('inf')
        for i in range(c):
            for j in range(i+1, c):
                dist = np.linalg.norm(centers[i] - centers[j]) ** 2
                min_inter = min(min_inter, dist)
        
        return sum_intra / (n * min_inter)
    
    def validate_clustering(self, data: np.ndarray, centers: np.ndarray, u: np.ndarray) -> Dict[str, float]:
        """Validar calidad del clustering."""
        return {
            'partition_coefficient': self.calculate_partition_coefficient(u),
            'partition_entropy': self.calculate_partition_entropy(u),
            'xie_beni_index': self.calculate_xie_beni_index(data, centers, u),
            'silhouette_score': self._calculate_silhouette_score(data, centers, u)
        }
    
    def _calculate_silhouette_score(self, data: np.ndarray, centers: np.ndarray, u: np.ndarray) -> float:
        """Calcular silhouette score simplificado."""
        # Asignar cada punto al cluster con mayor pertenencia
        cluster_labels = np.argmax(u, axis=1)
        
        # Calcular silhouette score
        if len(np.unique(cluster_labels)) < 2:
            return 0.0
        
        try:
            from sklearn.metrics import silhouette_score
            return silhouette_score(data.reshape(-1, 1), cluster_labels)
        except ImportError:
            # Implementación simplificada sin sklearn
            return 0.5  # Valor por defecto

class DataDrivenMFGenerator:
    """Generador de funciones de pertenencia basado en datos."""
    
    def __init__(self, validator: ClusteringValidator):
        self.validator = validator
    
    def generate_membership_functions(self, data: np.ndarray, config: ClusteringConfig) -> Tuple[List[Term], Dict[str, Any]]:
        """Generar funciones de pertenencia usando fuzzy c-means clustering."""
        
        if not SKFUZZY_AVAILABLE:
            print("scikit-fuzzy no disponible, usando terminos por defecto")
            return self._create_default_terms(data)
        
        # Preparar datos
        data_2d = np.expand_dims(data, axis=0).T
        
        # Aplicar fuzzy c-means clustering
        try:
            centers, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                data_2d, 
                c=config.n_clusters,
                m=config.fuzziness,
                error=config.convergence_error,
                maxiter=config.max_iterations,
                init=None
            )
            
            # Validar clustering si está habilitado
            validation_results = {}
            if config.validation_enabled:
                validation_results = self.validator.validate_clustering(data_2d, centers, u)
            
            # Ordenar centros para interpretabilidad
            sorted_indices = np.argsort(centers[:, 0])
            centers_sorted = centers[sorted_indices, 0]
            
            # Generar términos
            terms = self._create_triangular_terms(data, centers_sorted, config.n_clusters)
            
            # Información adicional
            clustering_info = {
                'centers': centers_sorted,
                'membership_matrix': u,
                'convergence_iterations': p,
                'final_objective': jm,
                'partition_coefficient': fpc,
                'validation_results': validation_results
            }
            
            return terms, clustering_info
            
        except Exception as e:
            print(f"Error en clustering: {e}")
            # Fallback a términos por defecto
            return self._create_default_terms(data)
    
    def _create_triangular_terms(self, data: np.ndarray, centers: np.ndarray, n_clusters: int) -> List[Term]:
        """Crear términos triangulares basados en centros de clustering."""
        min_val = float(data.min())
        max_val = float(data.max())
        
        terms = []
        
        if n_clusters == 2:
            # Términos Bajo y Alto
            terms = [
                Term(label="Bajo", mf=TriangularMF(a=min_val, b=centers[0], c=centers[1])),
                Term(label="Alto", mf=TriangularMF(a=centers[0], b=centers[1], c=max_val))
            ]
        elif n_clusters == 3:
            # Términos Bajo, Medio y Alto
            terms = [
                Term(label="Bajo", mf=TriangularMF(a=min_val, b=centers[0], c=centers[1])),
                Term(label="Medio", mf=TriangularMF(a=centers[0], b=centers[1], c=centers[2])),
                Term(label="Alto", mf=TriangularMF(a=centers[1], b=centers[2], c=max_val))
            ]
        else:
            # Términos genéricos para más clusters
            for i in range(n_clusters):
                if i == 0:
                    a, b, c = min_val, centers[0], centers[1]
                elif i == n_clusters - 1:
                    a, b, c = centers[i-1], centers[i], max_val
                else:
                    a, b, c = centers[i-1], centers[i], centers[i+1]
                
                label = f"Término_{i+1}"
                terms.append(Term(label=label, mf=TriangularMF(a=a, b=b, c=c)))
        
        return terms
    
    def _create_default_terms(self, data: np.ndarray) -> Tuple[List[Term], Dict[str, Any]]:
        """Crear términos por defecto si falla el clustering."""
        min_val = float(data.min())
        max_val = float(data.max())
        mid_val = (min_val + max_val) / 2
        
        terms = [
            Term(label="Bajo", mf=TriangularMF(a=min_val, b=min_val, c=mid_val)),
            Term(label="Alto", mf=TriangularMF(a=mid_val, b=max_val, c=max_val))
        ]
        
        return terms, {'error': 'Clustering falló, usando términos por defecto'}

class ExcelDataLoader:
    """Cargador de datos desde archivos Excel."""
    
    def __init__(self):
        self.current_data = None
        self.column_mapping = {}
    
    def load_excel_data(self, file_path: str, sheet_name: str = None) -> pd.DataFrame:
        """Cargar datos desde archivo Excel."""
        try:
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(file_path)
            
            self.current_data = df
            return df
        except Exception as e:
            raise ValueError(f"Error cargando Excel: {e}")
    
    def get_available_columns(self) -> List[str]:
        """Obtener columnas disponibles en los datos cargados."""
        if self.current_data is not None:
            return self.current_data.columns.tolist()
        return []
    
    def get_data_for_column(self, column_name: str) -> np.ndarray:
        """Obtener datos de una columna específica."""
        if self.current_data is not None and column_name in self.current_data.columns:
            return self.current_data[column_name].dropna().values
        return np.array([])
    
    def get_row_data(self, row_index: int, columns: List[str]) -> Dict[str, float]:
        """Obtener datos de una fila específica."""
        if self.current_data is not None and row_index < len(self.current_data):
            row_data = {}
            for col in columns:
                if col in self.current_data.columns:
                    row_data[col] = float(self.current_data.iloc[row_index][col])
            return row_data
        return {}

class HybridFuzzySystemApp(param.Parameterized):
    """Aplicación principal del sistema difuso híbrido."""
    
    # Parámetros principales
    current_config = param.ClassSelector(class_=HybridSystemConfig)
    inference_result = param.Parameter(default=None)
    input_values = param.Dict(default={})
    current_mode = param.String(default="manual")
    
    def __init__(self, **params):
        super().__init__(**params)
        
        # Inicializar componentes
        self.validator = ClusteringValidator()
        self.mf_generator = DataDrivenMFGenerator(self.validator)
        self.data_loader = ExcelDataLoader()
        
        # Crear configuración por defecto
        if 'current_config' not in params:
            self.current_config = self._create_default_config()
        
        # Inicializar UI
        self._initialize_ui()
        self._setup_callbacks()
    
    def _create_default_config(self) -> HybridSystemConfig:
        """Crear configuración por defecto."""
        # Usar la configuración del sistema original
        from fuzzy_system_complete import create_example_config
        original_config = create_example_config()
        
        # Convertir a configuración híbrida
        return HybridSystemConfig(
            project=original_config.project,
            description=original_config.description,
            author=original_config.author,
            logic=original_config.logic,
            variables=original_config.variables,
            rules=original_config.rules,
            clustering=ClusteringConfig(),
            data_source=DataSourceConfig()
        )
    
    def _initialize_ui(self):
        """Inicializar componentes de UI."""
        # Crear paneles principales
        self.main_panel = pn.Column(sizing_mode='stretch_width')
        
        # Crear controles de modo
        self._create_mode_controls()
        
        # Crear controles de entrada
        self._create_input_controls()
        
        # Crear controles de clustering
        self._create_clustering_controls()
        
        # Crear controles de Excel
        self._create_excel_controls()
        
        # Crear controles de inferencia
        self._create_inference_controls()
        
        # Crear paneles de resultados
        self._create_result_panels()
        
        # Crear layout principal
        self._create_main_layout()
    
    def _create_mode_controls(self):
        """Crear controles de modo de operación."""
        self.mode_selector = pn.widgets.RadioButtonGroup(
            name="Modo de Operación",
            options=["Manual", "Excel", "Híbrido"],
            value="Manual",
            button_type="primary",
            width=400
        )
        
        self.mode_info = pn.pane.HTML(
            """
            <div style="background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin: 10px 0;">
                <strong>Modos disponibles:</strong><br>
                • <strong>Manual:</strong> Configuración manual de variables y reglas<br>
                • <strong>Excel:</strong> Carga automática desde archivo Excel<br>
                • <strong>Híbrido:</strong> Combina configuración manual con datos de Excel
            </div>
            """,
            width=600
        )
    
    def _create_input_controls(self):
        """Crear controles de entrada de datos."""
        # Controles manuales (siempre visibles)
        self.deficit_input = pn.widgets.NumberInput(
            name="Déficit",
            value=5.0,
            start=0.0,
            end=10.0,
            step=0.1,
            width=150
        )
        
        self.presion_input = pn.widgets.NumberInput(
            name="Presión",
            value=50.0,
            start=0.0,
            end=100.0,
            step=1.0,
            width=150
        )
        
        # Botón de inferencia manual
        self.infer_button = pn.widgets.Button(
            name="Ejecutar Inferencia",
            button_type="primary",
            width=200
        )
    
    def _create_clustering_controls(self):
        """Crear controles de configuración de clustering."""
        self.clustering_enabled = pn.widgets.Checkbox(
            name="Habilitar Clustering Automático",
            value=False,
            width=250
        )
        
        self.n_clusters_input = pn.widgets.IntInput(
            name="Número de Clusters",
            value=2,
            start=2,
            end=10,
            width=150
        )
        
        self.fuzziness_input = pn.widgets.NumberInput(
            name="Parámetro de Fuzziness (m)",
            value=2.0,
            start=1.1,
            end=10.0,
            step=0.1,
            width=200
        )
        
        self.max_iterations_input = pn.widgets.IntInput(
            name="Máximo de Iteraciones",
            value=1000,
            start=100,
            end=5000,
            width=200
        )
        
        self.convergence_error_input = pn.widgets.NumberInput(
            name="Error de Convergencia",
            value=0.005,
            start=0.001,
            end=0.1,
            step=0.001,
            width=200
        )
        
        self.generate_mf_button = pn.widgets.Button(
            name="Generar Funciones de Pertenencia",
            button_type="success",
            width=300
        )
    
    def _create_excel_controls(self):
        """Crear controles para carga de Excel."""
        self.file_input = pn.widgets.FileInput(
            name="Cargar Archivo Excel",
            accept=".xlsx,.xls,.csv",
            width=300
        )
        
        self.sheet_selector = pn.widgets.Select(
            name="Hoja de Cálculo",
            options=[],
            width=200
        )
        
        self.column_selector = pn.widgets.MultiSelect(
            name="Seleccionar Columnas",
            options=[],
            width=300
        )
        
        self.row_selector = pn.widgets.IntSlider(
            name="Fila de Datos",
            value=0,
            start=0,
            end=0,
            width=300
        )
        
        self.load_excel_button = pn.widgets.Button(
            name="Cargar Datos",
            button_type="primary",
            width=150
        )
        
        self.auto_fill_button = pn.widgets.Button(
            name="Llenar desde Excel",
            button_type="success",
            width=200
        )
        
        # Display de estado
        self.status_display = pn.pane.HTML(
            "<strong>Estado:</strong> Sistema listo",
            width=600
        )
    
    def _create_inference_controls(self):
        """Crear controles de inferencia."""
        self.inference_button = pn.widgets.Button(
            name="Ejecutar Inferencia Completa",
            button_type="primary",
            width=250
        )
        
        self.clear_button = pn.widgets.Button(
            name="Limpiar Resultados",
            button_type="light",
            width=200
        )
    
    def _create_result_panels(self):
        """Crear paneles de resultados."""
        # Panel de resultados de inferencia
        self.results_panel = pn.pane.HTML(
            "<div style='padding: 20px; text-align: center; color: #666;'>No hay resultados disponibles</div>",
            width=800,
            height=300
        )
        
        # Panel de información de clustering
        self.clustering_info_panel = pn.pane.HTML(
            "<div style='padding: 20px; text-align: center; color: #666;'>Información de clustering no disponible</div>",
            width=800,
            height=200
        )
        
        # Panel de gráficos
        if MATPLOTLIB_AVAILABLE:
            self.plots_panel = pn.pane.Matplotlib(
                plt.figure(figsize=(12, 8)),
                width=800,
                height=400
            )
        else:
            self.plots_panel = pn.pane.HTML(
                "<div style='padding: 20px; text-align: center; color: #666;'>Matplotlib no disponible para gráficos</div>",
                width=800,
                height=400
            )
    
    def _create_main_layout(self):
        """Crear layout principal."""
        # Tabs principales
        self.main_tabs = pn.Tabs(
            ("Configuración", self._create_config_tab()),
            ("Entrada de Datos", self._create_input_tab()),
            ("Clustering", self._create_clustering_tab()),
            ("Resultados", self._create_results_tab()),
            tabs_location='above'
        )
        
        self.main_panel.append(self.main_tabs)
    
    def _create_config_tab(self):
        """Crear tab de configuración."""
        return pn.Column(
            pn.Row(
                self.mode_selector,
                pn.Spacer(width=50),
                self.clustering_enabled
            ),
            self.mode_info,
            pn.Spacer(height=20),
            pn.pane.HTML("<h3>Configuración del Sistema</h3>"),
            pn.Row(
                pn.Column(
                    pn.pane.HTML("<h4>Variables del Sistema</h4>"),
                    self._create_variable_display(),
                    width=400
                ),
                pn.Column(
                    pn.pane.HTML("<h4>Reglas del Sistema</h4>"),
                    self._create_rules_display(),
                    width=400
                )
            )
        )
    
    def _create_input_tab(self):
        """Crear tab de entrada de datos."""
        return pn.Column(
            pn.pane.HTML("<h3>Entrada de Datos</h3>"),
            
            # Controles manuales
            pn.Card(
                pn.Column(
                    pn.pane.HTML("<h4>Entrada Manual</h4>"),
                    pn.Row(self.deficit_input, self.presion_input),
                    self.infer_button
                ),
                title="Modo Manual",
                width=500
            ),
            
            pn.Spacer(height=20),
            
            # Controles de Excel
            pn.Card(
                pn.Column(
                    pn.pane.HTML("<h4>Carga desde Excel</h4>"),
                    pn.Row(self.file_input, self.load_excel_button),
                    pn.Row(self.sheet_selector, self.column_selector),
                    pn.Row(self.row_selector, self.auto_fill_button),
                    pn.Spacer(height=10),
                    self.status_display
                ),
                title="Modo Excel",
                width=600
            ),
            
            pn.Spacer(height=20),
            
            # Botón de inferencia completa
            pn.Row(
                self.inference_button,
                self.clear_button,
                pn.Spacer()
            )
        )
    
    def _create_clustering_tab(self):
        """Crear tab de clustering."""
        return pn.Column(
            pn.pane.HTML("<h3>Configuración de Clustering</h3>"),
            
            pn.Card(
                pn.Column(
                    pn.Row(
                        pn.Column(
                            self.n_clusters_input,
                            self.fuzziness_input,
                            width=200
                        ),
                        pn.Column(
                            self.max_iterations_input,
                            self.convergence_error_input,
                            width=200
                        )
                    ),
                    pn.Spacer(height=10),
                    self.generate_mf_button
                ),
                title="Parámetros de Clustering",
                width=500
            ),
            
            pn.Spacer(height=20),
            
            # Información de clustering
            pn.Card(
                self.clustering_info_panel,
                title="Información de Clustering",
                width=800
            )
        )
    
    def _create_results_tab(self):
        """Crear tab de resultados."""
        return pn.Column(
            pn.pane.HTML("<h3>Resultados del Sistema</h3>"),
            
            # Resultados de inferencia
            pn.Card(
                self.results_panel,
                title="Resultados de Inferencia",
                width=800
            ),
            
            pn.Spacer(height=20),
            
            # Gráficos
            pn.Card(
                self.plots_panel,
                title="Visualizaciones",
                width=800
            )
        )
    
    def _create_variable_display(self):
        """Crear display de variables."""
        html = "<div style='background-color: #f5f5f5; padding: 15px; border-radius: 5px;'>"
        html += "<h4>Variables del Sistema</h4><ul>"
        
        for var in self.current_config.variables:
            html += f"<li><strong>{var.name}</strong> ({var.kind})</li>"
            html += f"<ul><li>Universo: {var.universe}</li>"
            html += f"<li>Términos: {', '.join([term.label for term in var.terms])}</li></ul>"
        
        html += "</ul></div>"
        return pn.pane.HTML(html)
    
    def _create_rules_display(self):
        """Crear display de reglas."""
        html = "<div style='background-color: #f5f5f5; padding: 15px; border-radius: 5px;'>"
        html += "<h4>Reglas del Sistema</h4><ul>"
        
        for rule in self.current_config.rules:
            html += f"<li><strong>{rule.id}:</strong> {rule.if_condition} → {rule.then_conclusions[0].variable} is {rule.then_conclusions[0].term}</li>"
        
        html += "</ul></div>"
        return pn.pane.HTML(html)
    
    def _setup_callbacks(self):
        """Configurar callbacks de UI."""
        # Callbacks de modo
        self.mode_selector.param.watch(self._on_mode_change, 'value')
        
        # Callbacks de clustering
        self.clustering_enabled.param.watch(self._on_clustering_enabled, 'value')
        self.generate_mf_button.on_click(self._generate_membership_functions)
        
        # Callbacks de Excel
        self.file_input.param.watch(self._on_file_upload, 'value')
        self.load_excel_button.on_click(self._load_excel_data)
        self.auto_fill_button.on_click(self._auto_fill_from_excel)
        
        # Callbacks de inferencia
        self.infer_button.on_click(self._execute_manual_inference)
        self.inference_button.on_click(self._execute_complete_inference)
        self.clear_button.on_click(self._clear_results)
    
    def _on_mode_change(self, event):
        """Manejar cambio de modo."""
        self.current_mode = event.new
        print(f"Modo cambiado a: {self.current_mode}")
    
    def _on_clustering_enabled(self, event):
        """Manejar habilitación de clustering."""
        self.current_config.clustering.enabled = event.new
        print(f"Clustering habilitado: {event.new}")
    
    def _on_file_upload(self, event):
        """Manejar carga de archivo."""
        if event.new is not None:
            self._update_status("Archivo cargado, use 'Cargar Datos' para procesar")
    
    def _load_excel_data(self, event):
        """Cargar datos desde Excel."""
        if self.file_input.value is None:
            self._update_status("No se ha seleccionado archivo")
            return
        
        try:
            # Guardar archivo temporalmente
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                tmp_file.write(self.file_input.value)
                tmp_file_path = tmp_file.name
            
            # Cargar datos
            df = self.data_loader.load_excel_data(tmp_file_path, self.sheet_selector.value)
            
            # Actualizar controles
            self.column_selector.options = self.data_loader.get_available_columns()
            self.row_selector.end = len(df) - 1
            
            # Limpiar archivo temporal
            os.unlink(tmp_file_path)
            
            self._update_status(f"Datos cargados: {len(df)} filas, {len(df.columns)} columnas")
            
        except Exception as e:
            self._update_status(f"Error cargando Excel: {str(e)}")
    
    def _auto_fill_from_excel(self, event):
        """Llenar campos desde Excel."""
        if self.data_loader.current_data is None:
            self._update_status("No hay datos cargados")
            return
        
        try:
            # Obtener datos de la fila seleccionada
            selected_columns = self.column_selector.value
            row_data = self.data_loader.get_row_data(self.row_selector.value, selected_columns)
            
            # Mapear columnas a variables del sistema
            column_mapping = {
                'yreal': 'Déficit',
                'INF': 'Presión',
                'deficit': 'Déficit', 
                'presion': 'Presión'
            }
            
            # Llenar campos manuales
            for col, value in row_data.items():
                if col in column_mapping:
                    var_name = column_mapping[col]
                    if var_name == 'Déficit':
                        self.deficit_input.value = value
                    elif var_name == 'Presión':
                        self.presion_input.value = value
            
            self._update_status(f"Datos llenados desde fila {self.row_selector.value}")
            
        except Exception as e:
            self._update_status(f"Error llenando campos: {str(e)}")
    
    def _generate_membership_functions(self, event):
        """Generar funciones de pertenencia usando clustering."""
        if not self.clustering_enabled.value:
            self._update_status("Clustering no habilitado")
            return
        
        try:
            # Obtener configuración de clustering
            clustering_config = ClusteringConfig(
                enabled=True,
                n_clusters=self.n_clusters_input.value,
                fuzziness=self.fuzziness_input.value,
                max_iterations=self.max_iterations_input.value,
                convergence_error=self.convergence_error_input.value
            )
            
            # Generar MF para cada variable si hay datos
            if self.data_loader.current_data is not None:
                selected_columns = self.column_selector.value
                new_variables = []
                
                for var in self.current_config.variables:
                    if var.name in ['Déficit', 'Presión']:  # Variables de entrada
                        # Mapear a columnas Excel
                        column_mapping = {'Déficit': 'yreal', 'Presión': 'INF'}
                        excel_col = column_mapping.get(var.name)
                        
                        if excel_col in selected_columns:
                            # Obtener datos de la columna
                            data = self.data_loader.get_data_for_column(excel_col)
                            
                            if len(data) > 0:
                                # Generar nuevas funciones de pertenencia
                                new_terms, clustering_info = self.mf_generator.generate_membership_functions(
                                    data, clustering_config
                                )
                                
                                # Crear nueva variable con clustering
                                new_var = Variable(
                                    name=var.name,
                                    kind=var.kind,
                                    universe=[float(data.min()), float(data.max())],
                                    defuzz=var.defuzz,
                                    terms=new_terms
                                )
                                new_variables.append(new_var)
                                
                                # Mostrar información de clustering
                                self._display_clustering_info(clustering_info)
                            else:
                                new_variables.append(var)
                        else:
                            new_variables.append(var)
                    else:
                        new_variables.append(var)
                
                # Actualizar configuración
                self.current_config.variables = new_variables
                self._update_status("Funciones de pertenencia generadas exitosamente")
            else:
                self._update_status("No hay datos disponibles para clustering")
                
        except Exception as e:
            self._update_status(f"Error generando MF: {str(e)}")
    
    def _execute_manual_inference(self, event):
        """Ejecutar inferencia manual."""
        try:
            # Obtener valores de entrada
            input_values = {
                'Déficit': self.deficit_input.value,
                'Presión': self.presion_input.value
            }
            
            # Validar entradas
            validation = self._validate_inputs(input_values)
            if not all(v['valid'] for v in validation.values()):
                self._update_status("Valores de entrada fuera de rango")
                return
            
            # Crear motor de inferencia
            engine = FuzzyInferenceEngine(self.current_config)
            
            # Ejecutar inferencia
            result = engine.infer(input_values)
            
            # Mostrar resultados
            self._display_inference_results(result)
            self._update_status("Inferencia manual completada")
            
        except Exception as e:
            self._update_status(f"Error en inferencia manual: {str(e)}")
    
    def _execute_complete_inference(self, event):
        """Ejecutar inferencia completa."""
        if self.current_mode == "Excel":
            # Usar datos de Excel
            if self.data_loader.current_data is None:
                self._update_status("No hay datos de Excel cargados")
                return
            
            try:
                # Obtener datos de la fila actual
                selected_columns = self.column_selector.value
                row_data = self.data_loader.get_row_data(self.row_selector.value, selected_columns)
                
                # Mapear a variables del sistema
                column_mapping = {'yreal': 'Déficit', 'INF': 'Presión'}
                input_values = {}
                
                for col, value in row_data.items():
                    if col in column_mapping:
                        input_values[column_mapping[col]] = value
                
                # Ejecutar inferencia
                engine = FuzzyInferenceEngine(self.current_config)
                result = engine.infer(input_values)
                
                # Mostrar resultados
                self._display_inference_results(result)
                self._update_status(f"Inferencia Excel completada - Fila {self.row_selector.value}")
                
            except Exception as e:
                self._update_status(f"Error en inferencia Excel: {str(e)}")
        
        elif self.current_mode == "Híbrido":
            # Combinar manual y Excel
            self._execute_hybrid_inference()
        
        else:
            # Modo manual
            self._execute_manual_inference(event)
    
    def _clear_results(self, event):
        """Limpiar resultados."""
        self.results_panel.object = "<div style='padding: 20px; text-align: center; color: #666;'>Resultados limpiados</div>"
        self.clustering_info_panel.object = "<div style='padding: 20px; text-align: center; color: #666;'>Información de clustering no disponible</div>"
        print("Resultados limpiados")
    
    def _update_status(self, message: str):
        """Actualizar mensaje de estado."""
        # Buscar el elemento de estado en la UI
        status_element = getattr(self, 'status_display', None)
        if status_element:
            status_element.object = f"<strong>Estado:</strong> {message}"
        print(f"Estado: {message}")
    
    def _validate_inputs(self, inputs: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Validar que las entradas estén dentro del universo de las variables."""
        validation_results = {}
        
        for var_name, value in inputs.items():
            variable = self.current_config.get_variable(var_name)
            if variable:
                universe = variable.universe
                is_valid = universe[0] <= value <= universe[1]
                validation_results[var_name] = {
                    'valid': is_valid,
                    'value': value,
                    'universe': universe,
                    'source': 'manual' if var_name in ['Déficit', 'Presión'] else 'excel'
                }
        
        return validation_results
    
    def _display_inference_results(self, result):
        """Mostrar resultados de inferencia."""
        html = f"""
        <div style="background-color: #f5f5f5; padding: 20px; border-radius: 5px;">
            <h3 style="margin: 0 0 15px 0; color: #2e7d32;">📊 Resultados de Inferencia</h3>
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
            """
            
            # Mostrar información adicional según el tipo de interpretación
            if linguistic.get('interpretation_type') == 'ambiguous':
                html += f"""
                    <br><span style="color: #FF9800; font-size: 0.85em; font-style: italic;">
                        ⚠️ Ambiguo: Múltiples términos con igual confianza máxima
                    </span>
                """
            elif linguistic.get('interpretation_type') == 'no_membership':
                html += f"""
                    <br><span style="color: #f44336; font-size: 0.85em; font-style: italic;">
                        ⚠️ Sin pertenencia: El valor no pertenece a ningún término
                    </span>
                """
            
            html += """
                </div>
                <div style="font-size: 0.9em; color: #666;">
                    <strong>Todos los términos:</strong>
                    <ul style="margin: 5px 0; padding-left: 20px;">
            """
            
            for term, confidence in linguistic.get('all_terms', {}).items():
                # Resaltar términos ambiguos
                if term in linguistic.get('ambiguous_terms', []):
                    html += f"<li><strong>{term}: {confidence:.3f}</strong> <span style='color: #FF9800;'>(ambiguo)</span></li>"
                else:
                    html += f"<li>{term}: {confidence:.3f}</li>"
            
            html += """
                    </ul>
                </div>
            </div>
            """
        
        html += "</div>"
        self.results_panel.object = html
    
    def _get_linguistic_interpretation(self, var_name: str, crisp_value: float) -> Dict[str, Any]:
        """Obtener interpretación lingüística del resultado conforme a la teoría de sistemas difusos."""
        var = self.current_config.get_variable(var_name)
        if var is None:
            return {
                "best_term": "Desconocido", 
                "confidence": 0.0, 
                "all_terms": {}, 
                "ambiguous_terms": [],
                "interpretation_type": "error"
            }
        
        # Crear motor temporal para evaluar funciones de pertenencia
        engine = FuzzyInferenceEngine(self.current_config)
        
        all_terms = {}
        max_confidence = 0.0
        ambiguous_terms = []
        
        # Recopilar todas las confianzas y encontrar la máxima
        for term in var.terms:
            key = (var_name, term.label)
            mf = engine.membership_functions[key]
            confidence = float(mf(crisp_value))
            all_terms[term.label] = confidence
            
            if confidence > max_confidence:
                max_confidence = confidence
                ambiguous_terms = [term.label]  # Resetear la lista
            elif confidence == max_confidence and confidence > 0.0:
                ambiguous_terms.append(term.label)  # Agregar términos con igual confianza máxima
        
        # Determinar el resultado según la teoría de sistemas difusos
        if max_confidence == 0.0:
            # Ningún término tiene pertenencia > 0
            return {
                "best_term": "Desconocido",
                "confidence": 0.0,
                "all_terms": all_terms,
                "ambiguous_terms": [],
                "interpretation_type": "no_membership"
            }
        elif len(ambiguous_terms) == 1:
            # Un solo término con máxima confianza
            return {
                "best_term": ambiguous_terms[0],
                "confidence": max_confidence,
                "all_terms": all_terms,
                "ambiguous_terms": [],
                "interpretation_type": "clear"
            }
        else:
            # Múltiples términos con igual confianza máxima (ambigüedad)
            return {
                "best_term": "/".join(ambiguous_terms),  # Mostrar todos los términos ambiguos
                "confidence": max_confidence,
                "all_terms": all_terms,
                "ambiguous_terms": ambiguous_terms,
                "interpretation_type": "ambiguous"
            }
    
    def _display_clustering_info(self, clustering_info: Dict[str, Any]):
        """Mostrar información de clustering."""
        html = f"""
        <div style="background-color: #e3f2fd; padding: 15px; border-radius: 5px;">
            <h4 style="margin: 0 0 10px 0; color: #1976D2;">📈 Información de Clustering</h4>
        """
        
        if 'error' in clustering_info:
            html += f"<p style='color: #f44336;'>{clustering_info['error']}</p>"
        else:
            html += f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div><strong>Centros de Cluster:</strong> {clustering_info.get('centers', 'N/A')}</div>
                <div><strong>Iteraciones:</strong> {clustering_info.get('convergence_iterations', 'N/A')}</div>
                <div><strong>Función Objetivo:</strong> {clustering_info.get('final_objective', 'N/A'):.4f}</div>
                <div><strong>Partition Coefficient:</strong> {clustering_info.get('partition_coefficient', 'N/A'):.4f}</div>
            </div>
            """
            
            # Mostrar métricas de validación si están disponibles
            validation_results = clustering_info.get('validation_results', {})
            if validation_results:
                html += """
                <div style="margin-top: 10px;">
                    <strong>Métricas de Validación:</strong>
                    <ul style="margin: 5px 0; padding-left: 20px;">
                """
                for metric, value in validation_results.items():
                    html += f"<li>{metric}: {value:.4f}</li>"
                html += "</ul></div>"
        
        html += "</div>"
        self.clustering_info_panel.object = html
    
    def _execute_hybrid_inference(self):
        """Ejecutar inferencia híbrida."""
        try:
            # Combinar datos manuales y de Excel
            input_values = {
                'Déficit': self.deficit_input.value,
                'Presión': self.presion_input.value
            }
            
            # Si hay datos de Excel, usar algunos valores de ahí
            if self.data_loader.current_data is not None:
                selected_columns = self.column_selector.value
                row_data = self.data_loader.get_row_data(self.row_selector.value, selected_columns)
                
                # Mapear columnas Excel a variables
                column_mapping = {'yreal': 'Déficit', 'INF': 'Presión'}
                for col, value in row_data.items():
                    if col in column_mapping:
                        # Permitir al usuario elegir qué valores usar
                        var_name = column_mapping[col]
                        if self.current_mode == "Híbrido":
                            # En modo híbrido, combinar ambos
                            excel_value = value
                            manual_value = input_values[var_name]
                            # Usar promedio o permitir selección
                            input_values[var_name] = (excel_value + manual_value) / 2
            
            # Validar y ejecutar inferencia
            validation = self._validate_inputs(input_values)
            if not all(v['valid'] for v in validation.values()):
                self._update_status("Valores de entrada fuera de rango")
                return
            
            engine = FuzzyInferenceEngine(self.current_config)
            result = engine.infer(input_values)
            
            self._display_inference_results(result)
            self._update_status("Inferencia híbrida completada")
            
        except Exception as e:
            self._update_status(f"Error en inferencia híbrida: {str(e)}")

def create_hybrid_app():
    """Crear aplicación híbrida."""
    app = HybridFuzzySystemApp()
    return app.main_panel

def main():
    """Función principal."""
    app = create_hybrid_app()
    return app

if __name__ == "__main__":
    # Crear y mostrar la aplicación
    app = main()
    app.servable("Sistema Difuso Híbrido")
    
    # Para desarrollo local
    app.show()
