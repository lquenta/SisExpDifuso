"""
Pydantic models for fuzzy system configuration schema validation.
"""

from typing import List, Dict, Any, Union, Literal, Optional
from pydantic import BaseModel, Field, validator
import numpy as np


class TriangularMF(BaseModel):
    """Triangular membership function parameters."""
    type: Literal["tri"] = "tri"
    a: float = Field(..., description="Left vertex (x-coordinate)")
    b: float = Field(..., description="Peak vertex (x-coordinate)")
    c: float = Field(..., description="Right vertex (x-coordinate)")
    
    @validator('a', 'b', 'c')
    def validate_vertices(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError("Vertices must be numeric")
        return float(v)
    
    @validator('c')
    def validate_triangle_order(cls, v, values):
        if 'a' in values and 'b' in values:
            a, b = values['a'], values['b']
            if not (a <= b <= v):
                raise ValueError("Triangular vertices must satisfy a <= b <= c")
        return v


class TrapezoidalMF(BaseModel):
    """Trapezoidal membership function parameters (future extension)."""
    type: Literal["trap"] = "trap"
    a: float = Field(..., description="Left foot (x-coordinate)")
    b: float = Field(..., description="Left shoulder (x-coordinate)")
    c: float = Field(..., description="Right shoulder (x-coordinate)")
    d: float = Field(..., description="Right foot (x-coordinate)")
    
    @validator('a', 'b', 'c', 'd')
    def validate_vertices(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError("Vertices must be numeric")
        return float(v)
    
    @validator('d')
    def validate_trapezoid_order(cls, v, values):
        if all(k in values for k in ['a', 'b', 'c']):
            a, b, c = values['a'], values['b'], values['c']
            if not (a <= b <= c <= v):
                raise ValueError("Trapezoidal vertices must satisfy a <= b <= c <= d")
        return v


class GaussianMF(BaseModel):
    """Gaussian membership function parameters (future extension)."""
    type: Literal["gauss"] = "gauss"
    center: float = Field(..., description="Center (mean)")
    width: float = Field(..., description="Width (standard deviation)")
    
    @validator('center', 'width')
    def validate_parameters(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError("Parameters must be numeric")
        return float(v)
    
    @validator('width')
    def validate_positive_width(cls, v):
        if v <= 0:
            raise ValueError("Gaussian width must be positive")
        return v


# Union type for all membership function types
MembershipFunction = Union[TriangularMF, TrapezoidalMF, GaussianMF]


class Term(BaseModel):
    """Fuzzy term with label and membership function."""
    label: str = Field(..., description="Linguistic label (e.g., 'Bajo', 'Medio', 'Alto')")
    mf: MembershipFunction = Field(..., description="Membership function parameters")
    
    @validator('label')
    def validate_label(cls, v):
        if not v or not v.strip():
            raise ValueError("Label cannot be empty")
        return v.strip()


class Variable(BaseModel):
    """Fuzzy variable (input or output)."""
    name: str = Field(..., description="Variable name (unique identifier)")
    kind: Literal["input", "output"] = Field(..., description="Variable type")
    universe: List[float] = Field(..., description="Universe of discourse [min, max]")
    terms: List[Term] = Field(..., description="Fuzzy terms for this variable")
    defuzz: Optional[Literal["centroid", "bisector", "mean_of_maxima"]] = Field(
        default="centroid", 
        description="Defuzzification method (for output variables)"
    )
    
    @validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Variable name cannot be empty")
        return v.strip()
    
    @validator('universe')
    def validate_universe(cls, v):
        if len(v) != 2:
            raise ValueError("Universe must have exactly 2 elements [min, max]")
        min_val, max_val = v
        if min_val >= max_val:
            raise ValueError("Universe min must be less than max")
        return [float(min_val), float(max_val)]
    
    @validator('terms')
    def validate_terms(cls, v):
        if not v:
            raise ValueError("Variable must have at least one term")
        
        # Check for duplicate labels
        labels = [term.label for term in v]
        if len(labels) != len(set(labels)):
            raise ValueError("Term labels must be unique within a variable")
        
        return v
    
    @validator('defuzz')
    def validate_defuzz_for_output(cls, v, values):
        if values.get('kind') == 'output' and v is None:
            return 'centroid'  # Default for output variables
        return v


class LogicConfig(BaseModel):
    """Logical operators configuration."""
    and_op: Literal["min", "prod"] = Field(default="min", description="AND operator")
    or_op: Literal["max", "prob_or"] = Field(default="max", description="OR operator")
    implication: Literal["min"] = Field(default="min", description="Implication method")
    aggregation: Literal["max"] = Field(default="max", description="Aggregation method")
    defuzz_default: Literal["centroid", "bisector", "mean_of_maxima"] = Field(
        default="centroid", 
        description="Default defuzzification method"
    )


class RuleConclusion(BaseModel):
    """Single conclusion in a rule (variable: term assignment)."""
    variable: str = Field(..., description="Output variable name")
    term: str = Field(..., description="Output term label")
    
    @validator('variable', 'term')
    def validate_non_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Variable and term names cannot be empty")
        return v.strip()


class Rule(BaseModel):
    """Fuzzy rule with natural language syntax."""
    id: str = Field(..., description="Unique rule identifier")
    if_condition: str = Field(..., alias="if", description="Antecedent condition")
    then_conclusions: List[RuleConclusion] = Field(..., alias="then", description="Consequent conclusions")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Rule weight")
    note: Optional[str] = Field(default=None, description="Rule description or note")
    
    class Config:
        populate_by_name = True
    
    @validator('id')
    def validate_id(cls, v):
        if not v or not v.strip():
            raise ValueError("Rule ID cannot be empty")
        return v.strip()
    
    @validator('if_condition')
    def validate_condition(cls, v):
        if not v or not v.strip():
            raise ValueError("Rule condition cannot be empty")
        return v.strip()
    
    @validator('then_conclusions')
    def validate_conclusions(cls, v):
        if not v:
            raise ValueError("Rule must have at least one conclusion")
        return v


class FuzzySystemConfig(BaseModel):
    """Complete fuzzy system configuration."""
    schema_version: str = Field(default="1.0", description="Schema version")
    project: str = Field(..., description="Project name")
    description: str = Field(default="", description="Project description")
    author: str = Field(default="", description="Project author")
    logic: LogicConfig = Field(default_factory=LogicConfig, description="Logical operators")
    variables: List[Variable] = Field(..., description="System variables")
    rules: List[Rule] = Field(..., description="Fuzzy rules")
    
    @validator('project')
    def validate_project_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Project name cannot be empty")
        return v.strip()
    
    @validator('variables')
    def validate_variables(cls, v):
        if not v:
            raise ValueError("System must have at least one variable")
        
        # Check for duplicate variable names
        names = [var.name for var in v]
        if len(names) != len(set(names)):
            raise ValueError("Variable names must be unique")
        
        # Check for at least one input and one output
        kinds = [var.kind for var in v]
        if "input" not in kinds:
            raise ValueError("System must have at least one input variable")
        if "output" not in kinds:
            raise ValueError("System must have at least one output variable")
        
        return v
    
    @validator('rules')
    def validate_rules(cls, v):
        if not v:
            raise ValueError("System must have at least one rule")
        
        # Check for duplicate rule IDs
        ids = [rule.id for rule in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Rule IDs must be unique")
        
        return v
    
    def get_variable(self, name: str) -> Optional[Variable]:
        """Get variable by name."""
        for var in self.variables:
            if var.name == name:
                return var
        return None
    
    def get_input_variables(self) -> List[Variable]:
        """Get all input variables."""
        return [var for var in self.variables if var.kind == "input"]
    
    def get_output_variables(self) -> List[Variable]:
        """Get all output variables."""
        return [var for var in self.variables if var.kind == "output"]
    
    def validate_rule_references(self) -> List[str]:
        """Validate that all rule references point to existing variables and terms."""
        errors = []
        
        for rule in self.rules:
            # This will be implemented in the rule parser
            # For now, just basic validation
            pass
        
        return errors


# Example configuration for testing
def create_example_config() -> FuzzySystemConfig:
    """Create an example configuration for testing."""
    return FuzzySystemConfig(
        schema_version="1.0",
        project="Politica_Arancelaria",
        description="Sistema experto para determinar política arancelaria basada en déficit y presión",
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
                if_condition="Déficit is Medio AND Presión is Alta",
                then_conclusions=[RuleConclusion(variable="Tarifa", term="Moderada")],
                weight=1.0,
                note="Equilibrio protección–competitividad"
            )
        ]
    )
