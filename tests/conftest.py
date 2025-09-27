"""
Pytest configuration and fixtures.
"""

import pytest
import numpy as np
from pathlib import Path

from fuzzy_panel.core.schema import create_example_config
from fuzzy_panel.core.engine import create_inference_engine


@pytest.fixture
def example_config():
    """Provide example configuration for tests."""
    return create_example_config()


@pytest.fixture
def inference_engine(example_config):
    """Provide inference engine for tests."""
    return create_inference_engine(example_config)


@pytest.fixture
def sample_inputs():
    """Provide sample input values for tests."""
    return {
        "Déficit": 5.0,
        "Presión": 50.0
    }


@pytest.fixture
def sample_inference_result(inference_engine, sample_inputs):
    """Provide sample inference result for tests."""
    return inference_engine.infer(sample_inputs)


@pytest.fixture
def temp_config_file(tmp_path):
    """Provide temporary configuration file for tests."""
    config_path = tmp_path / "test_config.json"
    return config_path


@pytest.fixture
def temp_output_file(tmp_path):
    """Provide temporary output file for tests."""
    output_path = tmp_path / "test_output.json"
    return output_path
