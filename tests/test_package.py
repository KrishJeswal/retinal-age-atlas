import importlib

import pytest

SUBPACKAGES = ["data", "audit", "models", "train", "analysis", "tracking"]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name):
    importlib.import_module(f"rra.{name}")
