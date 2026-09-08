"""The extension templates must stay importable against the current base classes.

A template is documentation that is also code: it is rendered verbatim on the
"Implement your own method" page and copied by contributors. Renaming a base
class or moving a module silently rots it, and nothing else in the test suite
imports these files.
"""

import importlib.util
from pathlib import Path

import pytest
from skbase.base import BaseObject

_TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "extension_templates"
_TEMPLATES = sorted(_TEMPLATE_DIR.glob("*.py"))


def _import_template(path):
    spec = importlib.util.spec_from_file_location(f"_template_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_templates_exist():
    assert _TEMPLATES


@pytest.mark.parametrize("path", _TEMPLATES, ids=lambda path: path.stem)
def test_template_class_is_importable(path):
    module = _import_template(path)

    (class_name,) = module.__all__
    template_class = getattr(module, class_name)

    assert issubclass(template_class, BaseObject)
    assert template_class.get_test_params()
