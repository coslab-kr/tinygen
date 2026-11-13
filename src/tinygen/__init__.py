"""TinyGen's Python interface to the native TOSA compiler."""
from .api import compile_model, lower_model

__all__ = ["compile_model", "lower_model"]
__version__ = "0.1.0"
