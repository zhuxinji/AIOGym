from aiogym.scenarios._shared import NumericProcessModel

from .numerical import CascadeModel as _NumericalModel


class CascadeModel(NumericProcessModel):
    numerical_type = _NumericalModel


__all__ = ["CascadeModel"]
