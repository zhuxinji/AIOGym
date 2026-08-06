from aiogym.scenarios._shared import NumericProcessModel

from .numerical import RecirculatingCascadeModel as _NumericalModel


class RecirculatingCascadeModel(NumericProcessModel):
    numerical_type = _NumericalModel


__all__ = ["RecirculatingCascadeModel"]
