from aiogym.scenarios._shared import NumericProcessModel

from .numerical import QuadrupleModel as _NumericalModel


class QuadrupleModel(NumericProcessModel):
    numerical_type = _NumericalModel


__all__ = ["QuadrupleModel"]
