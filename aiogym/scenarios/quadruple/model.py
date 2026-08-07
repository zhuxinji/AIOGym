"""Environment adapter for the fixed-topology quadruple physics model."""

from aiogym.scenarios._shared import NumericProcessModel

from .physics import QuadruplePhysicsModel


class QuadrupleModel(NumericProcessModel):
    numerical_type = QuadruplePhysicsModel


__all__ = ["QuadrupleModel"]
