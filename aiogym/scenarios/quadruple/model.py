from aiogym.scenarios._legacy import LegacyProcessModel


class QuadrupleModel(LegacyProcessModel):
    def __init__(self, plant):
        super().__init__("quadruple", plant)


__all__ = ["QuadrupleModel"]
