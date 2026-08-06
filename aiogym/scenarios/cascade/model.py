from aiogym.scenarios._legacy import LegacyProcessModel


class CascadeModel(LegacyProcessModel):
    def __init__(self, plant):
        super().__init__("cascade", plant)


__all__ = ["CascadeModel"]
