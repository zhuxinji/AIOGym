from aiogym.scenarios._legacy import LegacyProcessModel


class RecirculatingCascadeModel(LegacyProcessModel):
    def __init__(self, plant):
        super().__init__("cascade_recirculating", plant)


__all__ = ["RecirculatingCascadeModel"]
