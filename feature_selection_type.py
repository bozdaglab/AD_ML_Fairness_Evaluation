from enum import Enum, auto


class FeatureSelectionType(Enum):
    RFE = auto()
    SelectFromModel = auto()
    SequentialFeatureSelector = auto()
    SelectBySingleFeaturePerformance = auto()
    SelectByShuffling = auto()
    GeneticSelectionCV = auto()
    BorutaPy = auto
