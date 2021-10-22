from enum import Enum, auto
from typing import Literal

class Stage(Enum):
    INDIVIDUAL = auto()
    BATCH = auto()
    PYTORCH = auto()
    

ALL_STAGES = Literal[Stage.INDIVIDUAL,
                     Stage.BATCH,
                     Stage.PYTORCH]
