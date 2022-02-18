from typing import Tuple, TypedDict, List
import numpy as np

meshgrid = Tuple[np.ndarray, np.ndarray]


class Circle_Info(TypedDict):
    x: str
    y: str
    R: str


class Microstructure_Data(TypedDict):
    id: int
    porosity: str
    tortuosity: str
    circles: List[Circle_Info]
