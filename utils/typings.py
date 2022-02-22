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


class Metadata(TypedDict):
    micro: str
    x: str
    y: str
    R: str
    zoom_factor: float
    c_rate: str
    time: str
    dist_from_sep: float
    porosity: float
