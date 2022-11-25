from typing import Tuple, TypedDict, List, Callable
import numpy as np
import tensorflow as tf

meshgrid = Tuple[np.ndarray, np.ndarray]


ETL_key_fn = Callable[[int], tf.types.experimental.TensorLike]
ETL_key_and_tensor_fn = Callable[
    [int, tf.types.experimental.TensorLike], tf.types.experimental.TensorLike
]
ETL_fn = ETL_key_and_tensor_fn


class Circle_Info(TypedDict):
    x: str
    y: str
    R: str


class Microstructure_Data(TypedDict):
    id: int
    length: int
    porosity: str
    tortuosity: str
    circles: List[Circle_Info]


class Metadata(TypedDict):
    micro: str
    x: float
    y: str
    R: str
    L: int
    zoom_factor: float
    c_rate: str
    time: str
    dist_from_sep: float
    porosity: float


class Metadata_Normalizations(TypedDict):
    L: int
    h_cell: int
    R_max: float
    zoom_norm: float
    c_rate_norm: float
    time_norm: int


META_INDICES = {
    "x": 0,
    "y": 1,
    "zoom_factor": 4,
}
