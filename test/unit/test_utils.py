import os
from pathlib import Path

from math import ceil
import numpy as np

# Package-level imports
# Run from the terminal at the top-level directory...
#   python -m test.unit.test_utils
#
# https://stackoverflow.com/questions/14132789/relative-imports-for-the-billionth-time
from utils.image import extract_input
from utils.metrics import measure_porosity


class UtilsTest():

    def setup_class(self):
        micro_arr_path = Path(
            os.path.dirname(os.path.realpath(__file__)),
            '../../test/fixtures/micro_1.npy',
        )
        self.micro_arr = np.load(micro_arr_path)

    def test_measure_porosity(self):
        x = 60.7449
        y = 94.5362
        R = 4.9806

        pore_encoding: np.uint16 = 65535
        padding_encoding: np.uint16 = 62000

        scale: int = 10
        width_wrt_radius: int = 3

        box_radius = ceil(R * scale * width_wrt_radius)

        local_micro = extract_input(
            box_radius,
            self.micro_arr,
            (str(x), str(y), ""),
            scale,
            padding=padding_encoding
        )

        porosity = measure_porosity(
            local_micro,
            np.array(pore_encoding),
            np.array(padding_encoding),
        )

        assert np.allclose(porosity, 0.4970489)


if __name__ == "__main__":
    t = UtilsTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith("test"):
            print(f"Running test: {item}")
            t.__getattribute__(item)()
