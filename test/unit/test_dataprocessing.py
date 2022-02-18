import os
from pathlib import Path
import json

import numpy as np

# Package-level imports
# Run from the terminal at the top-level directory...
#   python -m test.unit.test_dataprocessing
#
# https://stackoverflow.com/questions/14132789/relative-imports-for-the-billionth-time
from utils import typings
from create_micro_pngs import create_micro_png


class DataProcessingTest():

    def setup_class(self):

        micro_data_path = Path(
            os.path.dirname(os.path.realpath(__file__)),
            '../../test/fixtures/micro_1_data.json',
        )
        with open(micro_data_path, 'r') as f:
            self.micro_data: typings.Microstructure_Data = json.load(f)

        micro_arr_path = Path(
            os.path.dirname(os.path.realpath(__file__)),
            '../../test/fixtures/micro_1.npy',
        )
        self.micro_arr = np.load(micro_arr_path)

    def test_create_micro_array(self):
        L = 176
        h_cell = 100
        grid_size = 1000
        scale = 10

        # Values to encode phases
        pore_phase = 2 ** 16 - 1
        solid_phase = 2 ** 16 - 2

        micro_arr = create_micro_png(
            self.micro_data,
            L,
            h_cell,
            grid_size,
            scale,
            pore_phase=pore_phase,
            solid_phase=solid_phase,
        )

        assert np.allclose(micro_arr, self.micro_arr)


if __name__ == "__main__":
    t = DataProcessingTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print(f"Running test: {item}")
            t.__getattribute__(item)()
