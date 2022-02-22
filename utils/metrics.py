import numpy as np


def measure_porosity(
        micro_im: np.ndarray,
        pore_encoding: np.ndarray,
        padding_encoding: np.ndarray,
) -> float:
    r''' measure_porosity measures the porosity of the extracted images used in
        the machine learning model. These images are different from the
        microstructural image, so this measures the LOCAL (extracted) and not
        the GLOBAL (microstructure) porosity.

        Inputs:
        - micro_im: np.array; extracted image with particle centered

        Returns:
        - porosity: float
    '''

    Y, X, _ = micro_im.shape

    padding = np.all(micro_im == padding_encoding, axis=-1)
    padding = np.sum(padding)

    pore_space = np.all(micro_im == pore_encoding, axis=-1)
    pore_space = np.sum(pore_space)

    electrode_domain = float(Y * X - padding)
    porosity = pore_space / electrode_domain

    return porosity
