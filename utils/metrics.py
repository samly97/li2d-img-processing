from utils import typings

from math import ceil

import numpy as np
import porespy as ps


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


def tortuosity_to_particle(
    micro_im: np.ndarray,
    particle: typings.Circle_Info,
    width_wrt_radius: int,
    pore_encoding: int,
    scale: int = 10,
) -> float:

    micro_im = np.copy(micro_im)
    Y_max = np.shape(micro_im)[1]

    # Location of particle centroids in pixel-scale
    x = float(particle["x"]) * scale
    y = float(particle["y"]) * scale

    x = ceil(x) - 1
    y = ceil(y) - 1

    # Get how much region to extract in the through-plane direction
    R = float(particle["R"])
    through_region = ceil(R * scale * width_wrt_radius)

    # - On how much in the through (y-dir) to extract
    # - Deal with edge cases
    t_idx_up = (y - through_region
                if y - through_region >= 0 else 0)
    t_idx_down = (y + through_region
                  if y + through_region <= Y_max - 1 else Y_max - 1)

    # Small region to calculate tortuosity from
    extracted_im = micro_im[t_idx_up:t_idx_down, :x, :]
    extracted_im = np.reshape(extracted_im, np.shape(extracted_im)[:-1])
    pores = extracted_im == pore_encoding

    extracted_im = np.zeros(np.shape(extracted_im), dtype=bool)
    extracted_im[pores] = True

    # Simulation along the in-plane (x-direction)
    sim = ps.dns.tortuosity(extracted_im, axis=0)
    tortuosity = sim.tortuosity

    return tortuosity
