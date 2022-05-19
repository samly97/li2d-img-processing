import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def save_color_img(
    ml_solmap: np.ndarray,
    comsol_solmap: np.ndarray,
    path: str,
    timestep: float,
):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass

    # Don't mutate original data
    ml_solmap = np.copy(ml_solmap)
    comsol_solmap = np.copy(comsol_solmap)

    # Masks to get rid of 0s
    mask_ml = ml_solmap == 0
    mask_comsol = comsol_solmap == 0

    ml_solmap[mask_ml] = np.nan
    comsol_solmap[mask_comsol] = np.nan

    def get_min_max(ml_solmap, comsol_solmap):
        ml_min = np.min(ml_solmap[~mask_ml])
        comsol_min = np.min(comsol_solmap[~mask_comsol])

        ml_max = np.max(ml_solmap[~mask_ml])
        comsol_max = np.max(comsol_solmap[~mask_comsol])

        _min = ml_min if ml_min < comsol_min else comsol_min
        _max = ml_max if ml_max > comsol_max else comsol_max

        return (_min, _max)

    def plot(im, source, timestep):
        plt.axis('off')
        plt.imshow(im, cmap="plasma")
        plt.colorbar()
        plt.clim(_min, _max)
        plt.savefig(
            os.path.join(path, "%s_%s.png" % (source, str(timestep)))
        )
        plt.close('all')

    (_min, _max) = get_min_max(ml_solmap, comsol_solmap)

    plot(ml_solmap, "ml", timestep)
    plot(comsol_solmap, "comsol", timestep)


def plot_sol_profile(
    ml_solmap: np.ndarray,
    comsol_solmap: np.ndarray,
    path: str,
    timestep: float,
    scale: int = 5,
):
    def get_sol_profile(solmap: np.ndarray):
        # Get coordinates where the particles are lithiated
        coords = np.where(solmap != 0)
        y_particle, x_particle, _ = coords

        # Length of electrode in pixels
        x_max = np.max(x_particle)

        y_particle = list(y_particle)
        x_particle = list(x_particle)

        # (x,y) pairing of (length, SoL)
        x_sol_arr = [(x, solmap[y, x, :])
                     for (y, x) in zip(y_particle, x_particle)]

        # Store list of SoL as fn of x
        sol_fn_of_L: Dict[float, List[float]] = {}
        for tup in x_sol_arr:
            x, sol = tup
            sol = list(sol)

            temp_arr = sol_fn_of_L.get(x, [])
            temp_arr.extend(sol)
            sol_fn_of_L[x] = temp_arr

        mean_sol_fn_of_L: Dict[float, float] = {}
        # Find the average SoL at location x in electrode
        for tup in x_sol_arr:
            x, _ = tup
            mean_sol_fn_of_L[x] = float(np.mean(np.array(sol_fn_of_L.get(x))))

        # Unpack SoL Distribution
        x_sol_arr = list(zip(*x_sol_arr))
        x, sol_dist = x_sol_arr

        # Get the mean SoL
        x_length = [x_coord for x_coord in range(0, x_max)]
        sol_length_mean = [mean_sol_fn_of_L.get(x) for x in x_length]

        return (
            (x, sol_dist),
            (x_length, sol_length_mean),
        )

    def plot(source, solmap):
        (
            (x, sol_dist),
            (x_length, sol_length_mean),
        ) = get_sol_profile(solmap)

        plt.plot(np.array(x) / scale, sol_dist, 'lightblue')
        plt.plot(np.array(x_length) / scale, sol_length_mean, 'bo')

        plt.axis([0, np.max(np.array(x)) / scale, 0, 1])
        plt.xlabel("Distance from the Separator [$\mu$m]")
        plt.ylabel("SoL [1]")
        plt.title("%s - SOL Distribution at t=%.2fs" % (source, timestep))
        plt.savefig(os.path.join(path, "%s_sol_dist_%s.png" %
                    (source, timestep)))
        plt.close('all')

    plot("ML", ml_solmap)
    plot("COMSOL", comsol_solmap)
