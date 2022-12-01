from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import tensorflow as tf

from utils.io import load_json
from utils import typings

from postprocessing.ml import Microstructure_to_ETL
from postprocessing.ml import electrode_sol_map_from_predictions

import os
import time

from tqdm import tqdm


class Create_Predicted_Solmaps():

    def __init__(
        self,
        experiments: Dict[int, Dict[str, List[Tuple[float, float]]]],
        params,
    ):

        # Expect experiments to have this format:
        # experiments = {
        #     1: {
        #         "0.25": [
        #             (0.25, 0),
        #             ...
        #         ],
        #         "0.5": ...
        self.experiments = experiments

        # Load the `JSON` file with the microstructure information
        self.micro_json: List[typings.Microstructure_Data] = load_json(
            params["metadata_filename"]
        )

        self.model = tf.keras.models.load_model(
            params["model_name"],
            compile=False,
        )

        # Filepaths
        self.micro_fname: str = params["micro_fname"]
        self.solmap_fpath: str = params["solmap_fpath"]

        # Create SoLmaps for these microstructures and at these C-rates
        self.micro_indices: List[int] = params["micro_indices"]
        self.L_arr: List[int] = params["L_arr"]
        self.c_rates: List[str] = params["c_rates"]

        # Pore phase encoding for `microstructure` NumPy arrays
        self.pore_encoding = params["data_specs"]["pore_encoding"]
        self.padding_encoding = params["data_specs"]["padding_encoding"]
        self.scale = params["data_specs"]["scale"]

        #############################################################
        # Important normalizations / cell / TensorFlow / parameters #
        #############################################################
        self.L = params["meta_norm"]["L"]
        self.h_cell = params["meta_norm"]["h_cell"]
        self.R_max = params["meta_norm"]["R_max"]
        self.c_rate_norm = params["meta_norm"]["c_rate_norm"]
        self.time_norm = params["meta_norm"]["time_norm"]
        self.zoom_norm = params["meta_norm"]["zoom_norm"]

        self.width_wrt_radius = params["img_settings"]["width_wrt_radius"]
        # What size the images are stored as
        self.img_size = params["img_settings"]["img_size"]
        # What type of zooming to perform, i.e. 0 is nearest neighbour
        self.order_zoom = params["img_settings"]["order_zoom"]

        # What the Neural Network expects for the size of the input and target
        # images
        self.tf_img_size = params["tensorflow"]["tf_img_size"]
        self.batch_size = params["tensorflow"]["batch_size"]

        ##################################################################
        # Create the dataloaders for the microstructures and experiments #
        ##################################################################
        self.dataloaders = self._create_dataloaders()

    def _create_dataloaders(
        self
    ) -> Dict[int, Dict[str, tf.data.Dataset]]:
        dataloaders: Dict[int, Dict[str, tf.data.Dataset]] = {}

        for micro_num in self.micro_indices:
            circle_data = self.micro_json[micro_num - 1]["circles"]
            micro_data = (
                self.micro_fname % micro_num,
                self.L_arr[micro_num - 1], self.h_cell)

            input_imgs = self._get_micro_loader(circle_data, micro_data)

            # Cache (edt, mask) and (target) datasets for
            # `Microstructure_to_ETL` loader
            cache = None

            for discharge in self.c_rates:
                start_time = time.time()

                studied_timesteps = self.experiments[micro_num][discharge]
                img_datasets, cache = self._get_dataset_loaders(
                    input_imgs, studied_timesteps, cache=cache,
                )

                print("It took %s seconds to generate all " % (
                    time.time() - start_time) +
                    "dataset loaders for the specified timesteps")

                micro_hash = dataloaders.get(micro_num, {})
                micro_hash[discharge] = img_datasets

                dataloaders[micro_num] = micro_hash

        return dataloaders

    def _get_micro_loader(self, circle_data, micro_data):
        norm_metadata = {
            "L": self.L,
            "h_cell": self.h_cell,
            "R_max": self.R_max,
            "zoom_norm": self.zoom_norm,
            "c_rate_norm": self.c_rate_norm,
            "time_norm": self.time_norm,
        }
        user_params = (self.width_wrt_radius, self.scale, self.img_size)

        micro_loader = Microstructure_to_ETL(
            self.tf_img_size,
            self.batch_size,
            norm_metadata,
            circle_data,
            micro_data,
            self.pore_encoding,
            self.padding_encoding,
            user_params,
        )
        return micro_loader

    def run_for_all_experiments(self):
        datastore = {}

        for micro_num in self.micro_indices:
            _ = self.compare_solmap_workflow(micro_num, datastore)

        return datastore

    def compare_solmap_workflow(self,  micro_num, datastore={}):
        fem_solmaps = self.load_fem_solmaps(micro_num)
        pred_solmaps = self.create_solmaps_for_microstructure(
            micro_num, self.dataloaders[micro_num],
        )

        for discharge in self.c_rates:
            studied_timesteps = self.experiments[micro_num][discharge]

            self._compare_pred_fem_solmaps(
                datastore,
                micro_num,
                pred_solmaps[discharge],
                fem_solmaps[discharge],
                studied_timesteps,
            )

        return datastore

    def load_fem_solmaps(self, micro_num) -> Dict[str, List[np.ndarray]]:
        fem_solmaps = {}

        for discharge in self.c_rates:
            studied_timesteps = self.experiments[micro_num][discharge]

            solmap_for_c_rate = []

            for _, (c_rate, timestep) in enumerate(studied_timesteps):
                path = os.path.join(self.solmap_fpath % micro_num)
                fname = Create_Predicted_Solmaps.format_filename(
                    path, str(c_rate), timestep,
                )
                solmap = np.load(fname)
                solmap_for_c_rate.append(solmap)

            fem_solmaps[discharge] = solmap_for_c_rate

        return fem_solmaps

    @ staticmethod
    def _compare_pred_fem_solmaps(
        datastore,
        micro_num,
        predicted_solmaps,
        fem_solmaps,
        studied_timesteps
    ):
        for idx, (c_rate, timestep) in enumerate(studied_timesteps):
            rmse = Create_Predicted_Solmaps.compare_solmap_rmse(
                predicted_solmaps[idx],
                fem_solmaps[idx],
            )

            # Store data
            micro_hash = datastore.get(micro_num, {})
            rmse_data = micro_hash.get((c_rate, timestep), 5000)
            rmse_data = rmse

            micro_hash[(c_rate, timestep)] = rmse_data
            datastore[micro_num] = micro_hash

    def create_solmaps_for_microstructure(
            self, micro_num, img_datasets) -> Dict[str, Dict[int, np.ndarray]]:
        # Store predicted solmaps here
        ret_pred_solmaps = {}

        # Create a microstructure mask for getting the predicted `solmap`
        micro_mask = np.load(self.micro_fname % micro_num)
        micro_mask = micro_mask < self.pore_encoding

        for discharge in self.c_rates:
            studied_timesteps = self.experiments[micro_num][discharge]

            ##############################################
            # Use ML Model to Predict on Dataset Loaders #
            ##############################################
            start_time = time.time()

            predicted_imgs = Create_Predicted_Solmaps._predict_imgs(
                self.model, img_datasets[discharge], studied_timesteps)

            print("It took %s for the ML network to predict on all " % (
                time.time() - start_time) + "specified timesteps")

            ###################################################
            # "Patch" Images from Machine Learning Prediction #
            ###################################################
            start_time = time.time()

            predicted_colormaps = self._prediction_to_solmap(
                img_datasets[discharge],
                predicted_imgs,
                micro_mask,
                studied_timesteps,
                self.L_arr[micro_num - 1],
            )

            ret_pred_solmaps[discharge] = predicted_colormaps

            print("It took %s to reconstruct colormaps from Machine " % (
                time.time() - start_time) + "Learning predictions")

        return ret_pred_solmaps

    def _prediction_to_solmap(
        self,
        img_datasets: Dict[int, tf.data.Dataset],
        predicted_imgs: Dict[int, tf.types.experimental.TensorLike],
        micro_mask: np.ndarray,
        studied_timesteps: List[Tuple[float, float]],
        L_electrode: int,
    ):
        predicted_colormaps = {}

        norms = (self.L, self.h_cell, self.R_max, self.zoom_norm, 0, 0)

        for idx, _ in enumerate(studied_timesteps):
            solmap, stats = electrode_sol_map_from_predictions(
                img_datasets[idx],
                predicted_imgs[idx],
                micro_mask,
                L_electrode,
                norms,
                self.batch_size,
                scale=self.scale,
            )
            predicted_colormaps[idx] = solmap

            print(stats)

            del predicted_imgs[idx]
            del img_datasets[idx]

        return predicted_colormaps

    @ staticmethod
    def _predict_imgs(
        model,
        img_datasets: Dict[int, tf.data.Dataset],
        studied_timesteps: List[Tuple[float, float]],
    ) -> Dict[int, tf.types.experimental.TensorLike]:
        predicted_imgs = {}

        for idx, _ in enumerate(studied_timesteps):
            predictions = model.predict(img_datasets[idx])
            predicted_imgs[idx] = predictions

        return predicted_imgs

    @ staticmethod
    def _get_dataset_loaders(
        etl_loader,
        studied_timesteps,
        cache=None
    ) -> Tuple[
        Dict[int, tf.data.Dataset],
        Dict[str, tf.data.Dataset],
    ]:
        img_datasets = {}

        for idx, (c_rate, timestep) in tqdm(enumerate(studied_timesteps)):
            dataloader, cache = etl_loader.get_loader(
                c_rate,
                float(timestep),
                cache,
            )
            img_datasets[idx] = dataloader

        return img_datasets, cache

    @ staticmethod
    def compare_solmap_rmse(
        predicted_solmap: np.ndarray,
        fem_solmap: np.ndarray,
    ) -> np.ndarray:

        err_im = tf.cast(predicted_solmap - fem_solmap, tf.float64)

        # Mask out `err_im` where pixels exist so these errors are not
        # normalized by a larger number, which represents 0s.
        err_im = tf.boolean_mask(err_im, fem_solmap > 0.0)

        err_im = tf.square(err_im)
        err_im = tf.sqrt(tf.reduce_mean(err_im))
        err_im = err_im.numpy()

        return err_im

    @ staticmethod
    def format_filename(
        path: str,
        c_rate: str,
        time: float,
    ):
        fname = "c%s_t%s.npy" % (c_rate, time)
        filepath = os.path.join(path, fname)
        return filepath


class Plot_Solmap():

    @staticmethod
    def plot_and_save_imgs(
        predicted_solmap: np.ndarray,
        fem_solmap: np.ndarray,
        save_path: str,
        timestep: float,
        cs_min: int = 300,
        cs_max: int = 48900,
    ) -> None:
        # Don't mutate original data
        predicted_solmap = np.copy(predicted_solmap)
        fem_solmap = np.copy(fem_solmap)

        predicted_solmap = Plot_Solmap._rescale_solmap(
            predicted_solmap, cs_min, cs_max,
        )
        fem_solmap = Plot_Solmap._rescale_solmap(
            fem_solmap, cs_min, cs_max,
        )

        Plot_Solmap._plot_solmap(predicted_solmap, save_path, "ML", timestep)
        Plot_Solmap._plot_solmap(fem_solmap, save_path, "FEM", timestep)
        Plot_Solmap._plot_solmap_err(
            predicted_solmap, fem_solmap, save_path, timestep,
        )

    @staticmethod
    def _plot_solmap(
        solmap: np.ndarray,
        save_path: str,
        source: str,
        timestep: float,
    ) -> None:
        plt.axis('off')
        plt.imshow(solmap, norm=colors.LogNorm(vmin=1e-3, vmax=1))
        clb = plt.colorbar()
        clb.ax.set_title('SoL')
        plt.savefig(
            os.path.join(save_path, "%s_%s.png" % (source, str(timestep)))
        )
        plt.close("all")

    @staticmethod
    def _plot_solmap_err(
        predicted_solmap: np.ndarray,
        fem_solmap: np.ndarray,
        save_path: str,
        timestep: float,
    ) -> None:
        err_im = np.abs(predicted_solmap - fem_solmap)

        plt.axis('off')
        plt.imshow(err_im, norm=colors.LogNorm(vmin=1e-3, vmax=1))
        clb = plt.colorbar()
        clb.ax.set_title('SoL')
        plt.savefig(
            os.path.join(save_path, "err_%s.png" % str(timestep))
        )
        plt.close("all")

    @staticmethod
    def _rescale_solmap(
        solmap: np.ndarray,
        cs_min: int,
        cs_max: int,
    ) -> np.ndarray:

        mask_not_sol = solmap == 0
        solmap[mask_not_sol] = np.nan

        # Rescale based on maximum and minimum concentrations
        solmap[~mask_not_sol] = (solmap[~mask_not_sol] * cs_max - cs_min) \
            / (cs_max - cs_min)

        return solmap


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
