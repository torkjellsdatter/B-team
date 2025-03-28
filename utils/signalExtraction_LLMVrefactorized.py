# Directory structure suggestion:
# signal_extraction/
# ├── __init__.py
# ├── extractor.py         <- Core logic and class
# ├── templates.py         <- MC template generation
# ├── plotting.py          <- All plotting logic
# ├── config.py            <- FitSettings and constants
# ├── utils.py             <- Small utilities (e.g. gaussian, lookup, etc.)
# ├── main.py              <- Example script to use the module
# └── test_extractor.py    <- Basic unit test for extractor

# === File: config.py ===
import json
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class FitSettings:
    pTmin: float
    pTmax: float
    fitMin: float
    fitMax: float
    bin_width: float
    mass_column_name: str = "correctedMass"
    pt_column_name: str = "fpTBcandidate"
    savefig_path: str = "plots"
    verbose: bool = False

    @staticmethod
    def from_json(json_path: str) -> "FitSettings":
        with open(json_path, "r") as f:
            config_dict = json.load(f)
        return FitSettings(**config_dict)

# === File: utils.py ===
import numpy as np

class FitUtils:
    @staticmethod
    def gauss(x, N, mu, sigma):
        return N * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    @staticmethod
    def gauss_integral(x, N, mu, sigma):
        return FitUtils.gauss(x, N, mu, sigma)

    @staticmethod
    def template_lookup(x, template, bins):
        bin_index = np.digitize(x, bins) - 1
        bin_index = np.clip(bin_index, 0, len(template) - 1)
        return template[bin_index]

# === File: plotting.py ===
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

class Plotting:
    @staticmethod
    def plot_background_fit(x_data, y_meas, y_err, fit_func, fit_params, save_path):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(x_data, y_meas, yerr=y_err, fmt='o', color='black', label='Data')
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = fit_func(x_fit, *fit_params)
        ax.plot(x_fit, y_fit, label="Fit", color='blue')
        plt.legend()
        plt.savefig(save_path)
        plt.close(fig)

# === File: templates.py ===
# Placeholder: Logic related to generate_mc_template and KDE processing will go here

# === File: extractor.py ===
import numpy as np
import pandas as pd
from iminuit import Minuit
from iminuit.cost import LeastSquares
from .config import FitSettings
from .utils import FitUtils
from .plotting import Plotting

class SignalExtractor:
    def __init__(self, df: pd.DataFrame, cuts: str, settings: FitSettings):
        self.df = df.loc[
            (df[settings.pt_column_name] > settings.pTmin) &
            (df[settings.pt_column_name] < settings.pTmax)
        ]
        self.cuts = cuts
        self.settings = settings

        self.bins = np.linspace(settings.fitMin, settings.fitMax, int((settings.fitMax - settings.fitMin) / settings.bin_width) + 1)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2

        self.data_counts = None
        self.data_errors = None
        self.bin_edges = None

        self.signal_function = None
        self.comb_background_function = None
        self.total_fit_func = None
        self.chi2_reduced = 0
        self.fit_result = None

    def bin_data(self):
        self.data_counts, self.bin_edges = np.histogram(self.df[self.settings.mass_column_name], bins=self.bins)
        self.data_errors = np.sqrt(self.data_counts)

    def _get_non_empty_bins(self):
        bin_mask = self.data_counts >= 1
        return self.bin_centers[bin_mask], self.data_counts[bin_mask], self.data_errors[bin_mask]

    def initialize_background_function(self, func, peak_mask=(5.0, 5.7), plot=False):
        self.comb_background_function = func
        x_data, y_meas, y_err = self._get_non_empty_bins()

        # Blind the peak region
        peak_mask_arr = ~((x_data >= peak_mask[0]) & (x_data <= peak_mask[1]))
        x_blind, y_blind, yerr_blind = x_data[peak_mask_arr], y_meas[peak_mask_arr], y_err[peak_mask_arr]

        start_vals = {p: 1. for p in func.__code__.co_varnames if p != "x"}

        minuit = Minuit(LeastSquares(x_blind, y_blind, yerr_blind, func), **start_vals)
        minuit.migrad()
        self.background_guess = dict(minuit.values)

        if plot:
            Plotting.plot_background_fit(x_blind, y_blind, yerr_blind, func, minuit.values, f"{self.settings.savefig_path}/background_fit.png")

    def fit(self, fit_func, initial_params=None, param_limits=None):
        x_data, y_meas, y_err = self._get_non_empty_bins()
        self.total_fit_func = fit_func

        if initial_params is None:
            initial_params = {p: 1.0 for p in fit_func.__code__.co_varnames if p != "x"}

        least_squares = LeastSquares(x_data, y_meas, y_err, fit_func)
        minuit = Minuit(least_squares, **initial_params)

        if param_limits:
            for key, lim in param_limits.items():
                minuit.limits[key] = lim

        minuit.migrad()
        self.fit_result = minuit
        self.chi2_reduced = minuit.fval / (len(x_data) - minuit.nfit)

        return dict(minuit.values)

    def template_fit_func(self, x, n_sig, a, b):
        if self.signal_function is None:
            raise ValueError("Signal template (self.signal_function) is not set!")

        signal = n_sig * FitUtils.template_lookup(x, self.signal_function, self.bins)
        background = a + b * x
        return signal + background
