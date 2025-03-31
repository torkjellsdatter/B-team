import numpy as np
import pandas as pd
from utils.config import FitSettings

from iminuit import Minuit
from iminuit.cost import LeastSquares
import inspect

import matplotlib.pyplot as plt 
from matplotlib.patches import Patch

class SignalExtractor:
    def __init__(self, df: pd.DataFrame, settings: FitSettings, combkg_function='exponential'):
        self.df = df.loc[
            (df[settings.pt_column_name] > settings.pTmin) &
            (df[settings.pt_column_name] < settings.pTmax)
        ]
        self.settings = settings

        self.number_of_bins = int((settings.fitMax-settings.fitMin)/settings.bin_width)
        self.bins = np.linspace(settings.fitMin, settings.fitMax, self.number_of_bins + 1)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2

        self.data_counts = None
        self.data_errors = None
        self.bin_edges = None
        
        self.combkg_function = combkg_function

    def bin_data(self):
        self.data_counts, self.bin_edges = np.histogram(self.df[self.settings.mass_column_name], bins=self.bins)
        self.data_errors = np.sqrt(self.data_counts)
        
    def _get_non_empty_bins(self):
        bin_mask = self.data_counts >= 1
        return self.bin_centers[bin_mask], self.data_counts[bin_mask], self.data_errors[bin_mask]
    
    def gauss(self, x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    def poly2(self, x, a, b, c):
        return a + b*x + c*x*x
    
    def corrbkg_template_func(self, x):

        bin_index = np.digitize(x, self.template_bins) - 1 
        bin_index = np.clip(bin_index, 0, len(self.corrbkg_template_counts) - 1) 
        return self.corrbkg_template_counts[bin_index]

    def total_fit_func_exp(self, x, n_corrbkg, n_sig, mu, sigma, n_combkg, a):
        
        return (n_corrbkg * self.corrbkg_template_func(x)) + (n_sig * self.gauss(x, mu, sigma)) + (n_combkg * np.exp(a * x))

    def total_fit_func_poly2(self, x, n_corrbkg, n_sig, mu, sigma, a, b, c):
        
        return (n_corrbkg * self.corrbkg_template_func(x)) + (n_sig * self.gauss(x, mu, sigma)) + self.poly2(x, a, b, c)

    def total_fit_func_poly2ratio(self, x, n_corrbkg, n_sig, mu, sigma, a, b, c, d, e, f):
        
        return (n_corrbkg * self.corrbkg_template_func(x)) + (n_sig * self.gauss(x, mu, sigma)) + (self.poly2(x, a, b, c)/self.poly2(x, d, e, f))

    def fit(self): 
    
        #exclude bins with few counts in the fit
        x_data, y_data, y_data_err = self._get_non_empty_bins()
        
        # define total fit function based on the combinatorial background defined in the initialization
        if self.combkg_function == 'exponential':
            self.total_fit_func = self.total_fit_func_exp
        elif self.combkg_function == 'second_degree_polynomial':
            self.total_fit_func = self.total_fit_func_poly2
        elif self.combkg_function == 'ratio_second_degree_polynomials':
            self.total_fit_func = self.total_fit_func_poly2ratio
        else:
            raise ValueError(f"Unknown background function: {self.combkg_function}")

        # define staring values for the fit
        param_values = {}
        func_params = inspect.signature(self.total_fit_func).parameters
        for f in func_params: 
            if f=='x': continue
            elif f=='n_sig': param_values[f] = 100
            elif f=='mu': param_values[f] = 5.3
            elif f=='sigma': param_values[f] = 0.01
            else: param_values[f] = 1
        print("Starting values for the fit: ", param_values)

        # ----- Peform fit ------
        least_squares = LeastSquares(x_data, y_data, y_data_err, self.total_fit_func)
        minuit = Minuit(least_squares, **param_values)
        if "sigma" in param_values: minuit.limits["sigma"] = (0.02, 0.5)
        if "n_sig" in param_values: minuit.limits["n_sig"] = (0., None) # scaling must be positive
        if "n_corrbkg" in param_values: minuit.limits["n_sig"] = (0., None) 
        
        minuit.migrad()
        self.fit_params = minuit.values

    def plot_invariant_mass(self, plot_contributions=False):
        
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.errorbar(self.bin_centers, self.data_counts, yerr=self.data_errors, fmt='o', color='black', ecolor='black', capsize=0, label='Data')

        x_fit = np.linspace(min(self.bin_centers), max(self.bin_centers), 500)
            
        # ----------- total fit ----------- 
        ax.plot(x_fit, self.total_fit_func(x_fit, *self.fit_params), linewidth=3, color='grey', label='Total fit')
        
        # ----------- signal (assuming gaussian signal) ----------- 
        signal = self.gauss(x_fit, self.fit_params["mu"], self.fit_params["sigma"])
        ax.plot(x_fit, self.fit_params["n_sig"]*signal, linewidth=2, color='tab:blue')
        ax.fill_between(x_fit, self.fit_params["n_sig"]*signal, 0, color='tab:blue', alpha=.3)
        legend_patches = [Patch(color='tab:blue', alpha=0.3, label=r"B$^\pm \rightarrow J/\psi + K^\pm$")]
        
        # ----------- combinatorial background ----------- 
        
        # ----------- correlated background ----------- 
        ax.plot(x_fit, self.fit_params["n_corrbkg"] * self.corrbkg_template_func(x_fit))

        # axis
        ax.set_ylabel(fr'Entries per {settings.bin_width*1000:.0f} MeV/$c^2$',fontsize=22)
        ax.set_xlabel(r'$m_{eeK^\pm} - m_{ee} + m_{J/\psi, DPG}$ (GeV/$c^2$)',fontsize=22)
        ax.set_xlim([settings.fitMin, settings.fitMax])
        ax.set_ylim([0, self.data_counts.max()*1.2])
        ax.legend(handles=ax.get_legend_handles_labels()[0] + legend_patches, loc="upper right", fontsize=20)

        fig.tight_layout()