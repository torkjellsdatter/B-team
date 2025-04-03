import numpy as np
import pandas as pd
from iminuit import Minuit
from iminuit.cost import LeastSquares
import inspect

import matplotlib.pyplot as plt 
from matplotlib.patches import Patch

import scipy.integrate as integrate

from utils.config import FitSettings

class SignalExtractor:
    def __init__(self, df: pd.DataFrame, settings: FitSettings, combkg_function='exponential'):
        
        self.settings = settings
            
        self.df = df.loc[
            (df[self.settings.pt_column_name] > self.settings.pTmin) &
            (df[self.settings.pt_column_name] < self.settings.pTmax)
        ]

        self.number_of_bins = int((self.settings.fitMax-self.settings.fitMin)/self.settings.bin_width)
        self.bins = np.linspace(self.settings.fitMin, self.settings.fitMax, self.number_of_bins + 1)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2

        self.data_counts = None
        self.data_errors = None
        self.bin_edges = None
        
        self.combkg_function = combkg_function
        
        self.signal_size = None
        self.fit_params = None

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
    
    def poly2_ratio(self, x, a, b, c, d, e, f):
        return self.poly2(x, a, b, c)/self.poly2(x, d, e, f)
    
    def corrbkg_template_func(self, x):
        total_corrbkg_template_counts = sum(self.corrbkg_template_counts.values())
        bin_index = np.digitize(x, self.template_bins) - 1 
        bin_index = np.clip(bin_index, 0, len(total_corrbkg_template_counts) - 1) 
        return total_corrbkg_template_counts[bin_index]
    
    def template_lookup(self, x, template):
        bin_index = np.digitize(x, self.bins) - 1
        bin_index = np.clip(bin_index, 0, len(template) - 1)
        return template[bin_index]

    def total_fit_func_exp(self, x, n_corrbkg, n_sig, mu, sigma, n_combkg, a):
        
        return (n_corrbkg * self.corrbkg_template_func(x)) + (n_sig * self.gauss(x, mu, sigma)) + (n_combkg * np.exp(a * x))

    def total_fit_func_poly2(self, x, n_corrbkg, n_sig, mu, sigma, a, b, c):
        
        return (n_corrbkg * self.corrbkg_template_func(x)) + (n_sig * self.gauss(x, mu, sigma)) + self.poly2(x, a, b, c)

    def total_fit_func_poly2ratio(self, x, n_corrbkg, n_sig, mu, sigma, a, b, c, d, e, f):
        
        return (n_corrbkg * self.corrbkg_template_func(x)) + (n_sig * self.gauss(x, mu, sigma)) + self.poly2_ratio(x, a, b, c, d, e, f)

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

        # ----- Define staring values of the fit ------
        if self.fit_params == None: # first time, set the starting guess to 1 and physical values for the gaussian
            param_values = {}
            func_params = inspect.signature(self.total_fit_func).parameters
            for f in func_params: 
                if f=='x': continue
                elif f=='n_sig': param_values[f] = 100
                elif f=='mu': param_values[f] = 5.3
                elif f=='sigma': param_values[f] = 0.01
                else: param_values[f] = 1
        else: param_values = self.fit_params.to_dict()  #or use the results from the last fit as starting values on the next
        #print("Starting values for the fit: ", param_values)
            
        # ----- Initialize fit ------
        least_squares = LeastSquares(x_data, y_data, y_data_err, self.total_fit_func)
        minuit = Minuit(least_squares, **param_values)
        
        # ----- Set limits ------
        if "sigma" in param_values: minuit.limits["sigma"] = (0.02, 0.5)
        if "n_sig" in param_values: minuit.limits["n_sig"] = (0., None) # scaling must be positive
        
        # fix the ratio of the correlated background to the signal size
        if self.signal_size is not None:
            self.fix_n_corr_to_signal_size()
            lower_limit_n_corr = self.n_corr_scaled_with_signal_size - (self.settings.allowed_fluctuation_for_corrbkg*self.n_corr_scaled_with_signal_size)
            upper_limit_n_corr = self.n_corr_scaled_with_signal_size + (self.settings.allowed_fluctuation_for_corrbkg*self.n_corr_scaled_with_signal_size)
            if "n_corrbkg" in param_values: minuit.limits["n_corrbkg"] = (lower_limit_n_corr, upper_limit_n_corr ) 
        else:
            if "n_corrbkg" in param_values: minuit.limits["n_corrbkg"] = (0., None) 

        # ----- Peform fit ------
        minuit.migrad()
        self.fit_params = minuit.values
        
        self.chi2_reduced = minuit.fval / (len(self.bin_centers) - minuit.nfit)

    def _integrate_signal(self):
        mu = self.fit_params["mu"]
        sigma = self.fit_params["sigma"]
        
        self.signal_range_min = mu - 3 * sigma
        self.signal_range_max = mu + 3 * sigma
        
        integral, error = integrate.quad(self.gauss, self.signal_range_min, self.signal_range_max, args=(mu, sigma))
        self.signal_size = (self.fit_params["n_sig"] * integral) / self.settings.bin_width

    def _calculate_signal_plus_background(self):
        
        mask = (self.bin_centers >= self.signal_range_min) & (self.bin_centers <= self.signal_range_max)
        self.signal_plus_background = np.sum(self.data_counts[mask])
    
    def calculate_signal_size(self): 
        
        self._integrate_signal()
        self._calculate_signal_plus_background()

        self.signal_size_error = np.sqrt(self.signal_plus_background)
        self.significance = self.signal_size / self.signal_size_error
        
    def fix_n_corr_to_signal_size(self):

        f_c = self.fraction_of_correlated_background
        
        # integrate the signal (must be done again as it is integrating over the whole fit_range)
        I_s, signal_integral_error = integrate.quad(self.gauss, self.settings.fitMin, self.settings.fitMax, args=(self.fit_params["mu"], self.fit_params["sigma"]))
        
        I_c, corrbkg_integral_error = integrate.quad(self.corrbkg_template_func, self.settings.fitMin, self.settings.fitMax)
        
        self.n_corr_scaled_with_signal_size = (f_c*self.fit_params["n_sig"]* I_s) / (I_c*(1 - f_c))
    
    def plot_invariant_mass(self, legend_loc="upper right"):
        
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.errorbar(self.bin_centers, self.data_counts, yerr=self.data_errors, fmt='o', color='black', ecolor='black', capsize=0, label='Data')

        x_fit = np.linspace(min(self.bin_centers), max(self.bin_centers), 500)
            
        # ----------- total fit ----------- 
        ax.plot(x_fit, self.total_fit_func(x_fit, *self.fit_params), linewidth=3, color=self.settings.color_palette["10"][3], label='Total fit')
        
        # ----------- signal (assuming gaussian signal) ----------- 
        sig_color = self.settings.color_palette["10"][0]
        signal = self.gauss(x_fit, self.fit_params["mu"], self.fit_params["sigma"])
        ax.plot(x_fit, self.fit_params["n_sig"]*signal, linewidth=2, color=sig_color)
        ax.fill_between(x_fit, self.fit_params["n_sig"]*signal, 0, color=sig_color, alpha=.5)
        legend_patches = [Patch(color=sig_color, alpha=0.5, label=r"B$^\pm \rightarrow J/\psi + K^\pm$")]
        
        # ----------- combinatorial background ----------- 
        if self.combkg_function == "exponential": combbkg_plot = self.fit_params["n_combkg"] * np.exp(self.fit_params["a"] * x_fit)
        elif self.combkg_function == "second_degree_polynomial": combbkg_plot = self.poly2(x_fit, self.fit_params["a"], self.fit_params["b"], self.fit_params["c"])
        elif self.combkg_function == "ratio_second_degree_polynomials": combbkg_plot = self.poly2(x_fit, self.fit_params["a"], self.fit_params["b"], self.fit_params["c"])/self.poly2(x_fit, self.fit_params["d"], self.fit_params["e"], self.fit_params["f"])
        else:
            raise ValueError(f"Unknown background function: {self.combkg_function}")
        ax.plot(x_fit, combbkg_plot, ls='dashdot', linewidth=3, label="Comb. background", color=self.settings.color_palette["10"][2])

        # ----------- correlated background ----------- 
        stacked_corr_bkg = combbkg_plot
        for corrbkg_source, color in zip(self.corrbkg_template_counts.keys(), [self.settings.color_palette["10"][1], self.settings.color_palette["10"][5], self.settings.color_palette["10"][7]]): 
            scaled_template = self.fit_params["n_corrbkg"] * self.template_lookup(x_fit, self.corrbkg_template_counts[corrbkg_source])
            ax.fill_between(
                    x_fit,
                    stacked_corr_bkg,  # Lower bound
                    stacked_corr_bkg + scaled_template,  # Upper bound
                    color=color,
                    alpha=0.6
                )
            stacked_corr_bkg  += scaled_template
            legend_patches.append(Patch(color=color, alpha=0.6, label=corrbkg_source))
            
        # ----------- line and text ----------- 
        #plt.axvline(x=self.settings.pdg_mass_b_meson, color=self.settings.color_palette["10"][3], linestyle='--', linewidth=1)
        #plt.text(5.28, plt.ylim()[1] * 0.2, r"$m^{B^\pm}_{\text{PDG}}$", color=settings.color_palette["10"][3], fontsize=14, rotation=270, verticalalignment='top')
        
        # axis
        ax.set_ylabel(fr"Entries per {self.settings.bin_width*1000:.0f} MeV/$\mathit{{c}}^2$",fontsize=self.settings.fontsize)
        ax.set_xlabel(self.settings.xaxis_label,fontsize=self.settings.fontsize)
        ax.set_xlim([self.settings.fitMin, self.settings.fitMax])
        ax.set_ylim([0, self.data_counts.max()*1.2])
        ax.legend(handles=ax.get_legend_handles_labels()[0] + legend_patches, loc=legend_loc, fontsize=self.settings.fontsize)

        fig.tight_layout()
        
        return fig, ax
    
    def generate_plot_text(self):
        plottext = (
            #f"ALICE performance\n"
            #r"pp, $\sqrt{\mathit{s}}$ = 13.6 TeV"
            f"\n$B^\\pm \\to J/\\psi(\\to e^+e^-) + K^\\pm$\n"
            #f"LHC22, LHC24am/aj/af/ag/al/ao/an (pp) \n"
            #f"Cuts: $\\tau$ > {self.tauCut} ns, $\\chi^2$ < {self.chi2cut} \n"
            #f"Cuts: {self.cuts} \n"
            f"$|\\mathit{{y}}| <$ 0.9, ${self.settings.pTmin} < \\mathit{{p}}_T <$ {self.settings.pTmax} GeV/$\\mathit{{c}}$ \n"
            f"$\\chi^2$/ndof = {self.chi2_reduced:.2f}\n"
            #f"Signal size = {signal_size:.0f} $\\pm$ {signal_size_error:.0f} in [{signal_range_min:.2f}, {signal_range_max:.2f}] GeV/$c^2$\n"
            f"Signal size = {self.signal_size:.0f} $\\pm$ {self.signal_size_error:.0f}\n"
            f"$S/\\sqrt{{S+B}}$ = {self.significance:.2f}\n"
        )
        return plottext