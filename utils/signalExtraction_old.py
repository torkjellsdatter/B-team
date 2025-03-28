import pandas as pd
import numpy as np
import scipy.integrate as integrate
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import inspect

from iminuit import Minuit
#from iminuit.cost import Template
from iminuit.cost import LeastSquares

class signalExtraction:
    
    def __init__(self, df, cuts, pTmin, pTmax, fitMin, fitMax, bin_width, savefig_path, mass_column_name="correctedMass", pt_column_name="fpTBcandidate", verbose=False):
        
        self.savefig_path = savefig_path
        
        self.mass_column_name = mass_column_name
        self.pt_column_name = pt_column_name
        
        self.fitMin = fitMin # GeV
        self.fitMax = fitMax # GeV
        self.pTmin = pTmin # GeV
        self.pTmax = pTmax # GeV
        self.bin_width = bin_width # GeV
        
        # initialize dataframe in the right pT range
        self.df = df.loc[(df[self.pt_column_name] > self.pTmin) & (df[self.pt_column_name] < self.pTmax)]
        
        # TODO: make agnositc to cuts
        self.cuts = cuts

        self.number_of_bins = int((self.fitMax-self.fitMin)/self.bin_width)
        self.bins = np.linspace(self.fitMin, self.fitMax, self.number_of_bins + 1)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2
        
        # filled in bin_data
        self.data_counts = None
        self.data_errors = None
        self.bin_edges = None
        self.chi2_reduced = 0
        
        # filled in initialize_background_function
        self.comb_background_function = None
        self.signal_function = None
        self.total_fit_func = None
        self.corrbkg_function = None
        
        self.template_of_all_correlated_background_sources = {}
        
        self.correlated_background_size_initialized = False
        self.signal_fraction = 0
        self.corr_bkg_fraction = 0
        self.n_corr_fixed = 0
    
    def bin_data(self):
        """
        Method to bin data
        """
        
        df_in_pt_bins = self.df
        
        self.data_counts, self.bin_edges = np.histogram(df_in_pt_bins[self.mass_column_name], bins=self.bins)
        self.data_errors = np.sqrt(self.data_counts)
    
    def initialize_background_function(self, func, peak_mask_min=5., peak_mask_max=5.7, plot=False): 
        """
        Method to find initial guesses for the combinatorial background
        It works by blinding the peak, and fitting a optional function to the reminding datapoints
        
        Args: 
            func (method): function of which to fit background
            peak_mask_min (float): default: 5, lower limit of the mask, 
            peak_mask_max (float): default: 5.7, upper limit of the mask, 
        """
        self.comb_background_function = func
        func_params = inspect.signature(self.comb_background_function).parameters
        
        #initialize starting values
        starting_values = {}
        for param in func_params:
                if param != "x":
                    starting_values[param] = 1.
        
        # exclude bins with few counts in the fit
        bin_mask = ~(self.data_counts[:] < 1)
        x_data_without_empty_bins = self.bin_centers[bin_mask]
        y_meas_without_empty_bins = self.data_counts[bin_mask]
        y_err_without_empty_bins = self.data_errors[bin_mask]
        
        # apply the mask on the peak
        peak_mask = ~((x_data_without_empty_bins[:] >= peak_mask_min) & (x_data_without_empty_bins[:] <= peak_mask_max))
        x_data = x_data_without_empty_bins[peak_mask] 
        y_meas = y_meas_without_empty_bins[peak_mask] 
        y_err = y_err_without_empty_bins[peak_mask]
        
        # least square regression (TODO change cost function to Chi2)
        least_squares = LeastSquares(x_data, y_meas, y_err, func)
        minuit = Minuit(least_squares, **starting_values)  # Initial guesses
        minuit.migrad()

        # Set new values to be used in the next fit
        self.background_guess = dict(zip(starting_values.keys(), minuit.values))
        
        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.errorbar(x_data, y_meas, yerr=y_err, fmt='o', color='black', ecolor='black', capsize=0, label='Data')

            x_fit = np.linspace(min(x_data), max(x_data), 100)
            y_fit = self.comb_background_function(x_fit, *minuit.values)
            
            ax.plot(x_fit, y_fit, label="Fit", color='blue')
            plt.savefig(f"{self.savefig_path}/background_function.png")
    
    # TODO: Not used, yet
    def gauss(self, x, N, mu, sigma):
        return N * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    def gauss_integral(self, x, N, mu, sigma):
        return self.gauss(x, N, mu, sigma)
    
    # TODO: Not used, yet
    def combined_func(self, x, params):
        background_params = {k: v for k, v in params.items() if k not in {"N", "sigma", "mu"}}
        return self.comb_background_function(x, *background_params.values()) + gauss(x, params["N"], params["sigma"], params["mu"])
    
    def generate_mc_template(self, mc_df, signal_channels, corr_bkg_channels, kde_window=0.3, plot=False, test=True):
        """
        Generates MC templates for signal and correlated background using histograms and KDE smoothing.

        Parameters:
        - mc_df (pd.DataFrame): The input Monte Carlo dataset.
        - signal_channels (dict): Mapping of signal channel names to corresponding fMcFlag values.
        - corr_bkg_channels (dict): Mapping of correlated background channel names to fMcFlag values.
        - kde_window (float): Bandwidth for Gaussian KDE smoothing.
        - plot (bool): If True, plots the results.
        - test (bool): If True, performs a test comparing summing first vs. smoothing first.

        Returns:
        - mc_template_norm (array): Normalized MC template for the signal.
        - mc_template_bin_edges (array): Bin edges for the signal histogram.
        - corr_bkg_templates_KDE (dict): KDE-smoothed templates for correlated backgrounds.
        """
        mc_df_in_pt_bins = mc_df.loc[(mc_df[self.pt_column_name] > self.pTmin) & (mc_df[self.pt_column_name] < self.pTmax)]
        
        # ---- Process Signal ----
        df_signal = pd.concat([mc_df_in_pt_bins.loc[mc_df_in_pt_bins["fMcFlag"] == signal_channels[sig]] for sig in signal_channels])
        mc_template, mc_template_bin_edges = np.histogram(df_signal[self.mass_column_name], bins=self.bins)
        mc_template_norm = mc_template/ np.sum(mc_template)
        self.signal_function = mc_template_norm
        
        if plot: 
            fig, ax = plt.subplots(figsize=(10, 6)) 
            plt.plot(self.bin_centers, mc_template_norm, linewidth=2, color='tab:blue', label=r"B$^\pm \rightarrow J/\psi + K^\pm$")
        
        # ---- Process Correlated Background ----
        df_corr_bkg_list = []
        corr_bkg_templates_KDE = {}

        for corr_bkg, flag in corr_bkg_channels.items():
            df_corr_bkg = mc_df_in_pt_bins.loc[mc_df_in_pt_bins["fMcFlag"] == flag]
            processed_corrbkg = df_corr_bkg.loc[df_corr_bkg[self.mass_column_name] < 5.2]  # Remove unexplained peak

            df_corr_bkg_list.append(processed_corrbkg)

            # KDE smoothing
            kde = gaussian_kde(processed_corrbkg[self.mass_column_name], bw_method=kde_window)
            kde_corr_bkg_template = kde(self.bin_centers)  # Evaluated KDE
            kde_scaled = kde_corr_bkg_template * len(processed_corrbkg) * self.bin_width  # Scale to match histogram

            corr_bkg_templates_KDE[corr_bkg] = kde_scaled  # Store the unnormalized KDE

        # used for unsmoothed template
        df_all_corr_bkg = pd.concat(df_corr_bkg_list, ignore_index=True)
        corrbkg_template, _ = np.histogram(df_all_corr_bkg[self.mass_column_name], bins=self.bins)
        corrbkg_template_norm = corrbkg_template / np.sum(corrbkg_template)
        
        # calculate signal and correlated background fraction to be used to restrict the fit
        total_number = df_signal.shape[0] + df_all_corr_bkg.shape[0]
        self.signal_fraction = df_signal.shape[0] / total_number
        self.corr_bkg_fraction = df_all_corr_bkg.shape[0] / total_number

        # ---- Calculate scaling factor ----- 
        all_KDE = sum(corr_bkg_templates_KDE.values()) # add the background components to a common KDE
        all_KDE_normalized = all_KDE / np.sum(all_KDE)
        
        for corr_bkg in corr_bkg_templates_KDE:
            scaling_factor = np.sum(corr_bkg_templates_KDE[corr_bkg]) / np.sum(all_KDE) # divide the total entries of one component by the total entries of the sum
            kde_normalized = corr_bkg_templates_KDE[corr_bkg] / np.sum(corr_bkg_templates_KDE[corr_bkg]) # normalize each KDE
            kde_scaled = kde_normalized * scaling_factor # Further reduvce the size by the scaling factor
            self.template_of_all_correlated_background_sources[corr_bkg] = kde_scaled
                
        if plot:
            # plot the normalized total KDE template
            plt.plot(self.bin_centers, all_KDE / np.sum(all_KDE), color='grey', linewidth=2, label="Total Correlated Background")
            
            # Plot all contributions
            total_filled_kde = np.zeros_like(self.bin_centers)
            for corr_bkg in self.template_of_all_correlated_background_sources:
                total_filled_kde += self.template_of_all_correlated_background_sources[corr_bkg]
                plt.fill_between(self.bin_centers, total_filled_kde, total_filled_kde - self.template_of_all_correlated_background_sources[corr_bkg], alpha=0.3, label=corr_bkg)

            plt.ylabel("Normalized counts")
            plt.xlabel(r'$m_{eeK^\pm} - m_{ee} + m_{J\psi, DPG}$ (GeV/$c^2$)')
            
            plt.legend(loc="upper left")
            plt.savefig(f"{self.savefig_path}/MCtemplates.png")
            
        # This test is to check that smoothing all contributions before addin is the same as adding and then smoothing
        if test:
            
            # Do KDE smoothing on the total dataframe
            kde_all = gaussian_kde(df_all_corr_bkg[self.mass_column_name], bw_method=kde_window)
            kde_corr_bkg_template_all = kde_all(self.bin_centers)
            kde_scaled_all = kde_corr_bkg_template_all * len(df_all_corr_bkg) * self.bin_width
            corr_bkg_template_norm_KDE = kde_scaled_all / np.sum(kde_scaled_all)
            
            if plot: 
                fig, ax = plt.subplots(figsize=(14, 10)) 
                plt.plot(self.bin_centers, all_KDE / np.sum(all_KDE), color='grey', linewidth=2, label="Smoothed, then added")
                plt.plot(self.bin_centers, corr_bkg_template_norm_KDE, ls='--', label="Added, then smoothed", linewidth=3, color="Gray")#
                plt.legend()
                plt.savefig(f"{self.savefig_path}/KDEtest.png")
            
        return mc_template_norm, all_KDE_normalized, corrbkg_template_norm, mc_template_bin_edges
    
    def fit(self, func, signal_func, corr_bkg_func=None, corr_bkg_limits=0.1, plot=False):
        """
        Fitting the data with a background function, specified in initialize_background_function and a gaussian
        """
        self.total_fit_func = func
        self.signal_function = signal_func
        self.corrbkg_function = corr_bkg_func
        
        # exclude bins with few counts in the fit
        bin_mask = ~(self.data_counts[:] < 1)
        x_data_without_empty_bins = self.bin_centers[bin_mask]
        y_meas_without_empty_bins = self.data_counts[bin_mask]
        y_err_without_empty_bins = self.data_errors[bin_mask]
        
        # define staring values for the fit
        param_values = self.background_guess # take values from initalization of the background function
        func_params = inspect.signature(func).parameters
        for param in func_params: # use physically motivated parameters for a gaussian, else scale to 1
            if param not in param_values: 
                if param=='x': continue
                elif param=='n_sig': param_values.update({"n_sig":100})
                elif param=='mu': param_values.update({"mu":5.3})
                elif param=='sigma': param_values.update({"sigma":0.1})
                else: param_values.update({param:1.})
        
        # ----- Peform fit ------
        least_squares = LeastSquares(x_data_without_empty_bins, y_meas_without_empty_bins, y_err_without_empty_bins, self.total_fit_func)
        minuit = Minuit(least_squares, **param_values)
        if "sigma" in param_values: minuit.limits["sigma"] = (0.02, 0.5)
        if "n_sig" in param_values: minuit.limits["n_sig"] = (0., None) # scaling must be positive
            
        # First fit: set the limits of n_corr to be positive
        if "n_corrbkg" in param_values:
            if self.n_corr_fixed == 0: minuit.limits["n_corrbkg"] = (0., None)
            else:
                lower_limit_n_corr = self.n_corr_fixed - (corr_bkg_limits*self.n_corr_fixed)
                upper_limit_n_corr = self.n_corr_fixed + (corr_bkg_limits*self.n_corr_fixed)
                #print("n_corr: ", self.n_corr_fixed, lower_limit_n_corr, upper_limit_n_corr)
                minuit.limits["n_corrbkg"] = (lower_limit_n_corr, upper_limit_n_corr)

        minuit.migrad()
        self.fit_params = minuit.values

        # ratio of correlated background to signal (integrate over the whole fit range)
        if self.correlated_background_size_initialized == False:
            I_s, signal_integral_error = integrate.quad(self.gauss_integral, self.fitMin, self.fitMax, args=(1, self.fit_params["mu"], self.fit_params["sigma"]))
            I_c, corrbkg_error = integrate.quad(self.corrbkg_function, self.fitMin, self.fitMax, args=(1))
            f_c = self.corr_bkg_fraction
            
            self.n_corr_fixed = (f_c*self.fit_params["n_sig"]* I_s) / (I_c*(1 - f_c))

            # To avoid the if statement the second time around
            self.correlated_background_size_initialized = True
        
        chi2 = minuit.fval
        dof = len(self.bin_centers) - minuit.nfit
        self.chi2_reduced = chi2 / dof
        
        if plot: 
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.errorbar(x_data_without_empty_bins, y_meas_without_empty_bins, yerr=y_err_without_empty_bins, fmt='o', color='black', ecolor='black', capsize=0, label='Data')
            x_fit = np.linspace(self.fitMin, self.fitMax, 100)
            y_fit = self.total_fit_func(x_fit, *minuit.values)
            ax.plot(x_fit, y_fit, label="Fit", color='blue')
            plt.savefig(f"{self.savefig_path}/fit_initalization.png")
        
        return self.fit_params
    
    def template_lookup(self, x, template):
        """
        Generalized function to look up template values based on binning.

        Parameters:
        - x (array-like): Input values to be binned.
        - template (array-like): Template array to retrieve values from.
        - bins (array-like): Bin edges used for digitization.

        Returns:
        - array-like: Values from the template corresponding to the bins of x.
        """
        bin_index = np.digitize(x, self.bins) - 1
        bin_index = np.clip(bin_index, 0, len(template) - 1)
        return template[bin_index]

    def plot_invariant_mass(self, plot_contributions=False):
        
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.errorbar(self.bin_centers, self.data_counts, yerr=self.data_errors, fmt='o', color='black', ecolor='black', capsize=0, label='Data')
    
        if plot_contributions:
            x_fit = np.linspace(min(self.bin_centers), max(self.bin_centers), 500)
            
            # total fit 
            ax.plot(x_fit, self.total_fit_func(x_fit, *self.fit_params), linewidth=3, color='grey', label='Total fit')

            # ---------- Plotting -------------
            # Vaues needed for the parameterized background
            paramnames_for_background_function = inspect.signature(self.comb_background_function).parameters
            params_for_background_function = {}
            for name in paramnames_for_background_function:
                if name=='x': continue
                params_for_background_function[name] = self.fit_params[name]
            comb_background = self.comb_background_function(x_fit, **params_for_background_function)
            ax.plot(x_fit, comb_background, ls='dashdot', linewidth=3, label="Comb.background", color='tab:red')
                
            # Fit values for signal
            paramnames_for_signal_function = inspect.signature(self.signal_function).parameters
            params_for_signal_function = {}
            for name in paramnames_for_signal_function:
                if name=='x': continue
                params_for_signal_function[name] = self.fit_params[name]
            signal = self.signal_function(x_fit, **params_for_signal_function)
            ax.plot(x_fit, self.fit_params["n_sig"]*signal, linewidth=2, color='tab:blue')
            ax.fill_between(x_fit, self.fit_params["n_sig"]*signal, 0, color='tab:blue', alpha=.3)
            legend_patches = [Patch(color='tab:blue', alpha=0.3, label=r"B$^\pm \rightarrow J/\psi + K^\pm$")]

            # correlated backgroun
            stacked_corr_bkg = comb_background
            for corr_bkg_source, color in zip(self.template_of_all_correlated_background_sources, ["tab:green", "tab:orange", "tab:purple"]):
                template = self.template_of_all_correlated_background_sources[corr_bkg_source]
                scaled_template = self.fit_params["n_corrbkg"] * self.template_lookup(x_fit, template)
                ax.fill_between(
                    x_fit,
                    stacked_corr_bkg,  # Lower bound
                    stacked_corr_bkg + scaled_template,  # Upper bound
                    color=color,
                    alpha=0.3
                )
                stacked_corr_bkg  += scaled_template
                legend_patches.append(Patch(color=color, alpha=0.3, label=corr_bkg_source))
            
            # ---------- Plotting -------------
            # Calculate signal size and significance
            signal_range_min = self.fit_params["mu"] - 3 * self.fit_params["sigma"]
            signal_range_max = self.fit_params["mu"] + 3 * self.fit_params["sigma"]
        
            mask = (self.bin_centers >= signal_range_min) & (self.bin_centers <= signal_range_max)
            signal_plus_background = np.sum(self.data_counts[mask])
            
            bkg_integral, bkg_error = integrate.quad(self.comb_background_function, signal_range_min, signal_range_max, args=tuple(params_for_background_function.values()))
            bkg_size = bkg_integral / self.bin_width
            #corrbkg_integral, corrbkg_error = integrate.quad(self.corrbkg_function, signal_range_min, signal_range_max, limit=100)
            #corrbkg_integral, corrbkg_error = 0,0
            
            #if corrbkg_integral <= 0:
            #    print("Warning: corrbkg_integral is non-positive! Applying safeguard.")
            #    corrbkg_integral = max(corrbkg_integral, 1e-6)  # Set a small positive value

            #scaled_corrbkg_integral = self.fit_params["n_corrbkg"] * corrbkg_integral
            #corrbkg_size = scaled_corrbkg_integral / self.bin_width

            integral, error = integrate.quad(self.gauss_integral, signal_range_min, signal_range_max, args=(self.fit_params["n_sig"], self.fit_params["mu"], self.fit_params["sigma"]))
            signal_size = integral / self.bin_width
            #signal_size = signal_plus_background - bkg_size - corrbkg_size
            #signal_size_error = SB_error / self.bin_width
            signal_size_error = np.sqrt(signal_plus_background)
            significance = signal_size/signal_size_error
            
        else: 
            signal_size = significance = signal_size_error= signal_range_min = signal_range_max = masked_counts= 0
            legend_patches= []

        
        plt.axvline(x=5.28, color='grey', linestyle='--', linewidth=1)
        plt.text(5.28, plt.ylim()[1] * 0.2, r"$m^{B^\pm}_{\text{PDG}}$", color='grey', fontsize=14, rotation=270, verticalalignment='top')
        
        # text on plot
        plottext = (
                f"$B^\\pm \\to J/\\psi(\\to e^+e^-) + K^\\pm$\n"
                #f"LHC22, LHC24am/aj/af/ag/al/ao/an (pp) \n"
                #f"Cuts: $\\tau$ > {self.tauCut} ns, $\\chi^2$ < {self.chi2cut} \n"
                #f"Cuts: {self.cuts} \n"
                f"${self.pTmin} < p_T$ < {self.pTmax} GeV/$c^2$ \n"
                f"$\\chi^2$/ndof = {self.chi2_reduced:.2f}\n"
                #f"Signal size = {signal_size:.0f} $\\pm$ {signal_size_error:.0f} in [{signal_range_min:.2f}, {signal_range_max:.2f}] GeV/$c^2$\n"
                f"Signal size = {signal_size:.0f} $\\pm$ {signal_size_error:.0f}\n"
                f"$S/\\sqrt{{S+B}}$ = {significance:.2f}\n"
        )
    
        # axis
        ax.set_ylabel(fr'Entries per {self.bin_width*1000:.0f} MeV/$c^2$',fontsize=22)
        ax.set_xlabel(r'$m_{eeK^\pm} - m_{ee} + m_{J\psi, DPG}$ (GeV/$c^2$)',fontsize=22)
        ax.set_xlim([self.fitMin, self.fitMax])
        ax.set_ylim([0, self.data_counts.max()*1.2])
        ax.legend(handles=ax.get_legend_handles_labels()[0] + legend_patches, loc="upper right", fontsize=20)

        fig.tight_layout()
        
        return fig, ax, plottext, signal_size, signal_size_error, self.chi2_reduced
