from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.config import FitSettings

class TemplateMaker: 
    def __init__(self, mc_df: pd.DataFrame, settings: FitSettings):
        
        self.settings = settings
        
        self.mc_df = mc_df.loc[
            (mc_df[self.settings.pt_column_name] > self.settings.pTmin) &
            (mc_df[self.settings.pt_column_name] < self.settings.pTmax)
        ]

        self.number_of_bins = int((self.settings.fitMax-self.settings.fitMin)/self.settings.bin_width)
        self.bins = np.linspace(self.settings.fitMin, self.settings.fitMax, self.number_of_bins + 1)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2
    
    def _generate_signal_template(self):
        
        df_signal = pd.concat([self.mc_df.loc[self.mc_df["fMcFlag"] == self.settings.signal_channels[sig]] for sig in self.settings.signal_channels])
        self.signal_template, _ = np.histogram(df_signal[self.settings.mass_column_name], bins=self.bins)
        self.signal_template_norm  = self.signal_template/np.sum(self.signal_template)
        
        self.mean_signal_mass = df_signal[self.settings.mass_column_name].mean()
        self.signal_size = df_signal.shape[0]
    
    def _generate_corrbkg_template(self):
        
        self.corrbkg_templates = {}
        self.kde_corrbkg_templates = {}
        
        self.corrbkg_size = 0 # Initialize the size of correlated background
        
        for corrbkg, flag in self.settings.background_channels.items():
            df_corrbkg = self.mc_df.loc[(self.mc_df["fMcFlag"] == flag)&(self.mc_df[self.settings.mass_column_name] < 5.2)]

            # templates without KDE
            corrbkg_template, _ = np.histogram(df_corrbkg[self.settings.mass_column_name], bins=self.bins)
            self.corrbkg_templates[corrbkg] = corrbkg_template
            
            # templates with KDE
            kde = gaussian_kde(df_corrbkg[self.settings.mass_column_name], bw_method=self.settings.kde_window)
            kde_scaled = kde(self.bin_centers) * len(df_corrbkg) * self.settings.bin_width  # Scale to match histogram
            self.kde_corrbkg_templates[corrbkg] = kde_scaled 
            
            self.corrbkg_size += df_corrbkg.shape[0] 
    
    def generate_mc_templates(self):

        self._generate_signal_template()
        self._generate_corrbkg_template()
        
        # Calculate the ratio of correlated background to signal
        if self.signal_size > 0: 
            self.fraction_of_corrbkg = self.corrbkg_size / (self.signal_size + self.corrbkg_size)
        else:
            self.fraction_of_corrbkg = 0  # Set to 0 if signal size is zero

        print(f"Ratio of Correlated Background to Signal: {self.fraction_of_corrbkg:.2f}")
        
        return self.corrbkg_templates, self.kde_corrbkg_templates, self.bins, self.mean_signal_mass, self.fraction_of_corrbkg

    def plot_mc_signals(self): 
        
        mc_df_binned, _ = np.histogram(self.mc_df[self.settings.mass_column_name], bins=self.bins)
        
        # Plot histograms
        plt.hist(self.bins[:-1], bins=self.bins, weights=mc_df_binned, alpha=0.4, label="All MC data")  # Filled histogram
        plt.hist(self.bins[:-1], bins=self.bins, weights=self.signal_template, alpha=0.4, label=r"$B^\pm \rightarrow J/\psi + K^\pm$ (exclusive)")  # Filled histogram
         
        for i in self.corrbkg_templates.keys(): 
            plt.hist(self.bins[:-1], bins=self.bins, weights=self.corrbkg_templates[i], histtype="step", linestyle="dotted", linewidth=1.5, label=i)

        plt.xlabel(self.settings.xaxis_label, fontsize=24)
        plt.yscale("log")
        plt.legend(loc="upper right")
        plt.show()
        #plt.("MCsignals.png")
        