import numpy as np
import pandas as pd
#from iminuit import Minuit
#from iminuit.cost import LeastSquares
from .config import FitSettings
#from .utils import FitUtils
#from .plotting import Plotting

class SignalExtractor:
    def __init__(self, df: pd.DataFrame, settings: FitSettings):
        self.df = df.loc[
            (df[settings.pt_column_name] > settings.pTmin) &
            (df[settings.pt_column_name] < settings.pTmax)
        ]
        self.settings = settings

        self.bins = np.linspace(settings.fitMin, settings.fitMax, int((settings.fitMax - settings.fitMin) / settings.bin_width) + 1)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2

        self.data_counts = None
        self.data_errors = None
        self.bin_edges = None

        #self.signal_function = None
        # self.comb_background_function = None
        # self.total_fit_func = None
        #self.chi2_reduced = 0

    def bin_data(self):
        self.data_counts, self.bin_edges = np.histogram(self.df[self.settings.mass_column_name], bins=self.bins)
        self.data_errors = np.sqrt(self.data_counts)

        plt.plot(self.bin_centers, self.data_counts)