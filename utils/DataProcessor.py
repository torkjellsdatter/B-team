import numpy as np
import matplotlib.pyplot as plt
import uproot
import pandas as pd
import os
import json

class DataProcessor:
    """
    Reads, pre-process and tests derived data sets from both tableReader and dqEfficiency

    Attributes:
        jpsi_mass (float): PDG mass of J/ψ (GeV/c²).
        Bplus_mass (float): PDG mass of B⁺ (GeV/c²).
    """

    jpsi_mass = 3.096916  # GeV/c²
    Bplus_mass = 5.27934  # GeV/c²

    def __init__(self, directory, datasets, electron_cut_name=None, kaon_cut_name = None, filename="AO2D.root", ARname="AnalysisResults.root", verbose=False):
        """
        Initializes the DataProcessor.

        Args:
            directory (str): Full path to the directory containing datasets.
            datasets (list): the datasets must be stored as folders within the directory that contains a merged AO2D
            filename (str, optional): HL output is always AO2D.root - but for local runs the name is specified in the writer config
            verbose (bool, optional): If True, prints debug messages.
        """
        self.directory = directory
        self.datasets = datasets
        self.electron_cut_name = electron_cut_name
        self.kaon_cut_name = kaon_cut_name
        self.filename = filename
        self.ARname = ARname
        self.verbose = verbose
        self.read_ME = False

    def read_data(self, read_ME=False):
        """
        Reads derived datasets (i.e. datasets that contain O2dqbmesonsa or O2dqbmesons) from a local directory, using uproot. 

        Args:
            compare_to_analysis_results (bool, optional): If True, compares with AnalysisResults.root.

        Returns:
            pd.DataFrame: Concatenated DataFrame containing all datasets.
        """
        self.read_ME = read_ME
        
        dataframes = []
        ME_dataframes = []
        
        list_of_runs_from_txt_file = []

        for dataset in self.datasets:
            print(f"******* DATASET: {dataset}")
            datapath = f"{self.directory}{dataset}"
            jobs_filepath = os.path.join(datapath, "jobs2.txt")
            runs_filepath = os.path.join(datapath, "runs.txt")

            # read job numbers 
            if os.path.exists(jobs_filepath):
                with open(f"{datapath}/jobs2.txt", 'r') as f:
                    jobs = [line.strip() for line in f if line.strip()]
            else:
                print(f"⚠️ WARNING 1 File not found: {jobs_filepath}")
                jobs = []

            # read run numbers 
            if os.path.exists(runs_filepath):
                with open(f"{datapath}/runs.txt", 'r') as f:
                    runs = f.readlines()
                runs = runs[0].split(",")
                run_numbers = [num.strip() for line in runs for num in line.split(",")]
                list_of_runs_from_txt_file.extend(run_numbers)
            else:
                print(f"⚠️ WARNING 2 File not found: {runs_filepath}")
                runs = []
        
            if len(runs) != len(jobs): print("⚠️ WARNING: Number of runs does not match number of jobs!")

            #Read each hy_directory
            for i, job in enumerate(jobs): 
                hy_job = job.split("/")[-1].strip()
                hy_path = os.path.join(datapath, hy_job)
                run = runs[i] if i < len(runs) else None

                if os.path.exists(hy_path):
                    
                    if self.verbose: print(f"======== RUN:{runs[i], }PROCESSING FILE {path}")
                        
                    path = f"{datapath}/{hy_job}"
                    
                    dataframe, MEdataframe = self.read_AO2D(path, dataset, run, hy_job, filename="AO2D.root")

                    dataframes.extend(dataframe)
                    ME_dataframes.extend(MEdataframe)

        data_df = pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()

        ME_df = pd.concat(ME_dataframes, ignore_index=True) if ME_dataframes else pd.DataFrame()

        return data_df, ME_df
    
    def read_AO2D(self, path, dataset, run, hy_job, filename="AO2D.root"):
        """
        Method reads a single AO2D, looped over i read_data
        """
        dataframes = []
        ME_dataframes = []
        
        path = f"{path}/{filename}"
        
        dataset_info = {"Dataset": dataset, "Run": run, "Job": hy_job}
        
        try:
                with uproot.open(path) as root_file:
                    for j in range(1, len(root_file.keys())):
                        key_name = root_file.keys()[j]

                         # Only trees with the name of the derived dataset (specified in writerconfig)
                        if any(substring in key_name for substring in ['O2dqbmesonsa', 'O2dqbmesons']):
                            if self.verbose: print(f"Keys in the ROOT file: {key_name}")
                            df = root_file[key_name].arrays(library="pd")
                            df["Dataset"] = dataset
                            df["Run"] = run
                            df["Job"] = hy_job

                            if df.empty: # some runs have empty dataframes
                                print(f"⚠️ Dataframe is empty. {run}, {path}/{hy_job} is not being read properly")
                                dataframes.append(pd.DataFrame([dataset_info]))

                            dataframes.append(df)

                        # Read ME histogram and make a dataframe of these histograms
                        if self.read_ME: 
                            ME_histogram = self.read_ME_histograms(f"{dataset}/{hy_job}")
                            ME_dict = {"Dataset": dataset, "Run": run, "Job": hy_job, "ME histogram": ME_histogram}
                            
                            ME_dataframes.append(pd.DataFrame([ME_dict]))  
        except FileNotFoundError:
            print(f"⚠️ WARNING: 3 File not found {path}")
            dataframes.append(pd.DataFrame([dataset_info]))
            ME_dataframes.append(pd.DataFrame([dataset_info]))
            #continue
            
        return dataframes, ME_dataframes
    
    def pre_processing(self, df):
        """
        Renames underflow and overflow to nan values and recomputes a corrected mass if 'fdeltamassBcandidate' is a column in the dataframe

        Args: df to be processes
        Returns: df after processing
        """
        # Undefined values in O2Physics
        undef = [-999, 999, -9999, 9999]

        if self.verbose:
            for col in df.columns:
                count_undef = df[col].isin(undef).sum()
                print(f"'{col}': {(count_undef / len(df)) * 100:.2f}% entries in underflow or overflow")

        # Replace undefined values with NaN
        df.replace(undef, np.nan, inplace=True)
        
        # The bitmap columns are filled with 0 if undefined and converted to integrals (format needed for convert_bitmap())
        cut_columns = ["fIsJpsiFromBSelected", "fIsBarrelSelected"]
        for cut in cut_columns: 
            if cut in df.columns:
                df[cut] = df[cut].replace([np.inf, -np.inf], np.nan).fillna(0).astype(int)
                test = df.loc[df[cut]==0].shape
                if test[0] != 0: print(f"⚠️ WARNING: {test[0]} candidates pass no cuts - {cut} =0 ")
    
        # make a new column with the deltaMass + PDGJpsimass
        # This accounts for that the column name is written differently in tableReader and dqEfficiency
        possible_columns = ['fdeltamassBcandidate', 'fdeltaMassBcandidate']
        mass_col = next((col for col in possible_columns if col in df.columns), None)

        if mass_col is None:
            print("⚠️ WARNING: Neither 'fdeltamassBcandidate' nor 'fdeltaMassBcandidate' found in DataFrame")
        else:
            df["correctedMass"] = df[mass_col] + self.jpsi_mass

        return df
    
    def read_cuts_from_configfile(self, config_file_name = "config.json"):
        """
        Reads a config file (stored in workdir/dataset) and extracts:
        - `config_cuts`: Collected cuts
        - `cfg_electron_cuts`: Same-event pairing cuts (electron cuts)
        - `cfg_kaon_cuts`: Dilepton-track cuts (kaon cuts)

        Also ensures that all datasets have the same cuts and prints a warning if differences are found.
        """
        
        # Needed for test that the configfile of all datasets are the same
        reference_cuts = None
        reference_electron_cuts = None
        reference_kaon_cuts = None
        
        for i, dataset in enumerate(self.datasets): 
            config_dir  = f"{self.directory}{dataset}/{config_file_name}"
            try:
                with open(config_dir, 'r') as file:
                    config_data = json.load(file)
                    # get track cuts 
                    cfg_barrel_track_cuts_json = config_data.get("analysis-track-selection", {}).get("cfgBarrelTrackCutsJSON", "")
                    cfg_barrel_track_cuts_not_json = config_data.get("analysis-track-selection", {}).get("cfgTrackCuts", "")
                    
                    # get pair and associated track cuts
                    cfg_electron_cuts_json = config_data.get("analysis-same-event-pairing", {}).get("cfgTrackCuts", "")
                    cfg_kaon_cuts_json = config_data.get("analysis-dilepton-track", {}).get("cfgTrackCuts", "")
                    self.electron_cut_name = cfg_electron_cuts_json.split(",")
                    self.kaon_cut_name = cfg_kaon_cuts_json.split(",")

                    config_cuts = {}
                    if cfg_barrel_track_cuts_json: # convert total cuts to dictionary
                        cfg_barrel_track_cuts = json.loads(cfg_barrel_track_cuts_json)
                        config_cuts = {key: value for key, value in cfg_barrel_track_cuts.items()}
                   
                    if cfg_barrel_track_cuts_not_json:
                        cfg_barrel_track_cuts_not_json.split(',')
                        config_cuts = {key for key in cfg_barrel_track_cuts_not_json.split(',')}
                        
                    if reference_cuts == None:  # Store the first dataset's cuts as reference
                        reference_cuts = config_cuts
                        reference_electron_cuts = self.electron_cut_name
                        reference_kaon_cuts = self.kaon_cut_name 
                    else: # Compare with reference and print warning if different
                        if (config_cuts != reference_cuts or 
                            self.electron_cut_name != reference_electron_cuts or 
                            self.kaon_cut_name  != reference_kaon_cuts):
                            print(f"⚠️ WARNING: Cuts in dataset '{dataset}' differ: there are different configurations in this workfolder")


            except FileNotFoundError:
                print(f"⚠️ WARNING: {config_dir} file not found. Continuing execution...")
                
        return config_cuts, self.electron_cut_name, self.kaon_cut_name 
    
    def decode_cuts(self, bit_map, cut_list, offset=0):
        """
        Decodes a bitmap into a dictionary mapping each cut to a boolean value.

        Parameters:
        - bit_map (int): Bitmap integer representing active cuts.
        - cut_list (list): List of cut names corresponding to bits.
        - offset (int): Bit offset (if bit mapping starts from a specific index).

        Returns:
        - dict: Mapping of cut names to boolean values.
        """
        return {cut_list[i - offset]: (bit_map & (1 << i)) != 0 for i in range(offset, len(cut_list) + offset)}

    def convert_bitmap(self, df, column_name, cut_names, offset=0):
        """
        Converts a bitmap column in a DataFrame into separate boolean columns.

        Parameters:
        - df (pd.DataFrame): The DataFrame containing the bitmap column.
        - column_name (str): The name of the column with bitmaps to be decoded.
        - cut_names (list): List of cut names to associate with bitmap bits.
        - offset (int, optional): Offset for bit indexing. Defaults to 0.

        Returns:
        - pd.DataFrame: The original DataFrame with additional columns for decoded bits.
        """
        if column_name not in df.columns:
            raise KeyError(f"Column '{column_name}' not found in DataFrame.")

        decoded_bits = df[column_name].apply(lambda x: self.decode_cuts(x, cut_names, offset))
        df_decoded_bits = decoded_bits.apply(pd.Series)  # Directly convert to DataFrame

        return df.join(df_decoded_bits)

    def compare_to_AnalysisResult(self, df, datapath = None, verbose = False):
        """
        Comparing the length of he read derived dataset to the number of entries in the AnalysisResults.root file
        NB: The name of the file to compare with MUST be AnalysisResults.root, and it MUST be located in the same directory as the AO2D.root file

        Args: 
            df: dataframe, i.e. the derived dataset to be compared to the AR
        """
        AR_entries = {}
        for dataset in self.datasets: 
            if verbose: print(f"************* {dataset} ***************")
            
            # Get number of entries from AnalysisResults file
            if datapath: datapath = datapath
            datapath = f"{self.directory}{dataset}/{self.ARname}"
                
            try:
                with uproot.open(datapath) as root_file:
                    hist_key = "analysis-dilepton-track/output;1"
                    if hist_key not in root_file:
                        print(f"⚠️ WARNING: Key '{hist_key}' not found in {datapath}")
                    for i, item in enumerate(root_file[hist_key]):
                        name = item.member("fName").replace("DileptonTrack_", "")
                        if len(root_file[hist_key][i]) <= 3: # ME histograms have less than 3 entries 
                            AR_entries[name] = root_file[hist_key][i][2].member("fEntries") 
                        else: 
                            AR_entries[name] = root_file[hist_key][i][6].member("fEntries") # use the entries in the mass histogram, idx=6

            except FileNotFoundError:
                print(f"⚠️ WARNING: {datapath} file not found")
                continue

            # Get number of entries per cut in each dataframe
            dataset_df = df.loc[df["Dataset"]==dataset]
            
            if "fIsJpsiFromBSelected" in dataset_df.columns: 
                for i, elcutname in enumerate(self.electron_cut_name):
                    for j, kaoncutname in enumerate(self.kaon_cut_name):

                        bit_elcut = 1 << i
                        bit_kacut = 1 << (j + len(self.electron_cut_name)) 

                        # Use a vectorized NumPy approach
                        n = np.sum(((dataset_df["fIsJpsiFromBSelected"].to_numpy(dtype=np.int64) & bit_elcut) != 0) & ((dataset_df["fIsBarrelSelected"].to_numpy(dtype=np.int64) & bit_kacut) != 0))

                        cutcombo = f"{elcutname}_{kaoncutname}"

                        if verbose: print(f"{cutcombo} has {n} entries")

                        # Do comparison: 
                        if AR_entries.get(cutcombo) != None:
                            if AR_entries[cutcombo] != n: 
                                print(f"⚠️ WARNING: Dataset '{dataset}' has mismatched entries!")
                                print(f"  - Derived dataset: {n} entries")
                                print(f"  - AnalysisResults: {AR_entries[cutcombo]} entries")
                                print(f"  - Difference: {(n / AR_entries[cutcombo]) * 100:.2f}% more events in derived ")
                        else: print(f"⚠️ WARNING: {cutcombo} histograms are not found in the AnalysisResults file")
            else: # This is for old datasets without columns for cut selection
                print(f"⚠️ WARNING: Derived dataset does not contain feature for cut selection")
                if dataset_df.shape[0] != AR_entries[next(iter(AR_entries))]:
                    print(f"⚠️ WARNING: Dataset '{dataset}' has mismatched entries!")
        datapath = ""
                
    def read_ME_histograms(self, dir):
        """
        This is not being maintained as ME did not work for my analysis
        Reads of the ME histogram ['analysis-dilepton-track/output;1'][DilptonTrackME_]["Mass_Pt"], 
        and returms that histogram
        """

        ARpath  = f"{self.directory}{dir}/{self.ARname}"
        ME_histogram = None

        ME_folder_name = f"DileptonTrackME_{self.electron_cut_name}"

        try:
            with uproot.open(ARpath) as AR_file:
                triplet_dir  = 'analysis-dilepton-track/output;1'

                if triplet_dir not in AR_file:
                    print(f"⚠️ WARNING: Key '{triplet_dir}' not found in {AR_file}")
                    
                triplet_folder = AR_file[triplet_dir]
                for folder in triplet_folder: 
                    if folder.member("fName") == f"DileptonTrackME_{self.electron_cut_name}":
                        for hist in folder: 
                            if hist.member("fName") == f"Mass_Pt":
                                ME_histogram = hist
        except FileNotFoundError:
            print(f"⚠️ WARNING: {ARpath} file not found")

        if ME_histogram is None: print(f"⚠️ WARNING: ME histogram is empty")
        
        return ME_histogram


    def plot_datasets(self, df, bins):
        """
        Plots the dataset distributions to check for duplications.

        Args:
            df (pd.DataFrame): DataFrame containing the dataset.
            bins (int or list): Number of bins or bin edges.
        """
        if "Dataset" not in df.columns:
            print("⚠️ ERROR: 'Dataset' column missing in DataFrame. Cannot plot.")
            return

        for run_period, group in df.groupby("Dataset"):
            counts, bin_edges = np.histogram(group["correctedMass"], bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            n = group["correctedMass"].shape[0]
            legend = f"{run_period}: {n} entries"

            plt.plot(bin_centers, counts, label=legend, marker=".")

        plt.xlabel(r"$m_{K^\pm ee} - m_{ee} + m_{\text{J}/\psi}^{\text{PDG}}$ (GeV/$c^2$)")
        plt.legend()
        plt.show()
