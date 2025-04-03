# utils/config.py
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class FitSettings:
    pdg_mass_b_meson:float = 5.27934
    pdg_mass_jpsi: float = 3.0969
    pTmin: float = 0
    pTmax: float = 100
    fitMin: float = 0
    fitMax: float = 100
    bin_width: float = 0.04
    kde_window: float = 0.2
    allowed_fluctuation_for_corrbkg: float = 0.1
    mass_column_name: str = "correctedMass"
    pt_column_name: str = "fpTBcandidate"
    savefig_path: str = "plots"
    verbose: bool = False
    fontsize: float = 28
    xaxis_label: str = r"$\mathit{m}_{eeK^\pm} - \mathit{m}_{ee} + \mathit{m}_{J/\psi, PDG}$ (GeV/$\mathit{c}^2)$"
    signal_channels: Dict[str, int] = field(default_factory=lambda: {
        r"$B^\pm \rightarrow J/\psi + K^\pm$ (exclusive)": 1
    })
    background_channels: Dict[str, int] = field(default_factory=lambda: {
        r"$B^\pm \rightarrow J/\psi + K^\pm + X$": 2,
        r"$B^\pm \rightarrow J/\psi + X \rightarrow ee + K^\pm$": 4,
        r"$B^0 \rightarrow J/\psi + K^\pm + X$": 8
    })
    color_palette: Dict[str, list] = field(default_factory=lambda: {
        "6": ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"],
        "8": ["#1845fb", "#ff5e02", "#c91f16", "#c849a9", "#adad7d", "#86c8dd", "#578dff", "#656364"],
        "10": ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
    })
