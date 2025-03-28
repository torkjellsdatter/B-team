import json
from dataclasses import dataclass, field
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
    xaxis_label: str = r"$m_{eeK^\pm} - m_{ee} + m_{J\psi, DPG}$ (GeV/$c^2$)"

    signal_channels: Dict[str, int] = field(default_factory=lambda: {
        r"$B^\pm \rightarrow J/\psi + K^\pm$ (exclusive)": 1
    })

    background_channels: Dict[str, int] = field(default_factory=lambda: {
        r"$B^\pm \rightarrow J/\psi + K^\pm + X$": 2,
        r"$B^\pm \rightarrow J/\psi + X \rightarrow ee + K^\pm$": 4,
        r"$B^0 \rightarrow J/\psi + X \rightarrow ee + K^\pm$": 8
    })