import pandas as pd
import numpy as np
import json, os
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

import git
repo_path = git.Repo('.', search_parent_directories=True).working_tree_dir


def _load_json(fp: Path) -> Dict[str, Any]:
    fp = Path(fp)
    with fp.open("r", encoding="utf-8") as f:
        return json.load(f)


@dataclass(slots=True)
class VQDTables:
    """
    Helper for plotting VQD table metrics.
    """
    data_path: Path
    potentials: List[str]
    cutoffs: list[int]
    converged_only: bool = True


    # ---------- loading ----------
    def _load(self, data_path: str, potential: str, cutoff: int) -> Tuple[np.ndarray, np.ndarray, float]:
        
        data = _load_json(data_path)

        num_VQD = data['num_VQD']

        exact_eigenvalues = np.asarray(data.get("exact_eigenvalues", []), dtype=float)

        results = np.asarray(data.get("results", []), dtype=float)
        success = np.asarray(data.get("success", []), dtype=bool)

        converged_idx = np.where(np.all(success, axis=1))[0]
        converged_runs = len(converged_idx) 

        if results.size:
            if self.converged_only:
                if converged_idx.size > 0: # only include runs where all 3 energy levels converged
                    energies = results[converged_idx]
                else:
                    energies = np.array([np.nan], dtype=float)
            else:
                energies = results
        else:
            energies = np.array([], dtype=float)

        energies = np.sort(energies) #sort energy levels within individual run

        num_evals = data['num_evaluations']
        sum_evals = [sum(x) for x in num_evals]
        mean_evals = np.mean(sum_evals)

        num_iters = data['num_iters']
        sum_iters = [sum(x) for x in num_iters]
        mean_iters = np.mean(sum_iters)

        e0 = []
        e1 = []
        e2 = []

        for e in np.sort(energies):
            e0.append(e[0])
            e1.append(e[1])
            e2.append(e[2])

        median_ratio = abs((np.median(e2) - np.median(e1)) / (np.median(e2) - np.median(e0)))

        e0_exact = exact_eigenvalues[0]
        e1_exact = exact_eigenvalues[1]
        e2_exact = exact_eigenvalues[2]
        exact_ratio = abs((e2_exact - e1_exact) / (e2_exact - e0_exact))

        row = {
            "potential": potential,
            "cutoff": cutoff,
            'Converged Runs': f"{converged_runs}/{str(num_VQD)}",
            r'$N_{\text{iters}}$': int(mean_iters),
            r'$N_{\text{evals}}$': int(mean_evals),
            "e0": np.median(e0),
            "e1": np.median(e1),
            "e2": np.median(e2),
            "median_ratio": median_ratio,
            "e0_exact": e0_exact,
            "e1_exact": e1_exact,
            "e2_exact": e2_exact,
            "exact_ratio": exact_ratio
            }

        return row
    
    def create_ratio_table(
        self,
        *,
        shots=None
    ):
        all_data = []

        for potential in self.potentials:
            for cutoff in self.cutoffs:

                path = os.path.join(self.data_path, str(shots), potential, f"{potential}_{cutoff}.json")
                row = self._load(path, potential, cutoff)
                all_data.append(row)

        return pd.DataFrame(all_data)


    def create_beta_comp_table(
        self,
        potential,
        cutoff
    ):

        all_data = []
        folder_path = os.path.join(self.data_path, potential)
   
        folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

        for folder in folders:

            path = os.path.join(folder_path, folder, f"{potential}_{cutoff}.json")  
            data = self._load(path, potential, cutoff)
            row = {
                "beta":float(folder),
                'Converged Runs': data['Converged Runs'],
                r'$N_{\text{iters}}$': data[r'$N_{\text{iters}}$'],
                r'$R_{\text{exact}}$': data['exact_ratio'],
                r'$R_{\text{med}}$': data['median_ratio'],
                r"$\Delta R$": np.abs(data['median_ratio'] - data['exact_ratio'])
                }
            all_data.append(row)

        df = pd.DataFrame(all_data).sort_values("beta")

        return df