from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple
import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


import git
repo_path = git.Repo('.', search_parent_directories=True).working_tree_dir

def _load_json(fp: Path) -> Dict[str, Any]:
    fp = Path(fp)
    with fp.open("r", encoding="utf-8") as f:
        return json.load(f)


@dataclass(slots=True)
class VQDPlotter:
    data_path: Path
    potentials: List[str]
    cutoff: int
    converged_only: bool = True


    # ---------- loading ----------
    def _load(self, data_path: str, potential: str, cutoff: int, shots: int) -> Tuple[np.ndarray, np.ndarray]:
        d_path = os.path.join(
            repo_path, data_path, str(shots), potential, f"{potential}_{cutoff}.json"
        )

        data = _load_json(d_path)

        exact_eigenvalues = np.asarray(data.get("exact_eigenvalues", []), dtype=float)

        results = np.asarray(data.get("results", []), dtype=float)
        success = np.asarray(data.get("success", []), dtype=bool)

        if results.size:
            if self.converged_only:
                converged_idx = np.where(np.all(success, axis=1))[0] # only include runs where all 3 energy levels converged
                if converged_idx.size > 0:
                    energies = results[converged_idx]
                else:
                    energies = np.array([np.nan], dtype=float)
            else:
                energies = results
        else:
            energies = np.array([], dtype=float)

        energies = np.sort(energies) #sort energy levels within individual run

        return energies, exact_eigenvalues
    

    def plot_vqd(
        self,
        *,
        shots=None,
        num_energy_levels=3
    ):
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10), sharex=True)

        marker_styles = ['o', '^', 's']
        color_map = ['tab:blue', 'tab:orange', 'tab:green']

        for plot_idx, potential in enumerate(self.potentials):
            energies, exact_eigenvalues = self._load(self.data_path, potential, self.cutoff, shots)
            ax = axes[plot_idx]

            for i in range(num_energy_levels):
                for run_idx, energy in enumerate(energies):
                    # plotting vqd data
                    ax.scatter(run_idx, energy[i], s=10, marker=marker_styles[i], color=color_map[i], alpha=0.5)

                # Add exact eigenvalue lines
                ax.axhline(exact_eigenvalues[i], color=color_map[i], linestyle='--', linewidth=0.5)

            # Title and axis labels
            ax.text(0.5, 0.95, f"{potential}",
                transform=ax.transAxes,
                fontsize='large',
                ha='center', va='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

            # Axis labels
            if plot_idx == 2:
                ax.set_xlabel("VQD Run")
            ax.set_ylabel("Energy")

            ymin, ymax = ax.get_ylim()
            buffer = 0.1 * (ymax - ymin)
            bottom = 0 if ymin > 0 else ymin
            ax.set_ylim(bottom=bottom, top=ymax + buffer)

            # exact energy legend
            exact_handles = [
                Line2D([], [], color=color_map[i], linestyle='--', linewidth=0.5,
                    label=f"$E_{{{i}}}^{{\\text{{ex}}}} = {exact_eigenvalues[i]:.3f}$")
                for i in range(3)
            ]
            legend1 = ax.legend(handles=exact_handles, fontsize='small',
                                markerscale=0.7, loc='upper right')
            ax.add_artist(legend1)

            # energy level indicators

            energy_level_indicators = [
                Line2D([], [], marker=marker_styles[i],
                    color=color_map[i],
                    linestyle='None', markersize=6,
                    label=f"$E_{{{i}}}$")
                for i in range(num_energy_levels)
            ]

            if plot_idx == 0:
                ax.legend(handles=energy_level_indicators, fontsize='small',
                        markerscale=0.7, loc='upper left')

        return fig, axes
        
