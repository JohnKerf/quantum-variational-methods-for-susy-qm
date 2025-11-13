from susy_qm.reporting.vqe_plotter import BoxPlotter, VQEPlotter
import matplotlib.pyplot as plt
import os

import git
repo_path = git.Repo('.', search_parent_directories=True).working_tree_dir

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

cutoffs = [2,4,8,16]
potentials = ["QHO", "AHO","DW"]
shots_list = [None,10000]

data_paths = [
    ("L-BFGS-B", r"paper_results_data\Figs5&6\L-BFGS-B"),
    ("COBYLA", r"paper_results_data\Figs5&6\COBYLA"),
    ("COBYQA", r"paper_results_data\Figs5&6\COBYQA"),
    ("Differential Evolution", r"paper_results_data\Figs5&6\Differential Evolution")
]

vqe_plotter = VQEPlotter(data_paths, potentials, cutoffs)

vqe_plotter.plot_delta_e_vs_cutoff_line(shots=None, metric='median', scale="symlog", linthresh=1e-1, sharey=True)
plt.savefig(os.path.join(repo_path,r"Figures","Fig5_1"))
vqe_plotter.plot_evals_vs_cutoff_box(shots=None, show_legend=False, show_title=False)
plt.savefig(os.path.join(repo_path,r"Figures","Fig5_2"))

vqe_plotter.plot_delta_e_vs_cutoff_line(shots=10000, metric='median', scale="symlog", linthresh=1e-1, sharey=True)
plt.savefig(os.path.join(repo_path,r"Figures","Fig6_1"))
vqe_plotter.plot_evals_vs_cutoff_box(shots=10000, show_legend=False, show_title=False)
plt.savefig(os.path.join(repo_path,r"Figures","Fig6_2"))




cutoffs = [2,4,8,16,32,64]
potentials = ["QHO", "AHO","DW"]
shots_list = [None,10000]

data_paths=[
        ("Full", r"paper_results_data\Figs9-12\Full"),
        ("Truncated", r"paper_results_data\Figs9-12\Truncated"),
        ("Real Amplitudes", r"paper_results_data\Figs9-12\Real Amplitudes"),
    ]

vqe_plotter = VQEPlotter(data_paths, potentials, cutoffs)

vqe_plotter.plot_delta_e_vs_cutoff_line(shots=None, metric='median', scale="symlog", linthresh=1e-1, sharey=True)
plt.savefig(os.path.join(repo_path,r"Figures","Fig9_1"))
vqe_plotter.plot_evals_vs_cutoff_box(shots=None, show_legend=False, show_title=False)
plt.savefig(os.path.join(repo_path,r"Figures","Fig9_2"))

vqe_plotter.plot_delta_e_vs_cutoff_line(shots=10000, metric='median', scale="symlog", linthresh=1e-1, sharey=True)
plt.savefig(os.path.join(repo_path,r"Figures","Fig10_1"))
vqe_plotter.plot_evals_vs_cutoff_box(shots=10000, show_legend=False, show_title=False)
plt.savefig(os.path.join(repo_path,r"Figures","Fig10_2"))




box_plotter = BoxPlotter(potentials, cutoffs, shots_list)

box_plotter.plot_energy_boxplot(data_paths=data_paths, showfliers=True, show_legend=True, shots=None)
plt.savefig(os.path.join(repo_path,r"Figures","Fig11"))

box_plotter.plot_energy_boxplot(data_paths=data_paths, showfliers=True, show_legend=True, shots=10000)
plt.savefig(os.path.join(repo_path,r"Figures","Fig12"))













