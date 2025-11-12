from susy_qm.reporting.vqe_plotter import BoxPlotter, VQEPlotter
import matplotlib.pyplot as plt
import os

import git
repo_path = git.Repo('.', search_parent_directories=True).working_tree_dir

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


cutoffs = [2,4,8,16,32,64]
potentials = ["QHO", "AHO","DW"]
shots_list = [None,10000]

data_paths=[
        ("Full", r"paper_results_data\Figs9-12\Full"),
        ("Truncated", r"paper_results_data\Figs9-12\Truncated"),
        ("Real Amplitudes", r"paper_results_data\Figs9-12\Real Amplitudes"),
    ]

line_plotter = VQEPlotter(data_paths, potentials, cutoffs)

line_plotter.plot_delta_e_vs_cutoff_line(shots=None, metric='median', scale="symlog", linthresh=1e-1, sharey=True)
plt.savefig(os.path.join(repo_path,r"Figures","Fig9"))

line_plotter.plot_delta_e_vs_cutoff_line(shots=10000, metric='median', scale="symlog", linthresh=1e-1, sharey=True)
plt.savefig(os.path.join(repo_path,r"Figures","Fig10"))




box_plotter = BoxPlotter(potentials, cutoffs, shots_list)

box_plotter.plot_energy_boxplot(data_paths=data_paths, showfliers=True, show_legend=True, shots=None)
plt.savefig(os.path.join(repo_path,r"Figures","Fig11"))

box_plotter.plot_energy_boxplot(data_paths=data_paths, showfliers=True, show_legend=True, shots=10000)
plt.savefig(os.path.join(repo_path,r"Figures","Fig12"))













