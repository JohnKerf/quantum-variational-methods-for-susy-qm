from susy_qm.reporting.vqd_plotter import VQDPlotter
import matplotlib.pyplot as plt
import os

import git
repo_path = git.Repo('.', search_parent_directories=True).working_tree_dir

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


data_path = os.path.join(repo_path,r"paper_results_data\Fig13")
potentials = ["QHO", "AHO","DW"]
cutoff=16
vqd_plotter = VQDPlotter(data_path,potentials,cutoff)

vqd_plotter.plot_vqd(shots=None)
plt.savefig(os.path.join(repo_path,r"Figures","Fig13"))












