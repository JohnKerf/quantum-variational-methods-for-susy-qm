import os
from susy_qm.reporting.avqe import AVQEProcessing
import git

repo_path = git.Repo('.', search_parent_directories=True).working_tree_dir

folder_path = os.path.join(repo_path, r"paper_results_data\Table2&Fig7")
cutoffs = [2,4,8,16,32,64]
potentials = ["QHO","DW","AHO"]

avqe = AVQEProcessing(folder_path, potentials, cutoffs)
avqe.save_avqe_data()

df = avqe.create_summary_table()
df.to_latex(os.path.join(repo_path, r"paper_results_data\Table2&Fig7", "tab.tex"), index=False)