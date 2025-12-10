
import os
from susy_qm.reporting.vqd_tables import VQDTables
import git

repo_path = git.Repo('.', search_parent_directories=True).working_tree_dir
        

folder_path = os.path.join(repo_path, r"paper_results_data\Table6&7")
cutoffs = [2,4,8,16,32]
potentials = ["QHO","DW","AHO"]


vqd = VQDTables(folder_path,potentials,cutoffs)

shots=None
ratio_table = vqd.create_ratio_table(shots=shots)
ratio_table.to_latex(os.path.join(folder_path, str(shots), f"ratios.tex"), index=False, float_format="%.4f")

shots=10000
ratio_table = vqd.create_ratio_table(shots=shots)
ratio_table.to_latex(os.path.join(folder_path, str(shots), f"ratios.tex"), index=False, float_format="%.4f")



folder_path = os.path.join(repo_path, r"paper_results_data\Table5")
potentials = ["QHO","DW","AHO"]
cutoff=16

vqd = VQDTables(folder_path,potentials,cutoffs)

for potential in potentials:
    beta_table = vqd.create_beta_comp_table(potential, cutoff)
    beta_table.to_latex(os.path.join(folder_path, potential, f"{potential}{cutoff}.tex"), index=False, float_format="%.4f")