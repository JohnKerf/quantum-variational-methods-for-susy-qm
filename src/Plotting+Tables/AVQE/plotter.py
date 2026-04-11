import os, json
import numpy as np
import matplotlib.pyplot as plt
import git

repo_path = git.Repo('.', search_parent_directories=True).working_tree_dir

os.makedirs("Figures", exist_ok=True)

import matplotlib as mpl
from susy_qm.reporting.plot_style import PLOT_STYLE
mpl.rcParams.update(PLOT_STYLE)

POTENTIAL_LABELS = {
    "QHO": "HO", #Needed to rename for paper
    "AHO": "AHO",
    "DW": "DW",
}

def pot_label(p):
    return POTENTIAL_LABELS.get(p, p)


markers = ["o", "s", "^"]
potentials = ["QHO","AHO","DW"]
cutoffs = [2,4,8,16,32,64]

fig, axes = plt.subplots(2, 3, figsize=(12,8), sharex=False, sharey=True)

for i, cutoff in enumerate(cutoffs):
    row, col = divmod(i, 3)         
    ax = axes[row, col]

    for potential, marker in zip(potentials,markers):

        num_qubits = int(np.log2(cutoff)+1)

        dpath = os.path.join(repo_path, r"data\Table2+Fig7\{}\data_{}.txt".format(potential,cutoff))

        with open(dpath, "r", encoding="utf-8") as file:
            content = file.read()

        json_part = content.split("###############################")[0].strip()
        data = json.loads(json_part)

        min_eigenvalue = data['min_eigenvalue']
        energies = data['best_energy_list_reduced']

        label = f"{pot_label(potential)}{cutoff}"

        line, = ax.plot(range(1, len(energies) + 1), energies, marker=marker, linestyle='--', alpha=0.8, linewidth=1, label=pot_label(potential))#markersize= 4,
        colour = line.get_color()
        ax.axhline(y=min_eigenvalue, color=colour, linestyle=':')

        ax.set_title(f"$\\Lambda$={cutoff}")

        xticks = range(1, len(energies)  + 1)

        if len(xticks) >= 18:
            ax.set_xticks(xticks[::2])
        else:
           ax.set_xticks(xticks)

for c in range(1, axes.shape[1]):
    for ax in axes[:, c]:
        ax.tick_params(left=False, labelleft=False)

# x-labels only on the bottom row
for ax in axes[-1, :]:
    ax.set_xlabel('AVQE Step')

# y-label only on the first column
for ax in axes[:, 0]:
    ax.set_ylabel('VQE Energy')

axes[0, 0].legend()

fig.tight_layout(pad=0.6)

save_path = os.path.join(repo_path, "Figures", "Fig7.png")
plt.savefig(save_path)
