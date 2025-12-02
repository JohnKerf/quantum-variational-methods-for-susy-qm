# Quantum Variational Methods for Supersymmetric Quantum Mechanics

This repository contains the research code accompanying the paper https://doi.org/10.48550/arXiv.2510.26506

> **Quantum Variational Methods for Supersymmetric Quantum Mechanics**  
> J. Kerfoot, E. Mendicelli, D. Schaich

We employ quantum variational methods to investigate a single-site interacting fermion–boson system—an example of a minimal supersymmetric model that can exhibit spontaneous supersymmetry breaking. By using adaptive variational techniques, we identify optimal ansätze that scale efficiently, allowing for reliable identification of spontaneous supersymmetry breaking. This work lays a foundation for future quantum computing investigations of more complex and physically rich fermion–boson quantum field theories in higher dimensions.

---

## Repository structure

```text
.
├── data/                # Reference data used in the paper
├── src/
│   ├── algorithms/      # Implementations of variational algorithms (VQE, AVQE, VQD)
│   ├── plotting&tables/ # Scripts to generate plots and tables used in the paper
│   └── susy_qm/         # Custom python package containing Hamiltonian, ansatze and reporting methods
├── pyproject.toml       # Build & dependency configuration
├── LICENSE              # MIT license
└── README.md
```

---

## Installation

The project is packaged via pyproject.toml and can be installed using the below code.

```python
git clone https://github.com/JohnKerf/quantum-variational-methods-for-susy-qm.git
cd quantum-variational-methods-for-susy-qm

python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -e .
```

Running this code will ensure all of the relevant packages are installed making the code contained in the repo runs as intended.

---

## Data

The `data/` directory contains all numerical results used in the paper.

 - Each subdirectory corresponds to a particular figure or table in the paper.

 - Data is stored in JSON format to keep runs, metadata and seeds together.

 - The "seeds" field in each JSON file records the random seeds used in the corresponding run so that each run can be replicated.

- Additional fields typically include things like:

    - Hamiltonian configuration (potential, cutoff, num_qubits, exact eigenvalues, ...)

    - Ansatz configuration (ansatz name, number of parameters, ... )

    - Optimizer configuration (optimizer name, tolerance, ... )

    - Algorithm configuration (number of shots, number of runs, ... )

    - Results (VQE energies, AVQE circuits, VQD energy levels, optimised parameters, ...)

    - Convergence information (number of iterations, number of evaluations, run times,...)

These files are read directly by the scripts in src/plotting&tables to recreate the figures and tables within the paper.

---

## Algorithms
The `src/algorithms/` directory contains implementations of the three variational quantum algorithms studied in the paper. Each subfolder corresponds to one algorithm and contains a .py file that can be ran. Each file contains variables that can be configured to alter the run outcomes. Variables can be taken from the files in `data/` to replicate runs.  

### Variational Quantum Eigensolver (VQE)

Within `src/algorithms/VQE` there are separate folders for each of the optimisers that were tested. 

- L-BFGS-B - local, gradient based 
- Differential Evolution - global, stochastic and gradient-free
- COBYLA - local, gradient-free (linear approximation)
- COBYQA - local, gradient-free (quadratic approximation)

The optimizer used for the AVQE and VQD algorithms is the COBYQA optimizer. Additionally, the final VQE results using the AVQE ansatze are using the `src/algorithms/VQE.COBYQA/VQE.py` file.

Code implementation:

1. Define the initial parameters for the algorithm
2. Choose an ansatz using the `src/susy_qm/ansatze` `.get` method
3. If you want to truncate the number of gates in the ansatz, set the `max_gate` variable to the truncation level (this is used for setting up the truncated ansatze in the paper). Leaving `max_gate` as None will use the complete ansatz. The number of parameters is automatically calcualted based on the chosen ansatz.
4. The Hamiltonian is created using the `src/susy_qm/hamiltonian` `.calculate_Hamiltonian` method using the `cutoff` and `potential` variables.
5. This information is passed to the `run_vqe` function
6. The optimization process is done using `scipy.optimize.minimize` which iteratively calls `wrapped_cost_function` which in turn calls the `cost_function` that ultimately computes the expectation energy.
7. This process will happen until either the optimizer converges to the `final_tr_radius` (when using COBYQA but can also be `tol` or another stopping criteria depending on the optimizer chosen) or if `maxiter` or `maxfev` is reached.
8. Data is then stored and saved in json format


### Adaptive-VQE (AVQE)

Within `src/algorithms/AVQE` there is a single `avqe.py` file that can be run to produce the avqe results. 

Code implementation:

1. Define the initial parameters for the algorithm
2. The Hamiltonian is created using the `src/susy_qm/hamiltonian` `.calculate_Hamiltonian` method using the `cutoff` and `potential` variables.
3. Construct the operator pool
4. Define the initial basis state depending on the choice of `potential` and `cutoff`
5. This information is passed to the `run_adapt_vqe` function
6. For each operator in the pool, calculate the gradient using the `compute_grad` function
7. Add to `op_list` the operator which resulted in the largest gradient (`max_op`). `op_list` is then used to construct the ansatz in the vqe step.
8. The vqe process is done using `scipy.optimize.minimize` which iteratively calls `wrapped_cost_function` which in turn calls the `cost_function` that ultimately computes the expectation energy using the ansatz constructed from `op_list`. `x0` (the intital set of parameters) is constructed using previous optimal parameters `op_params` from the previous step.
9. `max_op` is removed from the pool once it has been used and is only re-added after a 2-qubit (CRY) gate has been implemented with the qubit. This prevents redundant gates from being added. 
10. This process will happen until either the optimizer converges or until `num_steps` is reached.
11. Data is then stored and saved in json format


### Variational Quantum Deflation (VQD)

Within `src/algorithms/VQD` there is a sinlge `vqd.py` file that can be ran to produce the vqd results. 

Code implementation:

1. Define the initial parameters for the algorithm
2. The Hamiltonian is created using the `src/susy_qm/hamiltonian` `.calculate_Hamiltonian` method using the `cutoff` and `potential` variables.
3. This information is passed to the `run_vqd` function
4. A VQE is then performed for `num_energy_levels` number of energy levels
5. The vqe process is done using `scipy.optimize.minimize` which calls `cost_function`.
6. Within `cost_function` the overlap is computed using the `overlap` function which uses `swap_test` to calculate the overlap probability.
7. The combined loss term is a combination of the expectation energy computed using `expected_value` and the penalty term which is the overlap of the current state with all previously computed eigenstates scaled by `beta`. In the case of the groundstate no penalty term is added.
8. This combined loss function `loss_f` is what is returned to the optimizer which ideally will resolve the first `num_energy_levels` number of eigenstates.
9. Data is then stored and saved in json format

---

## Plotting & Tables

The `src/plotting&tables/` directory contains scripts that read the JSON files in `data/` and generate the figures and tables appearing in the paper. There is a subfolder for each variational algorithm (VQE, AVQE and VQD) each containing separate .py files for producing plots and tables.

### Variational Quantum Eigensolver (VQE)

`src/plotting&tables/VQE/`

- This directory contains a `plotter.py` file that creates the plots for figures 5, 6, 9, 10, 11 and 12.
- Plots are created using the `BoxPlotter` and `VQEPlotter` classes from `src/susy_qm/reporting/vqe_plotter`.
- These classes are initiated with `data_paths`, `potentials` and  `cutoffs` where `data_paths` is a list of data to be plotted and is in the form [(label, folder path)] where label is the label that is shown in the legend on the plot.
    - Figures 9.1 and 10.1 (E Vs Cutoff) are plotted using  `VQEPlotter.plot_delta_e_vs_cutoff_line` 
    - Figures 9.2 and 10.2 (Number of evaluations) are plotted using `VQEPlotter.plot_evals_vs_cutoff_box` 
    - Figures 11 and 12 Box plots are plotted using `BoxPlotter.plot_energy_boxplot`


### Adaptive-VQE (AVQE)

`src/plotting&tables/AVQE/` 

- The directory contains a `plotter.py` file that creates the plot for figure 7
    - Uses the AVQE data obtained from running the `avqe.py` algorithm to plot VQE energy at each step of the algorithm

- The directory contains a `tables.py` file that creates Tables 2 and 3.
    - This uses the `AVQEProcessing` class from `src/susy_qm/reporting/avqe` 
    - Table 2 is created using `AVQEProcessing.create_summary_table`
    - Table 3 is created using `AVQEProcessing.create_steps_table`


### Variational Quantum Deflation (VQD)

`src/plotting&tables/VQD/` 

- The directory contains a `plotter.py` file that creates the plot for figure 13
    - Uses the VQD data obtained from running the `vqd.py` algorithm to plot the VQE energies for each energy level within an individual VQD run.
    - Plots are created using the `VQDPlotter` class from `src/susy_qm/reporting/vqd_plotter`.

- The directory contains a `tables.py` file that creates Tables 5, 6 and 7.
    - This uses the `VQDTables` class from `src/susy_qm/reporting/vqd_tables` 
    - Table 5 is created using `VQDTables.create_beta_comp_table`
    - Table 6 & 7 are created using `VQDTables.create_ratio_table`
    
---

## susy_qm package

The `src/susy_qm/` directory contains a custom python package that has the reusable building blocks used throughout the project ( Hamiltonians, ansatze, and reporting utilities).

### ansatze

This module contains constructors for various parametrized quantum circuits used throughout the project:

- `real_amplitudes`
- The full ansatze produced from the AVQE algorithm which are called using `CQAVQE_*_exact` where * is the potential-cutoff combination you are using e.g. `CQAVQE_DW4_exact` would give you the full ansatze for the DW potential with a cutoff of 4.
- The reduced ansatze which are the full ansatze truncated to a maximum of 4 gates which are called by `CQAVQE_*_Reduced`

This module also contains utilities:

- `.get` is used to retrieve the desired circuit. e.g. .get(CQAVQE_DW4_exact)
- `.gate_list_from_ansatz` is used to retrieve a list of [{gate, wires, param}] for a quantum circuit
- `.ansatz_by_gate` is used to return an ansatz gate-by-gate and is what is used to truncate a quantum circuit to a `max_gate`  

### hamiltonian

This module contains the `calculate_Hamiltonian` function that is used to create the SUSY QM Hamiltonian in matrix form

### reporting

This directory contains the modules used in the plotting&tables code.

The `avqe.py` file contains the `AVQEProcessing` class.
- `find_optimal_circuit` is used to find the best circuit out of the multiple AVQE runs 
- `optimal_circuit_diagram` is used to create the circuit diagram for a circuit
- `save_avqe_data` saves both the json and circuit diagram
- `create_summary_table` is used to create table 2 in the paper
- `create_steps_table` is used to create table 3 in the paper


The `vqd_plotter.py` file contains the `VQDPlotter` class.
- `_load` loads the data from the json files and sorts and filters them
- `plot_vqd` is used to create figure 13

The `vqd_tables.py` file contains the `VQDTables` class.
- `_load` loads the data from the json files, aggregates the data and calculates the ratios
- `create_ratio_table` is used to create tables 6 & 7 and calls `_load` to create a table for given potentials and cutoffs
- `create_beta_comp_table`is used to create tables 5 and also uses `_load` to create a per beta value table

The `vqe_metrics.py` file contains the `VQESummary` class.
- This class takes a path along with a list of potentials and cutoffs to create an aggregated summary table for the multiple number of runs
- There are several getters to retrieve different forms of the aggregated data

The `vqe_plotter.py` uses the `VQESummary` class from `vqe_metrics.py` and also contains the `VQEPlotter` and `BoxPlotter` classes.
- `format_axis` is helper to consistently style the axes  
- `VQEPlotter` uses the aggregated data from `VQESummary`
    -  `plot_delta_e_vs_cutoff_line` plots the error (absolute difference to exact) vs cutoff for each dataset. This is used to create figures 9.1 and 10.1
    -  `plot_evals_vs_cutoff_box` grouped boxplots per dataset for the number of evaluations vs cutoff. This is used to create figures 9.2 and 10.2
- `BoxPlotter` plots boxplots for the distribution of the VQE runs
    - `plot_energy_boxplot` is used to plot the box plots and is used to create figures 11 & 12


