from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List
import os, json
import numpy as np

import pennylane as qml
import pandas as pd

from susy_qm import ansatze


import git
repo_path = git.Repo('.', search_parent_directories=True).working_tree_dir

def _load_json(fp: Path) -> Dict[str, Any]:
    fp = Path(fp)
    with fp.open("r", encoding="utf-8") as f:
        return json.load(f)


@dataclass(slots=True)
class AVQEProcessing:
    data_path: Path
    potentials: List[str]
    cutoffs: list[int]
    
    def find_optimal_circuit(self, potential, cutoff):

        path = os.path.join(self.data_path, potential, f"{potential}_{cutoff}.json")
        data = _load_json(path)

        energies = data['all_energies']
        
        num_qubits = int(np.log2(cutoff)) + 1
        min_eigenvalue = min(np.asarray(data.get("exact_eigenvalues", []), dtype=float))

        closest_e = np.inf
        best_index = 0
        gate_length = np.inf
        best_energy_value = None
        best_gate_position = None

        for i, e1 in enumerate(energies):
            for g_idx, e2 in enumerate(e1, start=1):  
                ediff = abs(e2 - min_eigenvalue)

                if ediff < closest_e or ((np.abs(ediff - closest_e) < 1e-6) and g_idx < gate_length):
                    closest_e = ediff
                    best_index = i
                    gate_length = g_idx
                    best_energy_value = e2
                    best_gate_position = g_idx


        best_gate_set = data['op_list'][best_index]

        best_energy_list_reduced = energies[best_index][:best_gate_position]
        energy_diffs = [abs(x - min_eigenvalue) for x in best_energy_list_reduced]
        energy_change = [0.0] + [abs(best_energy_list_reduced[i]-best_energy_list_reduced[i-1]) for i in range(1, len(best_energy_list_reduced))]
        seed = data['seeds'][best_index]
        success = data['success'][best_index]
        basis_state = data["basis_state"]
        best_circuit = best_gate_set[:best_gate_position]

        data_dict = {}
        data_dict['seed'] = seed
        data_dict['success'] = success
        data_dict['potential'] = potential
        data_dict['cutoff'] = cutoff
        data_dict['num_qubits'] = num_qubits
        data_dict['basis_state'] = basis_state
        data_dict['min_eigenvalue'] = min_eigenvalue
        data_dict['best_energy_value'] = best_energy_value
        data_dict['best_energy_list_reduced'] = best_energy_list_reduced
        data_dict['energy_diffs'] = energy_diffs
        data_dict['energy_change'] = energy_change
        data_dict['circuit'] = best_circuit

        return data_dict
            

    def optimal_circuit_diagram(self, optimal_circuit):

        num_qubits = optimal_circuit['num_qubits']
        basis_state = optimal_circuit['basis_state']

        dev = qml.device("default.qubit", wires=num_qubits)
        @qml.qnode(dev)
        def circuit():

            qml.BasisState(basis_state, wires=range(num_qubits))

            for op_dict in optimal_circuit['circuit']:
                op = getattr(qml, op_dict["name"])
                op(op_dict['param'], wires=op_dict['wires'])

            return qml.state()

        circuit_diagram = qml.draw(circuit)()
        mpl_diagram, ax = qml.draw_mpl(circuit, style='pennylane')()

        return circuit_diagram, mpl_diagram
    

    def save_avqe_data(self):

        for potential in self.potentials:
            for cutoff in self.cutoffs:

                optimal_circuit = self.find_optimal_circuit(potential,cutoff)

                txt_path = os.path.join(self.data_path, potential, f"data_{cutoff}.txt")

                with open(txt_path, "w") as file:
                    json.dump(optimal_circuit, file, indent=4)

                circuit_diagram, mpl_diagram = self.optimal_circuit_diagram(optimal_circuit)

                circuit_path = os.path.join(self.data_path, potential, "circuitDiagrams", f"{potential}_{cutoff}.png")
                mpl_diagram.savefig(circuit_path)

                with open(txt_path, "a", encoding="utf-8") as file:
                    file.write("\n###############################\n")
                    file.write(circuit_diagram)
                    file.write("\n###############################\n")
    

    def create_summary_table(self):

        all_data = []

        for potential in self.potentials:
            for cutoff in self.cutoffs:

                optimal_circuit = self.find_optimal_circuit(potential,cutoff)

                circuit = optimal_circuit['circuit']
                num_qubits = optimal_circuit['num_qubits']
                potential = optimal_circuit['potential']
                cutoff = optimal_circuit['cutoff']
                basis_state = optimal_circuit['basis_state']
                best_energy_value = optimal_circuit['best_energy_value']
                min_eigenvalue = optimal_circuit['min_eigenvalue']

                op_labels = ""
                counter = 0
                params = []
                for op_dict in circuit:

                    gate = op_dict["name"]
                    params.append(op_dict["param"])
                    wires = op_dict["wires"]

                    qiskit_wires = [f"$q_{(num_qubits - 1) - w}$" for w in wires]
                    label = f"{gate}[{', '.join(qiskit_wires)}]"

                    if counter == 0:
                        op_labels = label
                    else:
                        op_labels = op_labels + ", " + label

                    counter +=1


                num_gates = len(circuit)

                dd = {"Potential": potential,
                    r"$\Lambda$": cutoff,
                    "Basis State": r"$\ket{" + "".join(str(num) for num in basis_state) + "}$",
                    r"$N_\text{gates}$": num_gates,
                    "Ansatz":op_labels,
                    r"$E_{\text{VQE}}$": best_energy_value,
                    r"$E_{\text{Exact}}$": min_eigenvalue,
                    }

                all_data.append(dd)

        return pd.DataFrame(all_data).sort_values(['Potential',r'$\Lambda$'])
    

    def create_steps_table(self, shots):

        for potential in self.potentials:
            for cutoff in self.cutoffs:

                folder_path = os.path.join(repo_path, self.data_path, str(shots), potential, str(cutoff))
                folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

                num_qubits = int(np.log2(cutoff)+1) 
                ansatz_name = f"CQAVQE_{potential}{cutoff}_exact"
                ansatz = ansatze.get(ansatz_name)
                num_params = ansatz.n_params

                params = np.array([0]*num_params)
                gates = ansatze.gate_list_from_ansatz(ansatz, params, num_qubits)

                gate_list = {}
                op_labels = ""
                counter = 1
                for item in gates:
                    if item["gate"] == "BasisState":
                        continue

                    gate = item["gate"]
                    wires = item["wires"]

                    qiskit_wires = [f"$q_{(num_qubits - 1) - w}$" for w in wires]
                    label = f"{gate}[{', '.join(qiskit_wires)}]"

                    if counter == 1:
                        op_labels = label
                    else:
                        op_labels = op_labels + ", " + label

                    gate_list[str(counter)] = op_labels
                    counter +=1

                avqe_path = os.path.join(repo_path, self.data_path, "avqe", potential,f"data_{cutoff}.txt")

                with open(avqe_path, "r", encoding="utf-8") as f:
                    text = f.read()

                json_part = text[: text.rfind("}") + 1]
                data = json.loads(json_part)

                energy_diffs = data["energy_diffs"]
                step_energies = data["best_energy_list_reduced"]

                all_data = []

                for f in folders:

                    gate_count = int(f.split("_")[-1])

                    dpath = os.path.join(folder_path,f,f"{potential}_{cutoff}.json")
                    
                    with open(dpath, 'r') as file:
                        data = json.load(file)

                    exact_e = np.min(data['exact_eigenvalues'])

                    median_e = np.median(data['results'])
                    delta_median_e = abs(exact_e - np.median(data['results']))
                    

                    ansatz_step = gate_list[str(gate_count)]

                    data_dict = {"Potential": potential,
                                r"$\Lambda$": cutoff,
                                "avqe step": gate_count,
                                "Ansatz": ansatz_step, 
                                "VQE-None": step_energies[gate_count-1],
                                "VQE-None-Diff": energy_diffs[gate_count-1],
                                "VQE-10k": median_e,
                                "VQE-10K-Diff": delta_median_e
                                }
                    
                    all_data.append(data_dict)

                df = pd.DataFrame(all_data).sort_values("avqe step", ascending=True)
                df.to_latex(os.path.join(folder_path, f"{potential}{cutoff}.tex"), index=False, float_format="%.6f")








   