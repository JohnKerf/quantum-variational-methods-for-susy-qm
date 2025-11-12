import pennylane as qml
from pennylane import numpy as pnp

from scipy.optimize import minimize

import os
import json
import numpy as np
from datetime import datetime, timedelta
import time

from multiprocessing import Pool

from collections import Counter

from susy_qm import calculate_Hamiltonian

import git
repo_path = git.Repo('.', search_parent_directories=True).working_tree_dir



def compute_grad(param, H, num_qubits, operator_ham, op_list, op_params, basis_state, dev):

    @qml.qnode(dev)
    def grad_circuit(param, operator_ham, op_list, op_params):

        qml.BasisState(basis_state, wires=range(num_qubits))

        param_index = 0
        for op in op_list:
            o = type(op)
            o(op_params[param_index], wires=op.wires)
            param_index +=1

        oph = type(operator_ham)
        oph(param, wires=operator_ham.wires)

        return qml.expval(qml.Hermitian(H, wires=range(num_qubits)))
    
    params = pnp.tensor(param, requires_grad=True)
    grad_fn = qml.grad(grad_circuit)
    grad = grad_fn(params, operator_ham, op_list, op_params)
    
    return grad



def cost_function(params, H, num_qubits, shots, op_list, basis_state, dev):
   
    start = datetime.now()
  
    @qml.qnode(dev)
    def circuit(params):

        qml.BasisState(basis_state, wires=range(num_qubits))

        param_index = 0
        for op in op_list:
            o = type(op)
            o(params[param_index], wires=op.wires)
            param_index +=1

        return qml.expval(qml.Hermitian(H, wires=range(num_qubits)))

     
    end = datetime.now()
    device_time = (end - start)

    return circuit(params), device_time


def run_adapt_vqe(i, H, run_info):

    num_qubits = run_info["num_qubits"] 
    shots = run_info["shots"]  
    basis_state = run_info["basis_state"]
    phi = run_info["phi"]


    # We need to generate a random seed for each process otherwise each parallelised run will have the same result
    seed = (os.getpid() * int(time.time())) % 123456789
    dev = qml.device(run_info["device"], wires=num_qubits, shots=run_info["shots"], seed=seed)
    run_start = datetime.now()

    device_time = timedelta()

    def wrapped_cost_function(params):
        result, dt = cost_function(params, H, num_qubits, shots, op_list, basis_state, dev)
        nonlocal device_time
        device_time += dt
        return result
    

    # Main ADAPT-VQE script
    op_list = []
    op_params = []
    energies = []

    pool = run_info["operator_pool"].copy()
    success = False

    for i in range(run_info["num_steps"]):

        max_ops_list = []
        
        if i != 0:
            
            pool.remove(most_common_gate)

            if  (type(most_common_gate) == qml.CRY):
                cq = most_common_gate.wires[0]
                tq = most_common_gate.wires[1]

                if (qml.RY(phi, wires=cq) not in pool):
                    pool.append(qml.RY(phi, wires=cq))

                if (qml.RZ(phi, wires=cq) not in pool):
                    pool.append(qml.RZ(phi, wires=cq))

                if (qml.RY(phi, wires=tq) not in pool):
                    pool.append(qml.RY(phi, wires=tq))

                if (qml.RZ(phi, wires=tq) not in pool):
                    pool.append(qml.RZ(phi, wires=tq))

        
        for param in np.random.uniform(phi, phi, size=run_info["num_grad_checks"]):
            grad_list = []
            for op in pool:
                grad = compute_grad(param, H, num_qubits, op, op_list, op_params, basis_state, dev)
                o=type(op)

                if (o == qml.CNOT) or (o == qml.CZ):
                    grad_op = o(wires=op.wires)
                else:
                    grad_op = o(param, wires=op.wires)

                grad_list.append((grad_op,abs(grad)))

            max_op, max_grad = max(grad_list, key=lambda x: x[1])
            max_ops_list.append(max_op)

        counter = Counter(max_ops_list)
        most_common_gate, count = counter.most_common(1)[0]
        op_list.append(most_common_gate)


        np.random.seed(seed)
        x0 = np.concatenate((op_params, np.array([np.random.random()*2*np.pi])))
        
        res = minimize(
            wrapped_cost_function,
            x0,
            method= "COBYQA",
            options= run_info["optimizer_options"]
        )
        
        if i!=0: pre_min_e = min_e
        min_e = res.fun
        pre_op_params = op_params.copy()
        op_params = res.x

        energies.append(min_e)

        if i!=0:
            if abs(pre_min_e - min_e) < 1e-8:
                energies.pop()
                op_list.pop()
                final_params = pre_op_params
                success = True
                break
            if abs(run_info["min_eigenvalue"]-min_e) < 1e-6:
                success = True
                final_params = op_params
                break
        
    run_end = datetime.now()
    run_time = run_end - run_start

    if success == False:
        final_params = op_params

    final_ops = []
    for op, param in zip(op_list,final_params):
        dict = {"name": op.name,
                "param": param,
                "wires": op.wires.tolist()}
        final_ops.append(dict)

    return {
        "seed": seed,
        "energies": energies,
        "min_energy": min_e,
        "op_list": final_ops,
        "success": success,
        "num_iters": i+1,
        "run_time": run_time,
        "device_time": device_time
    }


if __name__ == "__main__":

    device = 'default.qubit'
    num_processes=1
    
    potential = "AHO"
    cutoff = 2
    shots = None

    # Optimizer
    num_steps = 10
    num_grad_checks = 1
    num_vqe_runs = 1
    max_iter = 10000
    initial_tr_radius = 1.0
    final_tr_radius = 1e-8
    scale=True

    optimizer_options = {
                    'maxiter':max_iter, 
                    'maxfev':max_iter, 
                    'initial_tr_radius':initial_tr_radius, 
                    'final_tr_radius':final_tr_radius, 
                    'scale':scale
                    }

    print(f"Running for {potential} potential, cutoff {cutoff}")

    starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_path = os.path.join(repo_path, r"scratch_data\avqe", potential)
    os.makedirs(base_path, exist_ok=True)


    # Calculate Hamiltonian and expected eigenvalues
    H = calculate_Hamiltonian(cutoff, potential)

    eigenvalues = np.sort(np.linalg.eig(H)[0])[:4]
    min_eigenvalue = np.min(eigenvalues)
    num_qubits = int(1 + np.log2(cutoff))

    #Create operator pool
    operator_pool = []
    phi = 0.0
    for i in range(1,num_qubits):
        operator_pool.append(qml.RY(phi,wires=[i]))
        operator_pool.append(qml.RZ(phi,wires=[i]))

    c_pool = []

    for control in range(1,num_qubits):
            for target in range(1,num_qubits):
                if control != target:
                    c_pool.append(qml.CRY(phi=phi, wires=[control, target]))

    operator_pool = operator_pool + c_pool    

    # Choose basis state
    if potential == 'DW':
        if cutoff == 4:
            basis_state = [1] + [0]*(num_qubits-1)
        else:
            basis_state = [0]*(num_qubits)
    else:
        basis_state = [1] + [0]*(num_qubits-1)


    run_info = {"device":device,
                "Potential":potential,
                "cutoff": cutoff,
                "num_qubits": num_qubits,
                "min_eigenvalue":min_eigenvalue,
                "shots": shots,
                "num_steps":num_steps,
                "num_grad_checks":num_grad_checks,
                "phi":phi,
                "num_vqe_runs": num_vqe_runs,
                "optimizer_options": optimizer_options,
                "basis_state":basis_state,
                "operator_pool":operator_pool,
                "path": base_path
                }
    

    vqe_starttime = datetime.now()

    print("Starting ADAPT-VQE")
    # Start multiprocessing for VQE runs
    with Pool(processes=num_processes) as pool:
        vqe_results = pool.starmap(
            run_adapt_vqe,
            [
                (i, H, run_info)
                for i in range(num_vqe_runs)
            ],
        )

    print("Finished ADAPT-VQE")
    # Collect results
    seeds = [res["seed"] for res in vqe_results]
    all_energies = [res["energies"] for res in vqe_results]
    min_energies = [res["min_energy"] for res in vqe_results]
    op_lists = [res["op_list"] for res in vqe_results]
    success = [res["success"] for res in vqe_results]
    num_iters = [res["num_iters"] for res in vqe_results]
    run_times = [str(res["run_time"]) for res in vqe_results]
    total_run_time = sum([res["run_time"] for res in vqe_results], timedelta())
    total_device_time = sum([res['device_time'] for res in vqe_results], timedelta())

    vqe_end = datetime.now()
    vqe_time = vqe_end - vqe_starttime

    # Save run
    run = {
        "starttime": starttime,
        "endtime": vqe_end.strftime("%Y-%m-%d_%H-%M-%S"),
        "potential": potential,
        "cutoff": cutoff,
        "exact_eigenvalues": [x.real.tolist() for x in eigenvalues],
        "ansatz": "circuit.txt",
        "shots": shots,
        "Optimizer": {
                "name": "COBYQA",
                "optimizer_options":optimizer_options
            },
        "num_VQE": num_vqe_runs,
        "num_steps":num_steps,
        "num_grad_checks":num_grad_checks,
        "phi": phi,
        "basis_state": basis_state,
        "operator_pool": [str(op) for op in operator_pool],
        "all_energies": all_energies,
        "min_energies": min_energies,
        "op_list": op_lists,
        "num_iters": num_iters,
        "success": np.array(success, dtype=bool).tolist(),
        "run_times": run_times,
        "seeds": seeds,
        "parallel_run_time": str(vqe_time),
        "total_VQE_time": str(total_run_time),
        "total_device_time": str(total_device_time)
    }

    # Save the variable to a JSON file
    path = os.path.join(base_path, "{}_{}.json".format(potential, cutoff))
    with open(path, "w") as json_file:
        json.dump(run, json_file, indent=4)

    print("Done")
