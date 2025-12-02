import pennylane as qml
from pennylane import numpy as pnp

from scipy.optimize import minimize

import os, json, time, logging
import numpy as np
from datetime import datetime, timedelta

from multiprocessing import Pool

from susy_qm import calculate_Hamiltonian, ansatze

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import git
repo_path = git.Repo('.', search_parent_directories=True).working_tree_dir

def setup_logger(logfile_path, name, enabled=True):
    if not enabled:
        
        logger = logging.getLogger(f"{name}_disabled")
        logger.handlers = []               
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        logger.setLevel(logging.CRITICAL)
        return logger

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(logfile_path)
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger



def cost_function(paulis, coeffs, ansatz, num_qubits, dev):

    def energy(params):

        @qml.qnode(dev)
        def circuit(params):
            ansatz(params, num_qubits)
            return [qml.expval(op) for op in paulis]

        expvals = circuit(params)                 
        energy = pnp.dot(coeffs, expvals)
            
        return energy

    grad_fn = qml.grad(energy)

    def f_and_g(x):
        x_pl = qml.numpy.array(x, requires_grad=True)
        val = energy(x_pl)
        grad = grad_fn(x_pl)
        
        return float(val), np.asarray(grad, dtype=float)

    return f_and_g

    
def run_vqe(i, paulis, coeffs, ansatz, run_info, log_enabled, log_dir):

    if log_enabled: os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"vqe_run_{i}.log")
    logger = setup_logger(log_path, f"logger_{i}", enabled=log_enabled)

    num_qubits = run_info["num_qubits"]
    num_params = run_info["num_params"]

    seed = (os.getpid() * int(time.time())) % 123456789
    dev = qml.device(run_info["device"], wires=num_qubits, shots=run_info["shots"], seed=seed)
    run_start = datetime.now()


    valgrad = cost_function(paulis, coeffs, ansatz, num_qubits, dev)

    def fun_and_jac(x):
        energy, grad = valgrad(x) 

        neg = max(0.0, -(energy + run_info["eps"]))
        e = energy + run_info["lam"] * (neg ** run_info["p"])
        
        return e, grad


    np.random.seed(seed)
    x0 = np.random.random(size=num_params)*2*np.pi

    if run_info["use_bounds"]:
        bounds = [(0, 2 * np.pi) for _ in range(num_params)]
    else:
        bounds = [(None, None) for _ in range(num_params)]

    run_info["seed"] = seed

    if log_enabled: logger.info(json.dumps(run_info, indent=4, default=str))
    if log_enabled: logger.info(f"Starting VQE run {i} (seed={seed})")

    with qml.Tracker(dev) as tracker:
        res = minimize(
            fun_and_jac,
            x0,
            bounds=bounds,
            method="L-BFGS-B",
            jac=True,
            options=run_info["optimizer_options"],
        )

    run_end = datetime.now()
    run_time = run_end - run_start

    if log_enabled: logger.info(f"Completed VQE run {i}: Energy = {res.fun:.6f}")
    if log_enabled: logger.info(f"optimizer message: {res.message}")

    totals = getattr(tracker, "totals", {})
    num_evals = int(totals.get("executions", 0))

    results_data = {
        "seed": seed,
        "energy": res.fun,
        "params": res.x.tolist(),
        "success": res.success,
        "num_iters": res.nit,
        "num_evaluations": num_evals,
        "run_time": run_time
    }

    if log_enabled: logger.info(json.dumps(results_data, indent=4, default=str))

    return results_data

if __name__ == "__main__":
    
    log_enabled = False
    num_processes = 1

    potential = "DW"
    device = 'default.qubit'

    shots = None
    use_bounds = False if shots == None else True
    cutoff = 4

    lam = 15
    p = 2

    # Optimizer
    num_vqe_runs = 1
    max_iter = 10000
    tol = 1e-8

    optimizer_options = {
                    'maxiter':max_iter, 
                    'tol':tol
                    }


    ansatz_name = 'real_amplitudes'
    num_layers = 1
    ansatz = ansatze.get(ansatz_name)
    
    if potential == "AHO":
        i = np.log2(cutoff)
        factor = 2**(((i-1)*i)/2)
        eps = 0.5 / factor
    else:
        eps = 0

    
    starttime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_path = os.path.join(repo_path, r"scratch_data\vqe", str(shots), potential)
    os.makedirs(base_path, exist_ok=True)

    log_path = os.path.join(base_path, f"logs_{str(cutoff)}")

    # Calculate Hamiltonian and expected eigenvalues
    H = calculate_Hamiltonian(cutoff, potential)
    
    eigenvalues = np.sort(np.linalg.eig(H)[0])[:4]
    num_qubits = int(1 + np.log2(cutoff))

    num_params = ansatz.n_params(num_qubits,num_layers)
    
    H_decomp = qml.pauli_decompose(H, wire_order=range(num_qubits))
    paulis = H_decomp.ops
    coeffs = H_decomp.coeffs
    
    run_info = {"device":device,
                "Potential":potential,
                "cutoff": cutoff,
                "ansatz_name": ansatz_name,
                "num_qubits": num_qubits,
                "num_paulis": len(paulis),
                "num_params": num_params,
                "shots": shots,
                "lam": lam,
                "p":p,
                "eps":eps,
                "num_vqe_runs": num_vqe_runs,
                "optimizer_options": optimizer_options,
                "use_bounds": use_bounds,
                "path": base_path
                }

    print(json.dumps(run_info, indent=4, default=str))


    vqe_starttime = datetime.now()

    # Start multiprocessing for VQE runs
    with Pool(processes=num_processes) as pool:
        vqe_results = pool.starmap(
            run_vqe,
            [
                (i, paulis, coeffs, ansatz, run_info, log_enabled, log_path)
                for i in range(num_vqe_runs)
            ],
        )

    # Collect results
    seeds = [res["seed"] for res in vqe_results]
    energies = [res["energy"] for res in vqe_results]
    x_values = [res["params"] for res in vqe_results]
    success = [res["success"] for res in vqe_results]
    num_iters = [res["num_iters"] for res in vqe_results]
    num_evals = [res["num_evaluations"] for res in vqe_results]
    run_times = [str(res["run_time"]) for res in vqe_results]
    total_run_time = sum([res["run_time"] for res in vqe_results], timedelta())

    vqe_end = datetime.now()
    vqe_time = vqe_end - vqe_starttime

    # Save run
    run = {
        "device": device,
        "starttime": starttime,
        "endtime": vqe_end.strftime("%Y-%m-%d_%H-%M-%S"),
        "potential": potential,
        "cutoff": cutoff,
        "num_qubits": num_qubits,
        "num_paulis": len(paulis),
        "num_params": num_params,
        "exact_eigenvalues": [x.real.tolist() for x in eigenvalues],
        "ansatz": ansatz_name,
        "num_VQE": num_vqe_runs,
        "shots": shots,
        "Optimizer": {
            "name": "L-BFGS-B",
            "optimizer_options":optimizer_options,
        },
        "cost function":{
            "type": "small negatives",
            "p":p,
            "lam":lam,
            "eps":eps
        },
        "results": energies,
        "params": x_values,
        "num_iters": num_iters,
        "num_evaluations": num_evals,
        "success": np.array(success, dtype=bool).tolist(),
        "run_times": run_times,
        "parallel_run_time": str(vqe_time),
        "total_VQE_time": str(total_run_time),
        "seeds": seeds
    }

    # Save the variable to a JSON file
    path = os.path.join(base_path, "{}_{}.json".format(potential, cutoff))
    with open(path, "w") as json_file:
        json.dump(run, json_file, indent=4)

    print("Done")
