import json
import numpy as np
import os
import pickle
import shutil
import sqlite3
import uuid
from concurrent.futures import ProcessPoolExecutor
from utils.database import write_optimization, init_db
from utils.evaluation import compute_image_metrics, compute_wasserstein_metrics
from utils.image_utils import compute_synthetic_reconstruction
from utils.graph_utils import optimize_graph
from utils.topology_utils import get_pd_dgms_from_dist_matrix, get_pd_dgms_from_image, nx_graph_to_distance_matrix

def optimization_worker(args):
    image_entry, rsa_entry, optimization_entry_base = args

    with open(rsa_entry["rsa_graph_path"], "rb") as f:
        rsa_graph = pickle.load(f)

    with open(image_entry["target_graph_pd_path"], "rb") as f:
        target_graph_pds = pickle.load(f)    

    optimized_graph, optimization_entry = optimize_graph(rsa_graph, target_graph_pds, optimization_entry_base, backend="sinkhorn")
    optimization_entry["time_optimization_graph"] = optimize_graph.last_time
    with open(optimization_entry_base["optimized_graph_path"], "wb") as f:
        pickle.dump(optimized_graph, f)

    optimized_graph_dist_matrix = nx_graph_to_distance_matrix(optimized_graph)
    optimized_graph_pd = get_pd_dgms_from_dist_matrix(optimized_graph_dist_matrix)
    with open(optimization_entry_base["optimized_graph_pd_path"], "wb") as f:
        pickle.dump(optimized_graph_pd, f)

    reconstructed_image_shape = tuple(s * 4 for s in image_entry["downscaled_image_size"])
    optimized_reconstructed_image = compute_synthetic_reconstruction(optimized_graph, reconstructed_image_shape)
    optimization_entry["time_computation_reconstruction"] = compute_synthetic_reconstruction.last_time
    np.savez_compressed(optimization_entry_base["optimized_reconstructed_image_path"], optimized_reconstructed_image)

    optimized_reconstructed_image_pd = get_pd_dgms_from_image(optimized_reconstructed_image > 0)
    with open(optimization_entry_base["optimized_reconstructed_image_pd_path"], "wb") as f:
        pickle.dump(optimized_reconstructed_image_pd, f)

    metrics = {}
    metrics["image_metrics"] = compute_image_metrics(optimized_reconstructed_image)
    
    with open(image_entry["target_image_pd_path"], "rb") as f:
        target_image_pds = pickle.load(f)
    metrics["1-wasserstein_image_to_target"], metrics["normalized_1-wasserstein_image_to_target"] = compute_wasserstein_metrics(target_image_pds, optimized_reconstructed_image_pd, min_persistence=2)
    metrics["1-wasserstein_graph_to_target"], metrics["normalized_1-wasserstein_graph_to_target"] = compute_wasserstein_metrics(target_graph_pds, optimized_graph_pd)

    optimization_entry["metrics"] = metrics

    return image_entry["name"], rsa_entry["rsa_id"], optimization_entry

def main(study_base_folder, num_optimization_workers, initial_lr, num_steps, gamma, blur, base_study_name, study_name):

    LOGS_DIR = os.path.join(study_base_folder, "logs")
    DATA_DIR = os.path.join(study_base_folder, "data")
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    base_log_path = os.path.join(LOGS_DIR, f"{base_study_name}.json")
    log_path = os.path.join(LOGS_DIR, f"{study_name}.json")

    base_db_path = os.path.join(LOGS_DIR, f"{base_study_name}.db")
    study_db_path = os.path.join(LOGS_DIR, f"{study_name}.db")

    shutil.copy2(base_db_path, study_db_path)
    conn = sqlite3.connect(study_db_path)

    with open(base_log_path, "r") as f:
        study_log = json.load(f)

    optimization_jobs = []
    for image_name, image_entry in study_log["images"].items():
        
        image_dir = os.path.join(DATA_DIR, image_name)
        image_optim_dir = os.path.join(image_dir, "optimized")
        os.makedirs(image_optim_dir, exist_ok=True)

        for rsa_entry in image_entry["rsa_initializations"]:
        
            os.makedirs(os.path.join(image_optim_dir, rsa_entry["rsa_id"], "images"), exist_ok=True)
            os.makedirs(os.path.join(image_optim_dir, rsa_entry["rsa_id"], "graphs"), exist_ok=True)
            os.makedirs(os.path.join(image_optim_dir, rsa_entry["rsa_id"], "pd_images"), exist_ok=True)
            os.makedirs(os.path.join(image_optim_dir, rsa_entry["rsa_id"], "pd_graphs"), exist_ok=True)

            for dims in ([0], [1], [2], [0,1], [0,2], [1,2], [0,1,2]): #[0], [1], [2], [0,1], [0,2], [1,2], [0,1,2]
                optimization_id = str(uuid.uuid4())
                optimized_graph_path = os.path.join(image_optim_dir, rsa_entry["rsa_id"], "graphs", f"{optimization_id}.pkl")
                optimized_reconstructed_image_path = os.path.join(image_optim_dir, rsa_entry["rsa_id"], "images", f"{optimization_id}.npz")
                optimized_graph_pd_path = os.path.join(image_optim_dir, rsa_entry["rsa_id"], "pd_graphs", f"{optimization_id}.pkl")
                optimized_reconstructed_image_pd_path = os.path.join(image_optim_dir, rsa_entry["rsa_id"], "pd_images", f"{optimization_id}.pkl")

                optimization_entry_base = {
                    "optimization_id": optimization_id,
                    "optimized_graph_path": optimized_graph_path,
                    "optimized_reconstructed_image_path": optimized_reconstructed_image_path,
                    "optimized_graph_pd_path": optimized_graph_pd_path,
                    "optimized_reconstructed_image_pd_path": optimized_reconstructed_image_pd_path,
                    "hyperparameters": {
                        "num_steps": num_steps,
                        "initial_lr": initial_lr,
                        "gamma": gamma,
                        "pd_dims": dims,
                        "blur": blur
                    },
                    "loss_history": []
                }

                optimization_jobs.append((image_entry, rsa_entry, optimization_entry_base))

    with ProcessPoolExecutor(max_workers=num_optimization_workers) as executor:
        for image_name, rsa_id, optimization_entry in executor.map(optimization_worker, optimization_jobs):
            rsa_list = study_log["images"][image_name]["rsa_initializations"]
            for rsa_entry in rsa_list:
                if rsa_entry["rsa_id"] == rsa_id:
                    rsa_entry["optimization_runs"].append(optimization_entry)
                    break
            write_optimization(conn, image_id=image_name, rsa_id=rsa_id, optimization_entry=optimization_entry)

    conn.commit()
    conn.close()
    with open(log_path, "w") as f:
        json.dump(study_log, f, indent=2)

if __name__ == "__main__":
    
    study_base_folder = "" #Add path here

    #Optimization Hyperparameter
    num_optimization_workers = 8
    initial_lr = 5e-1
    num_steps = 500
    gamma = 0.990 #Gamma from Learning Rate Decay
    blur = 100 #Gamma from Regularized Wasserstein Distance

    base_study_name = "base_study_concrete" #Name of the study containing the synthetic images
    study_name = "study_concrete" #Name of the study then also containing the optimizations

    main(study_base_folder=study_base_folder, 
         num_optimization_workers=num_optimization_workers, 
         initial_lr=initial_lr, 
         num_steps=num_steps, 
         gamma=gamma, 
         blur=blur, 
         base_study_name=base_study_name, 
         study_name=study_name)