import cupy as cp
import json
import numpy as np
import os
import pickle
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from multiprocessing import Manager, Queue
from utils.database import init_db, write_image, write_rsa
from utils.evaluation import compute_image_metrics, compute_wasserstein_metrics
from utils.graph_utils import calculate_graph, upscale_graph
from utils.image_utils import compute_synthetic_reconstruction, downsample_image_by_majority, load_images_from_scratch
from utils.topology_utils import get_pd_dgms_from_dist_matrix, get_pd_dgms_from_image, nx_graph_to_distance_matrix

def image_worker(args):

    image_source_folder, graph_source_folder, image_name, DATA_DIR, downscale_by = args

    os.makedirs(os.path.join(DATA_DIR, "target_data", "images"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "target_data", "downscaled_images"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "target_data", "particle_masks"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "target_data", "graphs"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "target_data", "pd_images"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "target_data", "pd_graphs"), exist_ok=True)
    image_path = os.path.join(DATA_DIR, "target_data", "images", f"{image_name}.npz")
    downscaled_image_path = os.path.join(DATA_DIR, "target_data", "downscaled_images", f"{image_name}.npz")
    paired_particle_masks_path = os.path.join(DATA_DIR, "target_data", "particle_masks", f"{image_name}.pkl")
    target_graph_path = os.path.join(DATA_DIR, "target_data", "graphs", f"{image_name}.pkl")
    image_pd_path = os.path.join(DATA_DIR, "target_data", "pd_images", f"{image_name}.pkl")
    target_graph_pd_path = os.path.join(DATA_DIR, "target_data", "pd_graphs", f"{image_name}.pkl")

    image_source_path = os.path.join(image_source_folder, image_name)
    labeled_orig_image = load_images_from_scratch(image_source_path, min_particle_size=2000, close_boundaries=True, keep_only_largest=True)

    downscaled_image = downsample_image_by_majority(labeled_orig_image, downscale_by=downscale_by)

    np.savez_compressed(image_path, labeled_orig_image)
    np.savez_compressed(downscaled_image_path, downscaled_image)

    from utils.rsa import extract_paired_particle_masks
    paired_particle_masks = extract_paired_particle_masks(downscaled_image, labeled_orig_image)
    with open(paired_particle_masks_path, "wb") as f:
        pickle.dump(paired_particle_masks, f)
        
    with open(os.path.join(graph_source_folder, f"{image_name}.pkl"), "rb") as f:
        target_graph = pickle.load(f)        

    with open(target_graph_path, "wb") as f:
        pickle.dump(target_graph, f)

    target_image_pd = get_pd_dgms_from_image(labeled_orig_image > 0) 
    with open(image_pd_path, "wb") as f:
        pickle.dump(target_image_pd,f)

    target_graph_dist_matrix = nx_graph_to_distance_matrix(target_graph)
    target_downscaled_graph_pd = get_pd_dgms_from_dist_matrix(target_graph_dist_matrix)
    with open(target_graph_pd_path, "wb") as f:
        pickle.dump(target_downscaled_graph_pd, f)

    metrics = {}
    metrics["image_metrics"] = compute_image_metrics(labeled_orig_image)

    image_entry = {
        "name": image_name,
        "target_image_path": image_path,
        "target_downscaled_image_path": downscaled_image_path,
        "target_paired_particle_masks_path": paired_particle_masks_path,
        "target_graph_path": target_graph_path,
        "target_image_pd_path": image_pd_path,
        "target_graph_pd_path": target_graph_pd_path,
        "downscaled_image_size": downscaled_image.shape,
        "metrics": metrics,
        "rsa_initializations": []
    }

    return image_name, image_entry

def rsa_worker(args):
    from utils.rsa import  rsa_3d_cylinder_gpu_dual_flat, set_rsa_seed
    image_name, image_entry, gpu_queue, config = args
    set_rsa_seed(config["seed"])
    gpu_id = gpu_queue.get()
    try:
        with open(image_entry["target_paired_particle_masks_path"], "rb") as f:
            paired_particle_masks = pickle.load(f)

        downscaled_rsa_image, rsa_image = rsa_3d_cylinder_gpu_dual_flat(image_entry["downscaled_image_size"], paired_particle_masks, len(paired_particle_masks), scale_factor=config["scale_factor"], gpu_id = gpu_id)
        
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

        np.savez_compressed(config["downscaled_rsa_image_path"], downscaled_rsa_image)
        np.savez_compressed(config["rsa_image_path"], rsa_image)

    finally:
        gpu_queue.put(gpu_id)

    rsa_image_pd = get_pd_dgms_from_image(rsa_image > 0)
    with open(config["rsa_image_pd_path"], "wb") as f:
        pickle.dump(rsa_image_pd, f)

    downscaled_rsa_graph = calculate_graph(downscaled_rsa_image, edge_threshold=config["threshold"] / config["scale_factor"], num_threads=config["num_threads"])    
    rsa_graph = upscale_graph(downscaled_rsa_graph, scale_factor=config["scale_factor"])
    with open(config["rsa_graph_path"], "wb") as f:
        pickle.dump(rsa_graph, f)

    rsa_graph_dist_matrix = nx_graph_to_distance_matrix(rsa_graph)
    rsa_graph_pd = get_pd_dgms_from_dist_matrix(rsa_graph_dist_matrix)
    with open(config["rsa_graph_pd_path"], "wb") as f:
        pickle.dump(rsa_graph_pd, f)

    reconstructed_rsa_image = compute_synthetic_reconstruction(rsa_graph, rsa_image.shape)
    np.savez_compressed(config["reconstructed_rsa_image_path"], reconstructed_rsa_image)
    
    reconstructed_rsa_image_pd = get_pd_dgms_from_image(reconstructed_rsa_image > 0)
    with open(config["reconstructed_rsa_image_pd_path"], "wb") as f:
        pickle.dump(reconstructed_rsa_image_pd, f)

    metrics = {}
    reconstruction_metrics = {}

    metrics["image_metrics"] = compute_image_metrics(rsa_image)
    reconstruction_metrics["image_metrics"] = compute_image_metrics(reconstructed_rsa_image)
    
    with open(image_entry["target_image_pd_path"], "rb") as f:
        target_image_pds = pickle.load(f)
    metrics["1-wasserstein_image_to_target"], metrics["normalized_1-wasserstein_image_to_target"] = compute_wasserstein_metrics(target_image_pds, rsa_image_pd, min_persistence=2)
    reconstruction_metrics["1-wasserstein_image_to_target"], reconstruction_metrics["normalized_1-wasserstein_image_to_target"] = compute_wasserstein_metrics(target_image_pds, reconstructed_rsa_image_pd, min_persistence=2)

    with open(image_entry["target_graph_pd_path"], "rb") as f:
        target_graph_pds = pickle.load(f)
    metrics["1-wasserstein_graph_to_target"], metrics["normalized_1-wasserstein_graph_to_target"] = compute_wasserstein_metrics(target_graph_pds, rsa_graph_pd)

    reconstruction_entry = {
        "reconstruction_id": str(uuid.uuid4()),
        "reconstructed_rsa_image_path": config["reconstructed_rsa_image_path"],
        "reconstructed_rsa_image_pd_path": config["reconstructed_rsa_image_pd_path"],
        "time_computation_reconstruction": compute_synthetic_reconstruction.last_time,
        "metrics": reconstruction_metrics
    }

    rsa_entry = {
        "rsa_id": config["rsa_id"],
        "seed": config["seed"],
        "rsa_image_path": config["rsa_image_path"],
        "rsa_graph_path": config["rsa_graph_path"],
        "rsa_image_pd_path": config["rsa_image_pd_path"],
        "rsa_graph_pd_path": config["rsa_graph_pd_path"],
        "time_computation_image": rsa_3d_cylinder_gpu_dual_flat.last_time,
        "time_computation_graph": calculate_graph.last_time,
        "metrics": metrics,
        "reconstruction_entry": reconstruction_entry,
        "optimization_runs": []
    }

    return image_name, rsa_entry

def main(study_base_folder, image_names, image_source_folder, graph_source_folder, downscale_by, num_rsa_images, num_rsa_workers, NUM_GPUS, threshold, num_threads_graph_computation, study_name):

    LOGS_DIR = os.path.join(study_base_folder, "logs")
    DATA_DIR = os.path.join(study_base_folder, "data")
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    #GPU queue setup
    gpu_ids = list(range(NUM_GPUS))
    manager = Manager()
    gpu_queue = manager.Queue()
    for gpu_id in gpu_ids:
        gpu_queue.put(gpu_id)

    base_seed = 42
    rng = np.random.default_rng(base_seed)

    db_path = os.path.join(LOGS_DIR, f"{study_name}.db")
    conn = init_db(db_path)

    log_path = os.path.join(LOGS_DIR, f"{study_name}.json")
    study_log = {
        "base_study_name": study_name,
        "study_id": str(uuid.uuid4()),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "base_seed": base_seed,
            "image_names": image_names,
            "num_rsa_images": num_rsa_images,
            "downscale_by": downscale_by,
            "graph_hyperparameters": {
                "threshold": threshold,
                "num_threads": num_threads_graph_computation
            }
    },
        "images": {}
    }

    image_jobs = [(image_source_folder, graph_source_folder, image_name, DATA_DIR, study_log["config"]["downscale_by"]) for image_name in image_names]

    with ProcessPoolExecutor(max_workers=4) as executor:
        for image_name, image_entry in executor.map(image_worker, image_jobs):
            study_log["images"][image_name] = image_entry
            write_image(conn, image_id=image_name, image_entry=image_entry)

    rsa_jobs = []
    for image_name in image_names:
        image_entry = study_log["images"][image_name]

        image_dir = os.path.join(DATA_DIR, image_name)
        image_rsa_dir = os.path.join(image_dir, "rsa")
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(image_rsa_dir, exist_ok=True)
        os.makedirs(os.path.join(image_rsa_dir, "downscaled_images"), exist_ok=True)
        os.makedirs(os.path.join(image_rsa_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(image_rsa_dir, "graphs"), exist_ok=True)
        os.makedirs(os.path.join(image_rsa_dir, "pd_images"), exist_ok=True)
        os.makedirs(os.path.join(image_rsa_dir, "pd_graphs"), exist_ok=True)
        os.makedirs(os.path.join(image_rsa_dir, "reconstructed_images"), exist_ok=True)
        os.makedirs(os.path.join(image_rsa_dir, "pd_reconstructed_images"), exist_ok=True)
        
        for _ in range(num_rsa_images):
            seed = int(rng.integers(0, 2**32-1))
            rsa_id = str(uuid.uuid4())
            downscaled_rsa_image_path = os.path.join(image_rsa_dir, "downscaled_images", f"{rsa_id}.npz")
            rsa_image_path = os.path.join(image_rsa_dir, "images", f"{rsa_id}.npz")
            rsa_graph_path = os.path.join(image_rsa_dir, "graphs", f"{rsa_id}.pkl")
            rsa_image_pd_path = os.path.join(image_rsa_dir, "pd_images", f"{rsa_id}.pkl")
            rsa_graph_pd_path = os.path.join(image_rsa_dir, "pd_graphs", f"{rsa_id}.pkl")
            reconstructed_rsa_image_path = os.path.join(image_rsa_dir, "reconstructed_images", f"{rsa_id}.npz")
            reconstructed_rsa_image_pd_path = os.path.join(image_rsa_dir, "pd_reconstructed_images", f"{rsa_id}.pkl")

            config = {
                "rsa_id": rsa_id,
                "seed": seed,
                "downscaled_rsa_image_path": downscaled_rsa_image_path,
                "rsa_image_path": rsa_image_path,
                "rsa_graph_path": rsa_graph_path,
                "rsa_image_pd_path": rsa_image_pd_path,
                "rsa_graph_pd_path": rsa_graph_pd_path,
                "reconstructed_rsa_image_path": reconstructed_rsa_image_path,
                "reconstructed_rsa_image_pd_path": reconstructed_rsa_image_pd_path,
                "scale_factor": downscale_by,
                "threshold": threshold,
                "num_threads": num_threads_graph_computation
            }
            rsa_jobs.append((image_name, image_entry, gpu_queue, config))


    with ProcessPoolExecutor(max_workers=num_rsa_workers) as executor:
        for image_name, rsa_entry in executor.map(rsa_worker, rsa_jobs):
            study_log["images"][image_name]["rsa_initializations"].append(rsa_entry)
            write_rsa(conn, image_id=image_name, rsa_entry=rsa_entry)

    manager.shutdown()
    conn.commit()
    conn.close() 
    with open(log_path, "w") as f:
        json.dump(study_log, f, indent=2)

if __name__ == "__main__":

    study_base_folder = ""
    image_names = ["image_name_1", "image_name_2"]

    #Source data folder from which the data is loaded and then saved to study_base_folder
    image_source_folder = ""
    graph_source_folder = "target_graphs_thresh_400"
    
    downscale_by = 4
    num_rsa_images = 16
    
    #Parallelization Hyperparameters
    num_rsa_workers = 8
    NUM_GPUS = 4

    #Graph Hyperparameters
    threshold = 400
    num_threads_graph_computation = 4

    study_name = "base_study_concrete"

    main(study_base_folder=study_base_folder, 
         image_names=image_names, 
         image_source_folder=image_source_folder, 
         graph_source_folder=graph_source_folder, 
         downscale_by=downscale_by, 
         num_rsa_images=num_rsa_images, 
         num_rsa_workers=num_rsa_workers, 
         NUM_GPUS=NUM_GPUS, 
         threshold=threshold, 
         num_threads_graph_computation=num_threads_graph_computation, 
         study_name=study_name)
