"""
Utility library to create and write the results of the experiments to a database.
"""

import numpy as np
import sqlite3
from collections.abc import Mapping

def init_db(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS images (
        image_id TEXT PRIMARY KEY,

        target_image_path TEXT,
        target_downscaled_image_path TEXT,
        target_paired_particle_masks_path TEXT,
        target_graph_path TEXT,
        target_image_pd_path TEXT,
        target_graph_pd_path TEXT,

        downscaled_shape TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS rsa_runs (
        rsa_id TEXT PRIMARY KEY,
        image_id TEXT,
        seed INTEGER,

        rsa_image_path TEXT,
        rsa_graph_path TEXT,
        rsa_image_pd_path TEXT,
        rsa_graph_pd_path TEXT,

        time_computation_image REAL,
        time_computation_graph REAL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS reconstructions (
        reconstruction_id TEXT PRIMARY KEY,
        rsa_id TEXT,
        image_id TEXT,

        reconstructed_image_path TEXT,
        reconstructed_image_pd_path TEXT,
        time_computation_reconstruction REAL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS optimization_runs (
        optimization_id TEXT PRIMARY KEY,
        rsa_id TEXT,
        image_id TEXT,

        num_steps INTEGER,
        initial_lr REAL,
        gamma REAL,
        pd_dims TEXT,
        blur INTEGER,

        optimized_graph_path TEXT,
        optimized_reconstructed_image_path TEXT,
        optimized_graph_pd_path TEXT,
        optimized_reconstructed_image_pd_path TEXT,
                
        time_optimization_graph REAL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS metrics (
        parent_image_id TEXT,
        entity_type TEXT,   -- "image", "rsa", "reconstruction", "optimization"
        entity_id TEXT,

        metric TEXT,
        submetric TEXT,
        value REAL,
                
        PRIMARY KEY (entity_type, entity_id, metric, submetric)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS distribution_values (
        parent_image_id TEXT,
        entity_type TEXT,   -- "image", "rsa", "reconstruction", "optimization"
        entity_id TEXT,
        metric TEXT,
        feature TEXT,
        idx INTEGER,
        value REAL,
                
        PRIMARY KEY (entity_type, entity_id, metric, feature, idx)
    )
    """)


    cur.execute("""
    CREATE TABLE IF NOT EXISTS optimization_history (
        optimization_id TEXT,
        step INTEGER,

        total_loss REAL,
        wasserstein_dim0 REAL,
        wasserstein_dim1 REAL,
        wasserstein_dim2 REAL,

        learning_rate REAL,
        grad_norm REAL,

        PRIMARY KEY (optimization_id, step)
    )
    """)

    conn.commit()
    return conn

def write_image(conn, image_id, image_entry):
    cur = conn.cursor()

    cur.execute("""
        INSERT OR REPLACE INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        image_id,
        image_entry["target_image_path"],
        image_entry["target_downscaled_image_path"],
        image_entry["target_paired_particle_masks_path"],
        image_entry["target_graph_path"],
        image_entry["target_image_pd_path"],
        image_entry["target_graph_pd_path"],
        str(image_entry["downscaled_image_size"])
    ))

    write_metrics(conn, image_id, "image", image_id, image_entry["metrics"])

    conn.commit()

def write_rsa(conn, image_id, rsa_entry):
    cur = conn.cursor()

    rsa_id = rsa_entry["rsa_id"]

    cur.execute("""
        INSERT INTO rsa_runs
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        rsa_id,
        image_id,
        rsa_entry["seed"],

        rsa_entry["rsa_image_path"],
        rsa_entry["rsa_graph_path"],
        rsa_entry["rsa_image_pd_path"],
        rsa_entry["rsa_graph_pd_path"],

        rsa_entry["time_computation_image"],
        rsa_entry["time_computation_graph"]
    ))

    write_metrics(
        conn,
        parent_image_id=image_id,
        entity_type="rsa",
        entity_id=rsa_id,
        metrics=rsa_entry["metrics"]
    )

    recon = rsa_entry["reconstruction_entry"]
    reconstruction_id = recon["reconstruction_id"]

    cur.execute("""
        INSERT INTO reconstructions
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        reconstruction_id,
        rsa_id,
        image_id,

        recon["reconstructed_rsa_image_path"],
        recon["reconstructed_rsa_image_pd_path"],
        recon["time_computation_reconstruction"]
    ))

    write_metrics(
        conn,
        parent_image_id=image_id,
        entity_type="reconstruction",
        entity_id=reconstruction_id,
        metrics=recon["metrics"]
    )

    conn.commit()

def write_optimization(
        conn,
        image_id,
        rsa_id,
        optimization_entry):

    cur = conn.cursor()

    opt_id = optimization_entry["optimization_id"]
    hp = optimization_entry["hyperparameters"]

    cur.execute("""
        INSERT INTO optimization_runs
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        opt_id,
        rsa_id,
        image_id,

        hp["num_steps"],
        hp["initial_lr"],
        hp["gamma"],
        str(hp["pd_dims"]),     # tuple/list → string
        hp["blur"],

        optimization_entry["optimized_graph_path"],
        optimization_entry["optimized_reconstructed_image_path"],
        optimization_entry["optimized_graph_pd_path"],
        optimization_entry["optimized_reconstructed_image_pd_path"],

        optimization_entry["time_optimization_graph"]
    ))

    write_metrics(
        conn,
        parent_image_id=image_id,
        entity_type="optimization",
        entity_id=opt_id,
        metrics=optimization_entry["metrics"]
    )

    cur.execute("""
        DELETE FROM optimization_history
        WHERE optimization_id=?
    """, (opt_id,))

    cur.executemany("""
        INSERT INTO optimization_history
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        (
            opt_id, 
            step_entry["step"], 
            step_entry["total_loss"], 
            step_entry["wasserstein_individual"].get(0), 
            step_entry["wasserstein_individual"].get(1), 
            step_entry["wasserstein_individual"].get(2), 
            step_entry["lr"], 
            step_entry["grad_norm"]
        )
        for step_entry in optimization_entry["loss_history"]
    ])

    conn.commit()

def write_metrics(conn, parent_image_id, entity_type, entity_id, metrics):
    cur = conn.cursor()

    def recurse(obj, path=()):
        """
        path = hierarchy of keys:
        ("image_metrics", "particle_sizes")
        """
        
        # scalar -> metrics table
        if obj is None or isinstance(obj, (int, float, np.number)):

            metric = path[0]
            submetric = "/".join(map(str, path[1:])) if len(path) > 1 else None

            cur.execute("""
                INSERT INTO metrics
                (parent_image_id, entity_type, entity_id, metric, submetric, value)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                parent_image_id,
                entity_type,
                entity_id,
                metric,
                submetric,
                float(obj) if obj is not None else None #None values will appear if Wasserstein distance is either infinity or total persistence is 0
            ))

        # distributions -> separate table
        elif isinstance(obj, (list, tuple, np.ndarray)):
            metric = path[0]
            feature = "/".join(map(str, path[1:])) if len(path) > 1 else None
            
            cur.execute("""
                DELETE FROM distribution_values
                WHERE parent_image_id=? AND entity_type=? AND entity_id=? AND metric=? AND feature=?
            """, (
                parent_image_id,
                entity_type,
                entity_id,
                metric,
                feature
            ))

            cur.executemany("""
                INSERT INTO distribution_values
                (parent_image_id, entity_type, entity_id, metric, feature, idx, value)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                (parent_image_id,
                 entity_type,
                 entity_id,
                 metric,
                 feature,
                 i,
                 float(v))
                for i, v in enumerate(obj)
            ])

        # nested dict -> recurse deeper
        elif isinstance(obj, Mapping):
            for k, v in obj.items():
                recurse(v, path + (k,))

    recurse(metrics)