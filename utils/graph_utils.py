import networkx as nx
import numpy as np
import torch
torch.set_num_threads(1)

from geomloss import SamplesLoss
from scipy.ndimage import binary_erosion
from skimage.measure import label as sk_label, regionprops
from skimage.morphology import remove_small_objects
from utils.timing_utils import timing
from utils.topology_utils import compute_sinkhorn_from_ph_opt, compute_wasserstein_from_ph_opt, pd_cost_matrix

#Import the custom cpp implementation
import line_interference


@timing
def calculate_graph(labeled, min_area=None, edge_threshold=400, num_threads=1):
    """
    Return the thresholded particle graph.

    Paramters
    ---------
    labeled : numpy.ndarray
        Labeled image containing the particle system.
    min_area : int or None, optional
        Minimal Area of a Particle to be included in the graph. Particles smaller are removed from labeled. Default is ``None``.
    edge_threshold : int, optional
        Threshold for the edge length in the graph. Default is ``400``.
    num_threads : int, optional
        Number of threads used in the cpp backend during the computation of the edges. Default is ``1``.

    Returns
    -------
    networkx.Graph
        Particle graph of the input image.
    """
    
    if min_area is not None:
        labeled = remove_small_objects(labeled, max_size=min_area)

    regions = regionprops(labeled)
    
    G = nx.Graph()
    centers = []
    borders = []
    region_labels = []
    particle_radii = []

    for region in regions:
    
        center = np.array(region.centroid)  # (z, y, x)
        centers.append(center)

        min_z, min_row, min_col, max_z, max_row, max_col = region.bbox
        sub_labeled = labeled[min_z:max_z, min_row:max_row, min_col:max_col]
        mask = sub_labeled == region.label
        border = mask ^ binary_erosion(mask)
        border_coords = np.column_stack(np.nonzero(border))
        border_coords[:, 0] += min_z
        border_coords[:, 1] += min_row
        border_coords[:, 2] += min_col
        borders.append(border_coords)

        radius = np.max(np.abs(border_coords - center).sum(axis=1))
        particle_radii.append(radius)

        region_labels.append(region.label)
        
    for i, center in enumerate(centers):
        G.add_node(i, pos=(center[0], center[1], center[2]))  # (z, y, x)

    edge_list = line_interference.line_interference_cpp.compute_edge_information_parallel(centers, borders, labeled, region_labels, particle_radii, edge_threshold, num_threads)
    #To understand the return values of compute_edge_information_parallel call "help(line_interference.line_interference_cpp.compute_edge_information_parallel)"
    for (i, j, dist, border_index_i, border_index_j) in edge_list:
        z0, y0, x0 = borders[i][border_index_i]
        z1, y1, x1 = borders[j][border_index_j]
        G.add_edge(i, j, weight=dist, points= {i: (z0, y0, x0), j: (z1, y1, x1)})            
    
    return G

def upscale_graph(graph, scale_factor):
    """
    Return a particle graph with rescaled features.

    Parameters
    ----------
    graph : networkx.Graph
        Particle Graph of a real or synthetic image.
    scale_factor : int
        Factor with which the ``"pos"`` node attribute and the ``"weight"``, ``"points"`` edge attributes are scaled.

    Returns
    -------
    networkx.Graph
        Particle graph with rescaled features.
    """
    upscaled_graph = nx.Graph()

    for n, data in graph.nodes(data=True):
        pos = data.get("pos")
        up_pos = tuple(p * scale_factor for p in pos)
        upscaled_graph.add_node(n, pos=up_pos)
    
    for i,j, data in graph.edges(data=True):
        weight = data.get("weight")
        points = data.get("points")

        up_weight = weight * scale_factor

        up_points = {}
        for key, point_coord in points.items():
            up_points[key] = tuple(p * scale_factor for p in point_coord)
        
        upscaled_graph.add_edge(i, j, weight=up_weight, points=up_points)

    return upscaled_graph

@timing
def optimize_graph(rsa_graph, target_graph_pds, optimization_log, backend="giotto-ph"):
    """
    Return a graph with optimized edge features. The optimization employs the Adam optimizer in a gradient descent scheme.

    Parameters
    ----------
    rsa_graph : networkx.Graph
        Particle graph computed on a synthetic image.
    target_graph_pds : tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        A set of target persistence diagrams for the homology dimensions 0,1,2.
    optimization_log : dict
        A dictionary containing another dict under the key ``"hyperparameters"`` containing Number of iterations ``"num_steps"``, 
        the initial learning rate ``"initial_lr"``, the decay factor for the exponential learning rate scheduler ``"gamma"``, 
        the dimensions included in the loss term ``"pd_dims"``, and the gamma / blur used in the sinkhorn loss ``"blur"``

    Returns
    -------
    tuple[networkx.Graph, dict]
        Optimized graph and the optimization log with added ``"loss_history"`` and ``"final_losses"``.
    """

    hyper = optimization_log["hyperparameters"]
    num_steps  = hyper["num_steps"]
    initial_lr = hyper["initial_lr"]
    gamma = hyper["gamma"]
    pd_dims = hyper["pd_dims"]
    blur = hyper["blur"]

    #Preprocess rsa_graph
    edge_features = []
    edge_indices = []

    node_id_map = {n: i for i, n in enumerate(rsa_graph.nodes)}

    for u, v, data in rsa_graph.edges(data=True):
        edge_indices.append([node_id_map[u], node_id_map[v]])
        ports = data['points']
        u_tensor = torch.tensor(ports[u], dtype=torch.float)
        v_tensor = torch.tensor(ports[v], dtype=torch.float)
        edge_features.append(torch.cat([u_tensor, v_tensor]))

    edge_indices = torch.tensor(edge_indices, dtype=torch.long).T  # [2, E]
    Z_true = torch.stack(edge_features)

    Z_true = Z_true.clone().detach()

    Z = Z_true.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([Z], lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    if backend == "sinkhorn":
        sinkhorn_loss = SamplesLoss(
            loss="sinkhorn",
            p=1,
            blur=blur,
            backend="tensorized",
            cost=pd_cost_matrix
        )

    for step in range(num_steps):

        optimizer.zero_grad()

        if backend == "giotto-ph":
            wasserstein_loss, wasserstein_loss_dict = compute_wasserstein_from_ph_opt(edge_indices, Z, target_graph_pds, dims=pd_dims)
        elif backend == "sinkhorn":
            wasserstein_loss, wasserstein_loss_dict = compute_sinkhorn_from_ph_opt(edge_indices, Z, target_graph_pds, sinkhorn_loss, dims=pd_dims)
        else:
            raise NotImplementedError(f"Backend {backend} is not implemented.")

        wasserstein_loss.backward()
        grad_norm = Z.grad.norm().item()

        optimizer.step()
        scheduler.step()

        optimization_log["loss_history"].append({
            "step": step,
            "total_loss": wasserstein_loss.item(),
            #"wasserstein_total": wasserstein_loss.item(),
            "wasserstein_individual": wasserstein_loss_dict,
            #"regularization": reg_loss.item(),
            "lr": optimizer.param_groups[0]["lr"],
            "grad_norm": grad_norm
        })

    optimization_log["final_losses"] = {
        "total_loss": wasserstein_loss.item(),
        "wasserstein_individual": wasserstein_loss_dict,
        #"regularization": reg_loss.item()
    }

    Z_optimized = Z.detach()
    
    optimized_graph = rsa_graph.copy()
    for (u, v), edge_feat in zip(optimized_graph.edges(), Z_optimized):
        ports_u = edge_feat[:3].tolist()
        ports_v = edge_feat[3:].tolist()
        optimized_graph.edges[u, v]['points'] = {
            u: ports_u,
            v: ports_v
        }
    
    return optimized_graph, optimization_log
