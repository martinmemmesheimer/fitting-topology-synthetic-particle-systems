import cripser
import gudhi as gd
import gudhi.wasserstein
import numpy as np
import torch
from ripser import ripser
from scipy.ndimage import distance_transform_cdt
from utils.ph_opt import RipsPH

def edgeports_to_distance_matrices(edge_indices, edge_pred):
    """
    Compute the distance matrix from the edge features of a particle graph given as a tensor.

    Parameters
    ----------
    edge_indices : torch.tensor
        Tensor of shape (|E|, 2) containing the edges of the graph.
    edge_pred : torch.tensor
        Tensor of shape (|E|, 6) containing the edge features of the graph.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Distance matrix and a tensor containing the distances in the order of edge indices.
    """
    
    max_idx = int(torch.max(edge_indices)) + 1

    device = edge_pred.device
    dtype = edge_pred.dtype
    
    edge_indices = edge_indices.to(device)
    
    u_ports = edge_pred[:, :3]
    v_ports = edge_pred[:, 3:]
    
    dist = torch.abs(u_ports - v_ports).sum(dim=1)
    
    with torch.no_grad():

        dist_mat_flat = torch.full((max_idx * max_idx,), 1e8, device=device, dtype=dtype)

        rows, cols = edge_indices[0, :], edge_indices[1, :]
        indices_flat = rows * max_idx + cols
        indices_flat_sym = cols * max_idx + rows

        dist_mat_flat = dist_mat_flat.scatter(0, indices_flat.long(), dist.detach())
        dist_mat_flat = dist_mat_flat.scatter(0, indices_flat_sym.long(), dist.detach())

        dist_mat = dist_mat_flat.view(max_idx, max_idx)

        dist_mat.fill_diagonal_(0)
        
    return dist_mat, dist

def nx_graph_to_distance_matrix(G, metric="manhattan"):
    """
    Compute the distance matrix for a particle graph.

    Parameters
    ----------
    G : networkx.Graph
        Particle graph with ``'points'`` edge attributes.
    metric : string, optional
        Metric for the distance computation. Currently only ``'manhattan'`` is implemented.
    
    Returns
    -------
    numpy.ndarray
        Distance matrix of the input graph.
    """
    nodes = list(G.nodes)
    node_idx = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)
    
    dist_matrix = np.full((n, n), np.inf)  # No edge is infinity
    np.fill_diagonal(dist_matrix, 0)

    for u, v, data in G.edges(data=True):
        i, j = node_idx[u], node_idx[v]
        pos_u = np.array(data['points'][u])
        pos_v = np.array(data['points'][v])

        if metric == "manhattan":
            dist = np.sum(np.abs(pos_u - pos_v))
        else:
            raise ValueError(f"Unknown metric: {metric}")

        dist_matrix[i, j] = dist
        dist_matrix[j, i] = dist  #Symmetrie of undirected graph

    return dist_matrix

def get_pd_dgms_from_dist_matrix(dist_matrix_np):
    """
    Compute the persistence diagrams from a distance matrix using ripser.

    Parameters
    ----------
    dist_matrix_np : numpy.ndarray
        Distance matrix for which the persistence diagrams are computed. ``np.inf`` is not allowed as an entry. Instead set
        the distance to 1e8. The function prevents these simplices to be added to the filtration.
    
    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Persistence diagrams for the homology dimensions 0,1,2.   
    """
    result = ripser(dist_matrix_np, distance_matrix=True, maxdim=2, thresh=1e7)
    pd0 = result['dgms'][0]
    pd1 = result['dgms'][1]
    pd2 = result['dgms'][2]

    return pd0, pd1, pd2

def get_pd_dgms_from_image(binary_image):
    """
    Compute the persistence diagrams of a binary image using a l1-distance transform and cripser.

    Parameters
    ----------
    binary_image : numpy.ndarray
        Binary input image for which the persistence diagrams is computed.
    
    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Persistence diagrams for the homology dimensions 0,1,2.
    """
    
    dist_transform = distance_transform_cdt(1 - binary_image, metric='taxicab').astype(np.float32)

    pd = cripser.compute_ph(dist_transform, maxdim=3, filtration="V")

    pd0 = pd[pd[:, 0] == 0][:, 1:3]
    pd1 = pd[pd[:, 0] == 1][:, 1:3]
    pd2 = pd[pd[:, 0] == 2][:, 1:3]   

    return pd0, pd1, pd2

def compute_wasserstein_from_ph_opt(edge_indices, Z, target_diagrams, dims=[0,1,2]):
    """
    Compute the Wasserstein loss term using Gudhi and the ph_opt package as the backend.

    Parameters
    ----------
    edge_indices : torch.tensor
        Tensor of shape (|E|, 2) containing the edges of the graph.
    Z : torch.tensor
        Tensor of shape (|E|, 6) containing the edge features of the graph.
    target_diagrams : tuple[torch.tensor, torch.tensor, torch.tensor]
        Persistence Diagrams for the homology dimensions 0,1,2. All have to be included even if they are not in dims.
    sinkhorn_loss : SamplesLoss
        Configured GeomLoss Sinkhorn loss object used to compute the regularized OT distance.
    dims : list[int], optional
        Dimensions included in the regularized loss. Default is ``[0,1,2]``.

    Returns
    -------
    tuple[int, dict]
        Value of the Loss and a dictionary recording the values for the individual dimensions.
    """

    combined_wasserstein_loss = 0
    wasserstein_loss_dict = {}

    dist_matrix, dist_sparse = edgeports_to_distance_matrices(edge_indices, Z)
    rips_ph = RipsPH(dist_matrix, dist_sparse, edge_indices, maxdim=max(dims), distance_matrix=True)
    
    for dim in dims:
        pred_diagram = rips_ph.get_differentiable_barcode(dim)

        wasserstein_dim_loss = gd.wasserstein.wasserstein_distance(pred_diagram, torch.tensor(target_diagrams[dim]), order=2, internal_p=2, enable_autodiff=True, keep_essential_parts=False)
        combined_wasserstein_loss += wasserstein_dim_loss
        wasserstein_loss_dict[dim] = wasserstein_dim_loss.item()

    return combined_wasserstein_loss, wasserstein_loss_dict

def pd_cost_matrix(x, y):
    """
    Create the cost matrix used for the Sinkhorn loss computation.

    Parameters
    ----------

    x, y: torch.Tensor
        Tensors of size (B, N+1, 2) and (B, M+1, 2) containing the two diagrams with added virtual diagonal point.
    
    Returns
    -------
    torch.tensor
        Cost matrix of size (B, N+1, M+1).
    """

    C = torch.cdist(x[:, :-1, :], y[:, :-1, :], p=2)**2

    y_diag = ((y[:, :-1, 1] - y[:, :-1, 0])**2 / 2).unsqueeze(1)
    x_diag = ((x[..., 1] - x[..., 0])**2 / 2).unsqueeze(2)

    C = torch.cat([ torch.cat([C, y_diag], dim=1), x_diag], dim=2)

    return C

def compute_sinkhorn_from_ph_opt(edge_indices, Z, target_diagrams, sinkhorn_loss, dims=[0,1,2]):
    """
    Compute the regularized Wasserstein loss term using Geomloss and the ph_opt package as the backend.

    Parameters
    ----------
    edge_indices : torch.tensor
        Tensor of shape (|E|, 2) containing the edges of the graph.
    Z : torch.tensor
        Tensor of shape (|E|, 6) containing the edge features of the graph.
    target_diagrams : tuple[torch.tensor, torch.tensor, torch.tensor]
        Persistence Diagrams for the homology dimensions 0,1,2. All have to be included even if they are not in dims.
    sinkhorn_loss : SamplesLoss
        Configured GeomLoss Sinkhorn loss object used to compute the regularized OT distance.
    dims : list[int], optional
        Dimensions included in the regularized loss. Default is ``[0,1,2]``.

    Returns
    -------
    tuple[int, dict]
        Value of the Loss and a dictionary recording the values for the individual dimensions.
    """

    combined_wasserstein_loss = 0
    wasserstein_loss_dict = {}

    diag_point = torch.tensor([0,0]).unsqueeze(0)

    dist_matrix, dist_sparse = edgeports_to_distance_matrices(edge_indices, Z)
    rips_ph = RipsPH(dist_matrix, dist_sparse, edge_indices, maxdim=max(dims), distance_matrix=True)
    
    for dim in dims:

        pred_diagram = rips_ph.get_differentiable_barcode(dim)
        
        #Prepare diagrams for the sinkhorn loss
        mask = torch.isfinite(pred_diagram[:, 1])
        pred_diagram = pred_diagram[mask]
        pred_diagram_ext = torch.cat([pred_diagram, diag_point], dim=0)

        target_diagram = torch.tensor(target_diagrams[dim], dtype=pred_diagram.dtype)
        mask = torch.isfinite(target_diagram[:, 1])
        target_diagram = target_diagram[mask]
        target_diagram_ext = torch.cat([target_diagram, diag_point], dim=0)

        pred_diagram_weights = torch.ones(pred_diagram_ext.shape[0])
        pred_diagram_weights[-1] = target_diagram.shape[0]
        target_diagram_weights = torch.ones(target_diagram_ext.shape[0])
        target_diagram_weights[-1] = pred_diagram.shape[0]

        wasserstein_dim_loss = sinkhorn_loss(pred_diagram_weights, pred_diagram_ext, target_diagram_weights, target_diagram_ext)
        combined_wasserstein_loss += wasserstein_dim_loss
        wasserstein_loss_dict[dim] = wasserstein_dim_loss.item()

    return combined_wasserstein_loss, wasserstein_loss_dict

