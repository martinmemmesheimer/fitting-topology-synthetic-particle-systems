import gudhi as gd
import gudhi.wasserstein
import numpy as np
from skimage.graph import MCP_Geometric
from skimage.segmentation import find_boundaries
from sklearn.metrics import f1_score
from utils.rsa import create_cylinder_domain

def compute_image_metrics(labeled_image):
    """
    Return a set of image metrics.

    Parameters
    ----------
    labeled_image : numpy.ndarray
        Labeled image.

    Returns
    -------
    dictionary
        Dictionary containing ``"volume_fraction"``, ``"num_particles"``, ``"particle_sizes"``, ``"particle_surface_area"``, ``"geometric_tortuosity"`` 
    """
    metrics = {}

    binary_image = labeled_image > 0

    metrics["volume_fraction"] = binary_image.sum() / create_cylinder_domain(binary_image.shape).sum()
    
    labels = np.unique(labeled_image)
    labels = labels[labels != 0]

    metrics["num_particles"] = len(labels)
    metrics["particle_sizes"] = (np.bincount(labeled_image.ravel())[labels]).tolist()

    boundaries = find_boundaries(labeled_image, mode="inner")
    metrics["particle_surface_area"] = (np.bincount(labeled_image[boundaries].ravel())[labels]).tolist()

    metrics["geometric_tortuosity"] = compute_geometric_tortuosity_mcp(binary_image, axis=0, crop=(300,1300))

    return metrics

def compute_wasserstein_metrics(target_pds, comparison_pds, min_persistence=0):
    """
    Return (normalized) Wasserstein distances between a set of target and comparison persistence diagrams.

    Parameters
    ----------
    target_pds : tuple[numpy.ndarray,numpy.ndarray,numpy.ndarray]
        Set of target / ground truth persistence diagrams computed from real data.
    comparison_pds : tuple[numpy.ndarray,numpy.ndarray,numpy.ndarray]
        Set of comparison persistence diagrams computed from synthetic data.
    min_persistence : int, optional
        Features in the persistence diagrams with smaller persistence than the specified value are removed. Default is ``0``.
    
    Returns
    -------
    tuple[dict, dict]
        Returns two dictionaries. The first contains the Wasserstein distances and the second the normalized Wasserstein distances.        
    """
    distances = {}
    normalized_distances = {}
    for dim in range(3):
        target_pd = target_pds[dim]
        comparison_pd = comparison_pds[dim]

        #Filter not for dimension 0 since average lifetime in that dimension is small
        if min_persistence != 0 and dim != 0:
            target_pd = target_pd[target_pd[:, 1] - target_pd[:, 0] > min_persistence]
            comparison_pd = comparison_pd[comparison_pd[:, 1] - comparison_pd[:, 0] > min_persistence]

        total_persistence_target = (np.sum((target_pd[np.isfinite(target_pd[:,1]), 1] - target_pd[np.isfinite(target_pd[:,1]), 0])**2) / 2)**0.5
        
        #Guards against 0-dim diagram with empty finite and matching essential parts to avoid the throwing of warnings in wasserstein_distance
        if (
            dim == 0
            and np.sum(np.isfinite(target_pd[:, 1])) == 0 
            and np.sum(np.isfinite(comparison_pd[:, 1])) == 0
            and np.sum(~np.isfinite(target_pd[:, 1])) == np.sum(~np.isfinite(comparison_pd[:, 1]))
        ):
            dist = 0
        else:
            dist = gd.wasserstein.wasserstein_distance(target_pd, comparison_pd, order=2, internal_p=2)
        
        distances[dim] = dist if np.isfinite(dist) else None

        if np.isfinite(dist) and total_persistence_target > 0:
            normalized_distances[dim] = dist / total_persistence_target
        else:
            normalized_distances[dim] = None

    return distances, normalized_distances

def compute_geometric_tortuosity_mcp(binary, axis=0, crop=(300, 1300)):
    """
    Return the geometric tortuosity along a specified axis.

    Parameters
    ----------
    binary : numpy.ndarray
        Binary image containing the particle system.
    axis : int, optional
        Axis along which the tortuosity is computed. Default is ``0``.
    crop : tuple[int, int], optional
        Specified slices between which the tortousity is computed. Default is ``(300, 1300)``.

    Returns
    dict
        Dictionary containing ``"mean"`` and ``"std"``. Mean is the geometric tortuosity and std is the standard deviation of the length of the individual paths.
    """

    binary = binary.astype(bool)

    binary_cropped = np.take(binary, range(crop[0], crop[1]), axis=axis)

    shape = binary_cropped.shape

    cost = np.where(binary_cropped, 1.0, np.inf)
    mcp = MCP_Geometric(cost)

    top_mask = np.take(binary_cropped, -1, axis=axis)
    top_points = np.argwhere(top_mask)
    top_coords = [tuple(np.insert(p, axis, -1)) for p in top_points]

    bottom_mask = np.take(binary_cropped, 0, axis=axis)
    bottom_points = np.argwhere(bottom_mask)
    bottom_coords = [tuple(np.insert(p, axis, 0)) for p in bottom_points]

    if top_points.size == 0 or bottom_points.size == 0:
        print(RuntimeWarning("Number of top or bottom inlet points is 0"))
        return np.inf

    costs_top, _ = mcp.find_costs(starts=top_coords)
    bottom_L_geos = np.take(costs_top, 0, axis=axis)[bottom_mask]
    bottom_L_geos = bottom_L_geos[np.isfinite(bottom_L_geos)]

    costs_bottom, _ = mcp.find_costs(starts=bottom_coords)
    top_L_geos = np.take(costs_bottom, -1, axis=axis)[top_mask]
    top_L_geos = top_L_geos[np.isfinite(top_L_geos)]

    if top_L_geos.size == 0 and bottom_L_geos.size==0:
        return np.inf  # no percolating path

    L_geos = np.concatenate([top_L_geos, bottom_L_geos])

    L_euclid = shape[axis] - 1

    tau_mean = np.mean((L_geos / L_euclid))
    tau_std = np.std((L_geos / L_euclid))

    tau = {"mean": tau_mean, "std": tau_std}

    return tau

def compute_dice_score(y_true, y_pred):
    """
    Return the dice score between two binary images or the mean dice score for labeled images.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground-truth target values (usually the real image).
    y_pred : numpy.ndarray
        Predicted target values (usually the reconstructed image).

    Returns
    -------
    float
        Dice score or mean dice score between y_true and y_pred depending on wether y_true is binary or not.
    """
    y_true = y_true.ravel().astype(int)
    y_pred = y_pred.ravel().astype(int)
    
    unique_labels = np.unique(y_true)
    unique_labels = unique_labels[unique_labels!=0]

    if len(unique_labels) > 1:
        dice_score = f1_score(y_true, y_pred, labels=unique_labels, average=None)
    else:
        dice_score = f1_score(y_true, y_pred)
    
    return dice_score