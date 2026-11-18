import alphashape
import colorcet as cc
import os
import numpy as np
import pyvista as pv
import trimesh
import scipy.ndimage as ndi
from opensimplex import OpenSimplex
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import grey_dilation, binary_erosion, binary_closing
from scipy.stats import mode
from skimage.io import imread
from skimage.measure import marching_cubes, regionprops, label as sk_label
from skimage.morphology import remove_small_objects
from skimage.segmentation import find_boundaries, expand_labels, relabel_sequential
from sklearn.decomposition import PCA
from utils.timing_utils import timing

def load_images_from_scratch(image_path, min_particle_size=2000, close_boundaries=False, keep_only_largest=False):
    """
    Load an image from a TIF stack and perform some optional preprocessing steps.

    Parameters
    ----------
    image_path : string
        Path to the folder containing the TIF stack.
    min_particle_size : int, optional
        Minimum size of a connected component in the image. Connected components smaller than min_particle_size are removed. Default is ``2000``.
    close_boundaries : bool, optional
        If ``True``, a morphological closing with a ball of radius 1 is performed. Default is ``False``.
    keep_only_largest : bool, optional
        If ``True``, only the largest connected component in the image is kept. Default is ``False``.

    Returns
    -------
    numpy.ndarray
        Labeled image containing the particle system.
    
    """
    file_list = sorted([os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith(".tif")])
    labeled = np.stack([imread(f) for f in file_list], axis=0) #z,y,x

    clean_mask = remove_small_objects(labeled > 0, max_size=min_particle_size)
    labeled = np.where(clean_mask, labeled, 0)   

    if close_boundaries:
        binary_closed_img = binary_closing(labeled > 0, iterations=1)
        expanded_labels_img = expand_labels(labeled, distance=1)
        labeled = np.where(binary_closed_img, expanded_labels_img, 0)  

    if keep_only_largest:
        cc, _ = ndi.label(labeled > 0)
        component_sizes = np.bincount(cc.ravel())

        component_sizes[0] = 0 # ignore background
        largest_cc = np.argmax(component_sizes)
        largest_mask = (cc == largest_cc)

        labeled = np.where(largest_mask, labeled, 0)  

    labeled, _, _ = relabel_sequential(labeled)

    return labeled

def downsample_image_by_majority(img, downscale_by=4):
    """
    Downsample an image with a majority vote in each block.

    Parameters
    ----------
    img : numpy.ndarray
        Labeled image to scale.
    downscale_by : int, optional
        Factor by which the image is scaled down.

    Returns
    -------
    numpy.ndarray
        Labeled downscaled image.
    """
    Z, Y, X = img.shape
    Z2, Y2, X2 = Z//downscale_by, Y // downscale_by, X // downscale_by

    img = img[:Z2*downscale_by, :Y2*downscale_by, : X2*downscale_by]

    blocks = img.reshape(Z2, downscale_by, Y2, downscale_by, X2, downscale_by)
    blocks = blocks.transpose(0,2,4,1,3,5)
    blocks = blocks.reshape(-1, downscale_by**3)
    
    majority = mode(blocks, axis=1, keepdims=False).mode

    return majority.reshape(Z2, Y2, X2)

def add_simplex_noise(mesh, subdivide_iterations=2, scale=0.06, magnitude=7.0, seed=42):
    """
    Returns a subdivided mesh with simplex noise added to the vertices.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh representing the surface of the reconstructed particle.
    subdivide_iterations : int, optional
        Number of iterations for the subdivision of the mesh. Default is ``2``.
    scale : float, optional
        Scaling factor with which the positions of the vertices are scalled before evaluating the noise function. Default is ``0.06``.
    magnitude : float, optional
        Factor with which the generated noise is scaled. Default is ``7.0``.
    seed : int, optional
        Seed for the OpenSimplex generator. Default is ``42``.
    """
    noise_gen = OpenSimplex(seed)

    mesh = mesh.subdivide(iterations=subdivide_iterations)
    vertices = mesh.vertices
    normals = mesh.vertex_normals
    noise_values = []
    scale = scale
    for v in vertices:
        n = noise_gen.noise3(v[0] * scale, v[1] * scale, v[2] * scale)
        noise_values.append(magnitude * n)
    
    mesh.vertices = vertices + normals * np.array(noise_values)[:, None]

    return mesh

@timing
def compute_synthetic_reconstruction(G, img_size, alpha=None, add_noise=False, do_postprocess=True):
    """
    Compute a reconstruction of the image from a graph with either the convex hull or alphashapes and optional simplex noise or morphological closing.

    Parameters
    ----------
    G : networkx.Graph
        The particle graph containing node and edge features. Each node must have a ``'pos'`` attribute specifying its coordinates,
        and each edge must have ``'points'`` attribute containing a dictionary with the two boundary positions.
    img_size : tuple[int, int, int]
        Shape of the reconstructed image.
    alpha : float or None, optional
        Alpha value used for the reconstruction using alphashapes. If ``None`` then it defaults to a convex hull reconstruction. Default is ``None``.
    add_noise : Bool, optional
        Boolean value to decide wether simplex noise is added or not. Default is ``False``.
    do_postprocess : Bool, optional
        If ``true`` then a morphological closing with radius 1 is applied after the image is computed. Default is ``True``.

    Returns
    -------
    numpy.ndarray
        Labeled image containing the reconstructed particles.
    """
    reconstructed_img = np.zeros(img_size, dtype=np.uint16)

    label = 0
    for node, data in G.nodes(data=True):
        label += 1
        center = np.array(data['pos']).astype(int)
        points = [center]
        
        for neighbor in G[node]:
            points.append(G.edges[node, neighbor]['points'][node])
        
        points = np.unique(np.array(points, dtype=np.float32), axis=0)

        if points.shape[0] < points.shape[1] + 1:
            print("Not enough points to compute convex hull / alphashape in this dimensions.")
            continue

        if np.linalg.matrix_rank(points - points.mean(0)) < 3:
            print("The points are coplanar and the convex hull is therefore degenerate.")
            continue

        if alpha != None:
            mesh = alphashape.alphashape(points, alpha)
            mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        else:
            mesh = trimesh.convex.convex_hull(points)

        if add_noise:
            mesh = add_simplex_noise(mesh, subdivide_iterations=2, scale=0.06, magnitude=7)

        #mesh_scaled = trimesh.Trimesh(vertices=mesh.vertices * 4, faces=mesh.faces)

        voxelized = mesh.voxelized(pitch=1.0).fill()
        mask = voxelized.matrix.astype(np.uint32)
        filled_mask = mask
        #filled_mask = binary_fill_holes(mask)

        filled_mask = filled_mask * label

        bbox_min = np.floor(voxelized.translation).astype(int)

        region = reconstructed_img[bbox_min[0]:bbox_min[0]+mask.shape[0],
                   bbox_min[1]:bbox_min[1]+mask.shape[1],
                   bbox_min[2]:bbox_min[2]+mask.shape[2]]

        filled_mask = filled_mask[:region.shape[0], : region.shape[1], :region.shape[2]] #safety line if noise actually pushes a vertex out of the image bounds

        region[filled_mask > 0] = filled_mask[filled_mask > 0]

    if do_postprocess:
        binary_closed_img = binary_closing(reconstructed_img > 0, iterations=1)
        expanded_labels_img = expand_labels(reconstructed_img, distance=1)
        reconstructed_img = np.where(binary_closed_img, expanded_labels_img, 0)

    return reconstructed_img

def plot_particle_system(img, name):
    """
    Create and save a 3D Visualization using PyVista of a labeled particle system.

    Parameters
    ----------
    img : numpy.ndarray
        Labeled image containing the particle system.
    name : string
        Name or path under which the image should be saved.
    """
    palette = cc.glasbey[:128]
    regions = regionprops(img)

    merged_verts = []
    merged_faces = []
    merged_labels = []
    face_offset = 0

    for region in regions:
        min_z, min_y, min_x, max_z, max_y, max_x = region.bbox
        mask = img[min_z:max_z, min_y:max_y, min_x:max_x] == region.label
        mask_padded = np.pad(mask, 1, mode="constant")
        try:
            verts, faces, _, _ = marching_cubes((mask_padded).astype(float), level=0.5)
        except ValueError:
            print("Couldnt draw one particle.")
            continue
        verts = verts[:, [2,1,0]]  # z,y,x -> x,y,z for PyVista
        verts[:, 0] += min_x - 1
        verts[:, 1] += min_y - 1
        verts[:, 2] += min_z - 1

        merged_verts.append(verts)
        merged_labels.append(np.full(verts.shape[0], region.label, dtype=np.int32))

        faces = np.hstack([np.full((faces.shape[0],1),3), faces + face_offset])
        merged_faces.append(faces)
        face_offset += verts.shape[0]
    
    merged_verts = np.vstack(merged_verts)
    merged_faces = np.vstack(merged_faces)
    merged_labels = np.concatenate(merged_labels)

    mesh = pv.PolyData(merged_verts, merged_faces)
    mesh["label"] = merged_labels

    p = pv.Plotter(window_size=[1080, 1920])
    p.add_mesh(mesh, scalars="label", categories=True, cmap=palette, show_scalar_bar=False)

    p.reset_camera()
    p.view_isometric()
    p.camera.Zoom(2)
    p.show(screenshot=f"{name}.png")
