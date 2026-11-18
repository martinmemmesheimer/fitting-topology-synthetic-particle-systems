import cupy as cp
import cupyx.scipy.ndimage as cp_ndi
import numpy as np
import random
from cupyx.scipy import ndimage as cnd
from itertools import permutations
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.segmentation import expand_labels
from utils.timing_utils import timing

def set_rsa_seed(seed=42):
    """
    Function to set a set for standard python, numpy and cupy.
    """
    random.seed(seed)
    np.random.seed(seed)
    cp.random.seed(seed)

def create_cylinder_domain(shape):
    """
    Return a binary mask of the maximal cylinder inside the shape. The height of the cylinder is in the 0-axis.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Shape of the output mask.
    
    Returns
    -------
    numpy.ndarray
        Binary mask being True inside the cylinder.

    """
    Z, H, W = shape
    yy, xx = np.mgrid[:H, :W]
    center = (H // 2, W // 2)
    radius = min(H, W) // 2
    circle_mask = ((yy - center[0])**2 + (xx - center[1])**2) <= radius**2

    cylinder_mask = np.repeat(circle_mask[np.newaxis, :, :], Z, axis=0)
    return cylinder_mask.astype(np.uint8)

@timing
def rsa_3d_cylinder_gpu_dual_flat(
    container_shape_small,
    particle_masks,  # list of (mask_small, mask_large)
    num_particles_to_place,
    scale_factor,
    gpu_id=0,
    max_batch_attempts=2400,
    batch_size=5000,
):
    """
    Generate a synthetic image in a downscaled and full resolution with the RSA Algorithm on the GPU. 
    
    Parameters
    ----------
    container_shape_small : tuple[int, int, int]
        Shape of the downscaled generated image
    particle_masks : list(tuple[numpy.ndarray, numpy.ndarray])
        List of pairs of particle masks for the downscaled and full resolution.
    num_particles_to_place :
        Number of particles to be placed from the list. If all particles should be placed set it to ``len(particle_masks)``.
    scale_factor : int
        Factor of how much larger the full resolution is to the downscaled resolution.
    gpu_id : int, optional
        Id of the GPU which should be used. Default is ``0``. 
    max_batch_attemps : int
        The number of batches the gpu runs to find a valid position for the particle. 
        In every batch ``batch_size`` many positions are tested. Default is ``2400``.
    batch_size : int
        The number of positions tested in a batch. Default is ``5000``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Tuple of two 3D numpy arrays. One with shape container_shape_small and one with the container_shape_small multiplied by scale_factor in every axis.
    """

    cp.cuda.Device(gpu_id).use()

    container_small = cp.zeros(container_shape_small, dtype=cp.uint32)

    domain_mask_small = cp.array(create_cylinder_domain(container_shape_small), dtype=cp.uint16)

    container_small[domain_mask_small == 0] = num_particles_to_place + 1

    D_small, H_small, W_small = container_shape_small

    container_small_flat = container_small.ravel()

    container_shape_large = tuple(
        s * scale_factor for s in container_shape_small
    )

    container_large = np.zeros(container_shape_large, dtype=np.uint32)

    selected = random.sample(
        particle_masks,
        min(num_particles_to_place, len(particle_masks))
    )

    sizes = [m[0].sum() for m in selected]
    order = np.argsort(sizes)[::-1]
    selected_sorted = [selected[i] for i in order]

    label = 1

    for mask_small, mask_large in selected_sorted:

        placed = False

        for perm in permutations([0, 1, 2]):

            mask_small_gpu_perm = cp.array(
                mask_small.transpose(perm),
                dtype=cp.uint32
            )

            mask_large_perm = mask_large.transpose(perm)

            mask_coords = cp.argwhere(mask_small_gpu_perm > 0).astype(cp.uint32)

            mask_offsets = (
                mask_coords[:, 0] * (H_small * W_small)
                + mask_coords[:, 1] * W_small
                + mask_coords[:, 2]
            )

            z_range = D_small - mask_small_gpu_perm.shape[0]
            y_range = H_small - mask_small_gpu_perm.shape[1]
            x_range = W_small - mask_small_gpu_perm.shape[2]

            valid_domain = domain_mask_small[:z_range, :y_range, :x_range]
            z,y,x = cp.nonzero(valid_domain)
            candidate_flat = (
                z * (H_small * W_small)
                + y * W_small
                + x
            )

            cp.random.shuffle(candidate_flat)

            for i in range(max_batch_attempts):

                start = batch_size * i
                end = min(start + batch_size, candidate_flat.shape[0])

                if start >= candidate_flat.shape[0]:
                    break

                batch = candidate_flat[start:end]

                indices = (batch[:, None] + mask_offsets[None, :])

                values = container_small_flat[indices]

                valid_idx = cp.where(cp.all(values == 0, axis=1))[0]

                if len(valid_idx) > 0:

                    chosen = int(cp.random.choice(valid_idx, size=1)[0])

                    pos_flat_small = int(batch[chosen])

                    z = pos_flat_small // (H_small * W_small)
                    rem = pos_flat_small % (H_small * W_small)
                    y = rem // W_small
                    x = rem % W_small

                    pos_corner_small = (int(z), int(y), int(x))

                    place_indices = (pos_flat_small + mask_offsets)
                    container_small_flat[place_indices] = label

                    offset_small = tuple((s - 1) / 2 for s in mask_small_gpu_perm.shape)
                    pos_center_small = tuple(p + o for p, o in zip(pos_corner_small, offset_small))

                    pos_center_large = tuple(p * scale_factor for p in pos_center_small)

                    offset_large = tuple((s - 1) // 2 for s in mask_large_perm.shape)
                    pos_corner_large = tuple(int(p - o) for p, o in zip(pos_center_large, offset_large))

                    slices_large = tuple(slice(int(min(max(p, 0), container_shape_large[i] - s)), int(min(max(p, 0) + s, container_shape_large[i]))) 
                                         for i, (p, s) in enumerate(zip(pos_corner_large, mask_large_perm.shape)))

                    container_large[slices_large] = np.where(
                        mask_large_perm > 0,
                        label,
                        container_large[slices_large]
                    )

                    label += 1
                    placed = True
                    break

            if placed:
                break

    container_small = container_small_flat.reshape(container_shape_small)

    container_small[domain_mask_small == 0] = 0

    container_small_np = cp.asnumpy(container_small)

    binary_closed_img = ndi.binary_closing(container_small_np > 0, iterations=1)
    expanded_labels_img = expand_labels(container_small_np, distance=1)
    container_small_np = np.where(binary_closed_img, expanded_labels_img, 0)

    binary_closed_img = ndi.binary_closing(container_large > 0, iterations=1)
    expanded_labels_img = expand_labels(container_large, distance=1)
    container_large = np.where(binary_closed_img, expanded_labels_img, 0)

    return container_small_np, container_large

def extract_particle_masks(labeled_img):
    """
    Return a list of label masks for a labeled image. The masks are tightly cropped to their bounding box.

    Parameters 
    ----------
    labeled_img : numpy.ndarray
        Labeled image of a particle system.
    
    Returns
    -------
    list(numpy.ndarray)
        List of label masks.

    """
    masks = []

    for region in regionprops(labeled_img):
        mask = (labeled_img[region.slice] == region.label)
        masks.append(mask)
    return masks

def extract_paired_particle_masks(downscaled_img, img):
    """
    Return a list of pairs of label masks for two images. The masks are tightly cropped to their bounding box.

    Parameters 
    ----------
    downscaled_img : numpy.ndarray
        A downscaled version of the labeled image. Usually it is obtained by the function utils.image_utils.downsample_by_majority_vote.
    labeled_img : numpy.ndarray
        Labeled image of a particle system.
    
    Returns
    -------
    list(numpy.ndarray)
        List of pairs of label masks.

    """
    paired_masks = []

    for downscaled_region, region in zip(regionprops(downscaled_img), regionprops(img)):
        if downscaled_region.label != region.label:
            RuntimeError("Labels do not coincide when extracting particle masks.")
        downscaled_mask = (downscaled_img[downscaled_region.slice] == downscaled_region.label)
        mask = (img[region.slice] == region.label)
        paired_masks.append((downscaled_mask, mask))
    return paired_masks
