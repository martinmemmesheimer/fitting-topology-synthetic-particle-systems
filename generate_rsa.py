import os
from utils.image_utils import load_images_from_scratch, downsample_image_by_majority, plot_particle_system
from utils.rsa import extract_paired_particle_masks, rsa_3d_cylinder_gpu_dual_flat
import numpy as np

def main(image_folder_base, image_name, downscale_by, output_folder, gpu_id):

    labeled_orig_image = load_images_from_scratch(os.path.join(image_folder_base, image_name), min_particle_size=2000, close_boundaries=True, keep_only_largest=True)

    downscaled_image = downsample_image_by_majority(labeled_orig_image, downscale_by=downscale_by)

    paired_particle_masks = extract_paired_particle_masks(downscaled_image, labeled_orig_image)

    syn_downscaled_image, syn_image = rsa_3d_cylinder_gpu_dual_flat(downscaled_image.shape, paired_particle_masks, len(paired_particle_masks), scale_factor=downscale_by, gpu_id=gpu_id)
    print(f"Computation Time: {rsa_3d_cylinder_gpu_dual_flat.last_time}")

    np.savez_compressed(os.path.join(output_folder, f"{image_name}_syn_downscaled.npz"), syn_downscaled_image)
    np.savez_compressed(os.path.join(output_folder, f"{image_name}_syn.npz"), syn_image)

    plot_particle_system(syn_image, os.path.join(output_folder, f"{image_name}_syn_rendered"))

if __name__ == "__main__":
    image_folder_base = ""
    image_name = ""
    downscale_by = 4
    output_folder = ""
    gpu_id = 0

    main(image_folder_base, image_name, downscale_by, output_folder)