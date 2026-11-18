import numpy as np
import os
import pickle
from scipy.ndimage import binary_closing
from skimage.io import imread
from skimage.measure import label as sk_label
from skimage.morphology import remove_small_objects
from utils.evaluation import compute_wasserstein_metrics, compute_dice_score
from utils.image_utils import compute_synthetic_reconstruction, load_images_from_scratch
from utils.topology_utils import get_pd_dgms_from_image

def main(data_names, image_folder_base, graph_folder_base, output_name):

    reconstructions_dict = {}

    for name in data_names:
        print("=" * 80)
        print(f"Image: {name}")
        print("=" * 80)

        pkl_file = os.path.join(graph_folder_base, f"{name}.pkl")
        with open(pkl_file, "rb") as f:
            G = pickle.load(f)

        image_folder_path = os.path.join(image_folder_base, name)
        orig_img = load_images_from_scratch(image_folder_path, min_particle_size=2000, close_boundaries=True, keep_only_largest=True)

        reconstructed_img = compute_synthetic_reconstruction(G, orig_img.shape)
        reconstructions_dict[f"reconstruction_{name}"] = reconstructed_img

        print(f"Original foreground pixels: {np.sum(orig_img > 0)}")
        print(f"Reconstructed foreground pixels: {np.sum(reconstructed_img > 0)}")

        dice_scores = compute_dice_score(orig_img, reconstructed_img)
        mean = np.mean(dice_scores)
        std = np.std(dice_scores)
        print(f"Mean Dice score {mean:.3f} +- {std:.3f}")

        reconstructed_img_binary = (reconstructed_img != 0)
        orig_img_binary = (orig_img != 0)

        dice = compute_dice_score(orig_img_binary, reconstructed_img_binary)
        print(f"Dice score on the full image: {dice:.3f}")

        orig_pds = get_pd_dgms_from_image(orig_img_binary)
        reconstructed_pds = get_pd_dgms_from_image(reconstructed_img_binary)

        print("Min persistence for (1/2 Dim): 2")
        wasserstein_metrics, normalized_wasserstein_metrics = compute_wasserstein_metrics(orig_pds, reconstructed_pds, min_persistence=2)
        print(f"Wasserstein metrics of {name}: {wasserstein_metrics}")
        print(f"Normalized Wasserstein metrics of {name}: {normalized_wasserstein_metrics}")

        print("\n")

    np.savez_compressed(os.path.join(image_folder_base, output_name), **reconstructions_dict)

if __name__ == "__main__":
    data_names = ["image_name_1", "image_name_2"] #Replace by actual image names
    image_folder_base = "" 
    graph_folder_base = ""
    output_name = ""
    main(data_names, image_folder_base, graph_folder_base, output_name)