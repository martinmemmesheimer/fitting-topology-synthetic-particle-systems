import os
import pickle
from utils.image_utils import load_images_from_scratch
from utils.graph_utils import calculate_graph

def main(image_folder_base, image_name, edge_threshold, num_threads, graph_folder_base):

    img = load_images_from_scratch(os.path.join(image_folder_base, image_name), min_particle_size=2000, close_boundaries=True, keep_only_largest=True)

    print(f"Name: {image_name}, Threshold: {edge_threshold}")

    graph = calculate_graph(img, edge_threshold=edge_threshold, num_threads=num_threads)

    os.makedirs(graph_folder_base, exist_ok=True)
    with open(os.path.join(graph_folder_base, f"{image_name}.pkl"), "wb") as f:
        pickle.dump(graph, f)

if __name__ == "__main__":

    image_folder_base = f"" #Add path here
    image_name = ""
    edge_threshold = 400
    num_threads = 1
    graph_folder_base = f"target_graphs_thresh_{edge_threshold}"
    
    main(image_folder_base, image_name, edge_threshold, num_threads, graph_folder_base)
