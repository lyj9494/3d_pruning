import os
import json

def create_dataset_json(dataset_path, output_path):
    """
    Generates a JSON file describing the dataset structure.

    Args:
        dataset_path (str): The path to the demo_dataset directory.
        output_path (str): The path to save the output JSON file.
    """
    dataset_entries = []
    if not os.path.isdir(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' not found.")
        return

    for obj_dir in sorted(os.listdir(dataset_path)):
        obj_path = os.path.join(dataset_path, obj_dir)
        if os.path.isdir(obj_path):
            # Check for required files
            points_file = os.path.join(obj_path, 'points.npy')
            image_file = os.path.join(obj_path, 'rendering_rmbg.png')

            if os.path.exists(points_file) and os.path.exists(image_file):
                entry = {
                    "file": f"{obj_dir}.glb",
                    "mesh_path": f"assets/objects/{obj_dir}.glb",
                    "surface_path": f"preprocessed_data/{obj_dir}/points.npy",
                    "image_path": f"preprocessed_data/{obj_dir}/rendering_rmbg.png",
                }
                dataset_entries.append(entry)
            else:
                print(f"Skipping directory {obj_dir} as it's missing required files.")


    with open(output_path, 'w') as f:
        json.dump(dataset_entries, f, indent=4)

    print(f"Successfully created JSON file at '{output_path}'")

if __name__ == "__main__":
    demo_dataset_path = '/workspace/luoyajing/3d_pruning/demo_dataset'
    output_json_path = '/workspace/luoyajing/3d_pruning/demo_dataset.json'
    create_dataset_json(demo_dataset_path, output_json_path)

