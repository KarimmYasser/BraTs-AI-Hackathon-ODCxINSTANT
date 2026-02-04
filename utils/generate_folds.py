import os
import argparse
import json
import random
from pathlib import Path

def generate_folds(data_dir, output_path, num_folds=5):
    """
    Generates dataset_folds.json from a directory of prepared BraTS data.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Data directory {data_dir} does not exist.")
        return

    samples = []
    
    # Iterate over subdirectories in data_dir
    sample_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    print(f"Scanning {len(sample_dirs)} sample directories in {data_dir}")

    # Expected modalities based on prepare_dataset.py mapping
    modalities = ['t2f', 't1c', 't1n', 't2w']

    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        
        found_files = {}
        # Check for segmentation and modalities
        seg_file = sample_dir / f"{sample_id}-seg.nii.gz"
        if seg_file.exists():
            found_files['seg'] = f"{sample_id}/{seg_file.name}"
            
            for mod in modalities:
                mod_file = sample_dir / f"{sample_id}-{mod}.nii.gz"
                if mod_file.exists():
                    found_files[mod] = f"{sample_id}/{mod_file.name}"

        # Validation and assembly
        if 'seg' in found_files and all(m in found_files for m in modalities):
            samples.append({
                "label": found_files['seg'],
                "image": [found_files[m] for m in modalities],
                "fold": -1 # Placeholder
            })
        else:
            missing = [m for m in ['seg'] + modalities if m not in found_files]
            print(f"Warning: Sample {sample_id} is incomplete. Missing: {missing}")

    # Shuffle and assign folds
    random.seed(42)
    random.shuffle(samples)
    for i, sample in enumerate(samples):
        sample['fold'] = i % num_folds

    dataset_json = {
        "training": samples
    }

    # Save the folds JSON
    with open(output_path, "w") as f:
        json.dump(dataset_json, f, indent=4)
    
    print(f"\nSuccessfully processed {len(samples)} samples.")
    print(f"Dataset folds JSON saved to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset folds from prepared data.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to prepared dataset directory")
    parser.add_argument("--output", type=str, default="dataset_folds.json", help="Path to save output JSON")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds")
    args = parser.parse_args()

    generate_folds(args.data_dir, args.output, args.folds)
