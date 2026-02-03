import os
import argparse
import json
import shutil
import random
from pathlib import Path

def prepare_dataset(source_dir, target_dir, num_folds=5):
    """
    Converts BraTS 2021 formatted dataset to the format expected by this repository.
    
    Source format (BraTS 2021):
    sample_id/sample_id_flair.nii.gz
    sample_id/sample_id_t1.nii.gz
    sample_id/sample_id_t1ce.nii.gz
    sample_id/sample_id_t2.nii.gz
    sample_id/sample_id_seg.nii.gz
    
    Target format (BraTS 2024 style):
    sample_id/sample_id-t2f.nii.gz
    sample_id/sample_id-t1n.nii.gz
    sample_id/sample_id-t1c.nii.gz
    sample_id/sample_id-t2w.nii.gz
    sample_id/sample_id-seg.nii.gz
    """
    
    # Mapping from BraTS 2021 suffixes to repo expected suffixes
    mapping = {
        'flair': 't2f',
        't1'   : 't1n',
        't1ce' : 't1c',
        't2'   : 't2w',
        'seg'  : 'seg'
    }

    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    samples = []
    
    # Iterate over subdirectories in source_dir
    sample_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    print(f"Found {len(sample_dirs)} samples in {source_dir}")

    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        new_sample_dir = target_path / sample_id
        new_sample_dir.mkdir(parents=True, exist_ok=True)

        found_files = {}
        # Look for all nii.gz files in the sample directory
        for file in sample_dir.glob("*.nii.gz"):
            # Identify suffix by splitting on underscore
            # e.g., BraTS2021_00000_flair.nii.gz -> flair
            parts = file.name.replace(".nii.gz", "").split("_")
            suffix = parts[-1]
            
            if suffix in mapping:
                new_suffix = mapping[suffix]
                new_filename = f"{sample_id}-{new_suffix}.nii.gz"
                target_file = new_sample_dir / new_filename
                
                # Copy the file to the new location with the new name
                if not target_file.exists():
                    shutil.copy2(file, target_file)
                
                found_files[new_suffix] = f"{sample_id}/{new_filename}"

        # Only add to training list if segmentation mask is present
        if 'seg' in found_files:
            samples.append({
                "label": found_files['seg'],
                "image": [
                    found_files['t2f'],
                    found_files['t1c'],
                    found_files['t1n'],
                    found_files['t2w']
                ],
                "fold": -1 # Placeholder
            })
        else:
            print(f"Warning: Segmentation mask missing for {sample_id}, skipping.")

    # Shuffle and assign folds
    random.seed(42)
    random.shuffle(samples)
    for i, sample in enumerate(samples):
        sample['fold'] = i % num_folds

    dataset_json = {
        "training": samples
    }

    # Save the folds JSON in the current directory
    json_path = "dataset_folds.json"
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)
    
    print(f"\nSuccessfully prepared {len(samples)} samples.")
    print(f"Dataset folds JSON saved to: {os.path.abspath(json_path)}")
    print(f"Prepared data stored in: {target_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare BraTS 2021 dataset for training.")
    parser.add_argument("--source", type=str, required=True, help="Path to raw BraTS 2021 dataset (root folder containing sample IDs)")
    parser.add_argument("--target", type=str, required=True, help="Path to store prepared dataset")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds (default: 5)")
    args = parser.parse_args()

    # Expand paths to be absolute
    source_dir = os.path.abspath(args.source)
    target_dir = os.path.abspath(args.target)

    prepare_dataset(source_dir, target_dir, args.folds)
