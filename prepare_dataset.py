import os
import argparse
import json
import shutil
import random
import nibabel
from pathlib import Path

def prepare_dataset(source_dir, target_dir, num_folds=5):
    """
    Converts BraTS formatted dataset to the format expected by this repository.
    Supports both .nii and .nii.gz files, but FORCES the output to be compressed .nii.gz.
    """
    
    # Mapping from common suffixes to repo expected suffixes
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
        # Look for both .nii and .nii.gz files
        for ext in ["*.nii", "*.nii.gz"]:
            for file in sample_dir.glob(ext):
                stem = file.name
                if stem.endswith(".nii.gz"):
                    suffix_to_strip = ".nii.gz"
                elif stem.endswith(".nii"):
                    suffix_to_strip = ".nii"
                
                # Identify suffix by splitting on underscore and removing extension
                parts = stem.replace(suffix_to_strip, "").split("_")
                suffix = parts[-1]
                
                if suffix in mapping:
                    new_suffix = mapping[suffix]
                    # FORCE exactly .nii.gz for downstream compatibility
                    new_filename = f"{sample_id}-{new_suffix}.nii.gz"
                    target_file = new_sample_dir / new_filename
                    
                    if not target_file.exists():
                        if file.suffix == ".nii":
                            # Convert and compress .nii to .nii.gz
                            print(f"Compressing {file.name} -> {new_filename}")
                            img = nibabel.load(str(file))
                            nibabel.save(img, str(target_file))
                        else:
                            # Already .nii.gz, just copy
                            shutil.copy2(file, target_file)
                    
                    found_files[new_suffix] = f"{sample_id}/{new_filename}"

        # Only add to training list if segmentation mask is present
        if 'seg' in found_files:
            required = ['t2f', 't1c', 't1n', 't2w']
            if all(m in found_files for m in required):
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
                missing = [m for m in required if m not in found_files]
                print(f"Warning: Missing modalities {missing} for {sample_id}, skipping.")
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
    parser = argparse.ArgumentParser(description="Prepare BraTS dataset for training.")
    parser.add_argument("--source", type=str, required=True, help="Path to raw dataset")
    parser.add_argument("--target", type=str, required=True, help="Path to store prepared dataset")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds (default: 5)")
    args = parser.parse_args()

    # Expand paths to be absolute
    source_dir = os.path.abspath(args.source)
    target_dir = os.path.abspath(args.target)

    prepare_dataset(source_dir, target_dir, args.folds)
