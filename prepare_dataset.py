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
    Supports both .nii and .nii.gz files, and forces output to .nii.gz.
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
    print(f"Found {len(sample_dirs)} sample directories in {source_dir}")

    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        new_sample_dir = target_path / sample_id
        new_sample_dir.mkdir(parents=True, exist_ok=True)

        found_files = {}
        # Collect all files in the current sample directory
        all_files = list(sample_dir.glob("*"))
        
        for file in all_files:
            filename = file.name.lower()
            
            # Identify extension and base name
            if filename.endswith(".nii.gz"):
                stem = file.name[:-7]
                is_nii = False
            elif filename.endswith(".nii"):
                stem = file.name[:-4]
                is_nii = True
            else:
                continue # Skip non-nifti files
            
            # Identify suffix by splitting on the LAST underscore
            parts = stem.split("_")
            suffix = parts[-1].lower()
            
            if suffix in mapping:
                new_suffix = mapping[suffix]
                new_filename = f"{sample_id}-{new_suffix}.nii.gz"
                target_file = new_sample_dir / new_filename
                
                if not target_file.exists():
                    if is_nii:
                        # Convert and compress .nii to .nii.gz
                        print(f"[{sample_id}] Compressing {file.name} -> {new_filename}")
                        img = nibabel.load(str(file))
                        nibabel.save(img, str(target_file))
                    else:
                        # Already .nii.gz, just copy
                        shutil.copy2(file, target_file)
                
                found_files[new_suffix] = f"{sample_id}/{new_filename}"

        # Validation and assembly
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
                print(f"Warning: Sample {sample_id} missing modalities: {missing}")
        else:
            # Check what WAS found to help debug
            found_keys = list(found_files.keys())
            print(f"Warning: Segmentation mask ('seg') missing for {sample_id}. Found: {found_keys}")

    # Shuffle and assign folds
    random.seed(42)
    random.shuffle(samples)
    for i, sample in enumerate(samples):
        sample['fold'] = i % num_folds

    dataset_json = {
        "training": samples
    }

    # Save the folds JSON
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
    parser.add_argument("--folds", type=int, default=5, help="Number of folds")
    args = parser.parse_args()

    source_dir = os.path.abspath(args.source)
    target_dir = os.path.abspath(args.target)

    prepare_dataset(source_dir, target_dir, args.folds)
