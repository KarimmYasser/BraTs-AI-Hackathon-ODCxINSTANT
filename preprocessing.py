import os
import shutil
import tqdm
import argparse

from multiprocessing import Pool
from pathlib import Path
from functools import partial

from biomedmbz_glioma.dataset_preprocessing import preprocessing_and_save , collect_cropped_image_sizes

def main():
    parser = argparse.ArgumentParser(description='Preprocess BraTS 2024 dataset for MedNeXt.')
    parser.add_argument('--source', type=str, required=True, help='Path to the source dataset directory.')
    parser.add_argument('--target', type=str, required=True, help='Path to the target preprocessed data directory.')
    parser.add_argument('--patch_size', type=int, nargs=3, default=None, 
                        help='Patch size as three integers (e.g., 128 128 128). If not provided, it will be calculated as the median.')
    parser.add_argument('--n_jobs', type=int, default=8, help='Number of parallel jobs for multiprocessing.')

    args = parser.parse_args()

    source_directory = args.source
    target_directory = args.target
    n_jobs = args.n_jobs
    
    print('Preprocessing:')
    print('Source dir:', source_directory)
    print('Target dir:', target_directory)
    
    if os.path.exists(target_directory) and os.path.isdir(target_directory):
        print(f"Removing existing target directory: {target_directory}")
        shutil.rmtree(target_directory)
    
    Path(target_directory).mkdir(parents=True, exist_ok=False)
    example_ids = os.listdir(source_directory)

    # Determine patch size
    if args.patch_size:
        median_patch_size = args.patch_size
        print(f'Using provided patch size: {median_patch_size}')
    else:
        print('Calculating median patch size (this may take a while)...')
        median_patch_size = collect_cropped_image_sizes(source_directory, example_ids)
        print(f'Calculated median patch size: {median_patch_size}')
    
    # Use partial to make the worker function picklable on Windows
    worker_fn = partial(
        preprocessing_and_save, 
        target_directory, 
        source_directory, 
        patch_size=median_patch_size
    )
    
    print(f"Starting preprocessing with {n_jobs} jobs...")
    with Pool(n_jobs) as p:
        r = list(tqdm.tqdm(p.imap(worker_fn, example_ids), total=len(example_ids)))
    
    print('Preprocessing complete.')

if __name__ == '__main__':
    main()
