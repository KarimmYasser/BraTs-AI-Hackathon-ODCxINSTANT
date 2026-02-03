import os
import shutil
import tqdm

from multiprocessing import Pool
from pathlib import Path

from biomedmbz_glioma.dataset_preprocessing import preprocessing_and_save , collect_cropped_image_sizes

n_jobs=8

if __name__ == '__main__':
    # ---------------------------------------------------------------------------
    source_directory=r"D:\\Projects\\BioMbz-Optimizing-Brain-Tumor-Segmentation-with-MedNeXt-BraTS-2024-SSA-and-Pediatrics\\prepared_dataset"
    target_directory=r"D:\\Projects\\BioMbz-Optimizing-Brain-Tumor-Segmentation-with-MedNeXt-BraTS-2024-SSA-and-Pediatrics\\preprocessed_data"
    
    print('Preprocessing:')
    print('Source dir:', source_directory)
    print('Target dir:', target_directory)
    
    if os.path.exists(target_directory) and os.path.isdir(target_directory):
        shutil.rmtree(target_directory)
    Path(target_directory).mkdir(parents=True, exist_ok=False)
    example_ids = os.listdir(source_directory)
    # get the patch size (median)
    # OPTION: You can manually set median_patch_size = [128, 128, 128] to skip the slow collection step
    median_patch_size = [128, 128, 128] # collect_cropped_image_sizes(source_directory, example_ids)
    print('Patch size:', median_patch_size)
    
    # Use partial to make the worker function picklable on Windows
    from functools import partial
    worker_fn = partial(
        preprocessing_and_save, 
        target_directory, 
        source_directory, 
        patch_size=median_patch_size
    )
    
    with Pool(n_jobs) as p:
        r = list(tqdm.tqdm(p.imap(worker_fn, example_ids), total=len(example_ids)))
    # ---------------------------------------------------------------------------
