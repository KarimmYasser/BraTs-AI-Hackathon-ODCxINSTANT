import os
import gzip
import shutil
from multiprocessing import Pool, cpu_count

root_dir = "D:/Projects/BioMbz-Optimizing-Brain-Tumor-Segmentation-with-MedNeXt-BraTS-2024-SSA-and-Pediatrics/data-v2/archive/prepared_data"  # CHANGE THIS

def compress_nii(nii_path):
    gz_path = nii_path + ".gz"
    if os.path.exists(gz_path):
        return  # already compressed

    with open(nii_path, "rb") as f_in:
        with gzip.open(gz_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Optional: remove original file
    os.remove(nii_path)

    return nii_path

def collect_nii_files(root):
    nii_files = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.endswith(".nii") and not f.endswith(".nii.gz"):
                nii_files.append(os.path.join(r, f))
    return nii_files

if __name__ == "__main__":
    nii_files = collect_nii_files(root_dir)
    print(f"Found {len(nii_files)} files")

    workers = min(cpu_count(), 8)  # cap workers (important on Kaggle/Colab)
    with Pool(workers) as p:
        list(p.imap_unordered(compress_nii, nii_files))

    print("Compression done.")
