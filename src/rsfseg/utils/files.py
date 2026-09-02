import os
import shutil
from tqdm import tqdm


def copy_new_files(src, dst):
    """Copy data from one directory to another without overwriting 
    those that already exist.
    """
    # Collect all files to check
    files_to_copy = []
    for root, _, files in os.walk(src):
        rel_path = os.path.relpath(root, src)
        dest_root = os.path.join(dst, rel_path)
        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_root, file)
            if not os.path.exists(dest_file):
                files_to_copy.append((src_file, dest_file))

    # Copy with progress bar
    for src_file, dest_file in tqdm(files_to_copy, desc="Copying files", unit="file"):
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
        shutil.copy2(src_file, dest_file)
