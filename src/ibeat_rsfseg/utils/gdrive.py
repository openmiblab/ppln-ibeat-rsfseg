import os
import shutil
import time


from tqdm import tqdm


def copy(SOURCE_DIR, TARGET_DIR, BATCH_SIZE=500, DELAY_BETWEEN_BATCHES = 10):

    # === Step 1: Collect all files ===
    all_files = []
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, SOURCE_DIR)
            all_files.append((full_path, relative_path))

    total_files = len(all_files)
    print(f"📁 Found {total_files} files to copy.")

    # === Step 2: Create one global progress bar ===
    with tqdm(total=total_files, desc="Total Progress", unit="file") as progress:

        # Step 3: Copy files in batches
        for i in range(0, total_files, BATCH_SIZE):
            batch = all_files[i:i + BATCH_SIZE]
            print(f"\n📦 Copying batch {i // BATCH_SIZE + 1} ({len(batch)} files)...")

            for src_path, rel_path in batch:
                dest_path = os.path.join(TARGET_DIR, rel_path)
                dest_folder = os.path.dirname(dest_path)

                os.makedirs(dest_folder, exist_ok=True)

                try:
                    shutil.copy2(src_path, dest_path)
                    progress.update(1)  # ✅ Only update if copy succeeds
                except Exception as e:
                    print(f"❌ Failed to copy {src_path}: {e}")

            time.sleep(DELAY_BETWEEN_BATCHES)

    print("\n🎉 All files copied successfully.")









if __name__=='__main__':
    source_dir = os.path.join("C:\\Users", "md1jdsp", "Documents", 'GitHub', 'iBEAt-improc', 'build', 'kidneyvol_1_segment', 'Sheffield')
    target_dir = os.path.join("G:\\Shared drives", "iBEAt Build", "kidneyvol_1_segment", 'Sheffield')
    copy(source_dir, target_dir)
