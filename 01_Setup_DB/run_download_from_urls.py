import os
import time
from glob import glob
from multiprocessing import Pool, freeze_support


PYTHON_PATH = "/Users/eunsupark/Softwares/miniconda3/envs/ap-backend/bin/python"
SCRIPT_PATH = "scripts/download_from_urls.py"
JSON_DIR = "./json"
FAILED_DIR = "./json_failed"
NUM_PARALLEL_JSONS = 4  # Number of JSON files to process concurrently


def process_json(file_path: str) -> tuple[str, int]:
    """Process a single JSON file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Tuple of (file_path, exit_code).
    """
    file_name = os.path.basename(file_path)
    command = f"{PYTHON_PATH} {SCRIPT_PATH} --input {file_path} --parallel 1"
    print(f"[START] {file_name}")
    result = os.system(command)
    return (file_path, result)


def main():
    """Main entry point."""
    # Create failed directory if not exists
    os.makedirs(FAILED_DIR, exist_ok=True)

    while True:
        list_json = sorted(glob(f'{JSON_DIR}/*.json'))  # Sort for time order

        if len(list_json) > 0:
            # Take up to NUM_PARALLEL_JSONS files
            batch = list_json[:NUM_PARALLEL_JSONS]
            print(f"\n{'='*50}")
            print(f"Processing {len(batch)} JSON files in parallel")
            print(f"{'='*50}\n")

            # Process batch in parallel
            with Pool(processes=len(batch)) as pool:
                results = pool.map(process_json, batch)

            # Handle results
            for file_path, exit_code in results:
                file_name = os.path.basename(file_path)
                if exit_code == 0:
                    # Success: delete JSON
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    print(f"  [OK] {file_name}")
                else:
                    # Failed: move to failed directory
                    if os.path.exists(file_path):
                        failed_path = os.path.join(FAILED_DIR, file_name)
                        os.rename(file_path, failed_path)
                    print(f"  [FAIL] {file_name} -> {FAILED_DIR}/")

        else:
            print("No JSON files found. Waiting 60 seconds...")
            time.sleep(60)


if __name__ == '__main__':
    freeze_support()
    main()
