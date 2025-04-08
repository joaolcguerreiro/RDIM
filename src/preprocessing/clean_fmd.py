import os
import tarfile
import argparse
from glob import glob


def main():
    try:
        # Fetch all data file paths
        dataset_file_paths = sorted(glob(os.path.join(args.dir, '*')))

        # Discard non-tar files and test_mix
        for file_path in dataset_file_paths:
            # Pass directory files
            if os.path.isdir(file_path):
                continue

            # Untar files
            if file_path.endswith('.tar') and os.path.basename(file_path) != 'test_mix.tar':
                with tarfile.open(file_path, 'r') as tar_file:
                    tar_file.extractall(path=args.dir)

            # Remove file
            os.remove(file_path)

        # Remove all desktop.ini files recursively
        desktop_ini_file_paths = glob(os.path.join(args.dir, '**', 'desktop.ini'), recursive=True)

        for desktop_ini_file_path in desktop_ini_file_paths:
            os.remove(desktop_ini_file_path)

    except Exception as e:
        print(f'An error occurred: {e}.\n' + 'If error remains download data again.')


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Clean data')
    parser.add_argument('--dir', required=True, help='Path to directory with raw data', type=str)
    args = parser.parse_args()

    main()
