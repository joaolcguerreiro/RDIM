import shutil
import os
import argparse


def main():
    try:
        # Define path to destination directory
        dest_dir_path = os.path.join('..', '..', 'data', args.dataset_name or os.path.basename(args.dir))

        # Delete destination directory if already exists
        shutil.rmtree(dest_dir_path) if os.path.exists(dest_dir_path) else None

        # Clone directory
        shutil.copytree(args.dir, dest_dir_path)
    except Exception as e:
        print(f'An error occurred: {e}.\n' + 'If error remains download data again.')


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Clone data')
    parser.add_argument('--dir', required=True, help='Path to directory with raw data', type=str)
    parser.add_argument('--dataset-name', default=None, help='Name of the dataset', type=str)
    args = parser.parse_args()

    main()
