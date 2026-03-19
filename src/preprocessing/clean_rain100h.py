import os
import argparse
from glob import glob


def main():
    try:
        # Rename directories
        for old_split_name, new_split_name in [('RainTrainH', 'training'), ('Rain100H', 'testing')]:
            old_split_path = os.path.join(args.dir, old_split_name)
            new_split_path = os.path.join(args.dir, new_split_name)

            if os.path.exists(old_split_path) and not os.path.exists(new_split_path):
                os.rename(old_split_path, new_split_path)

            # Remove all rain region and streak files
            if os.path.exists(new_split_path):
                for file_modality in ['rainregion', 'rainstreak']:
                    # Fetch all modality file paths
                    modality_file_paths = sorted(glob(os.path.join(new_split_path, f'{file_modality}-*')))

                    for modality_file_path in modality_file_paths:
                        os.remove(modality_file_path)

    except Exception as e:
        print(f'An error occurred: {e}.\n' + 'If error remains download data again.')


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Clean data')
    parser.add_argument('--dir', required=True, help='Path to directory with raw data', type=str)
    args = parser.parse_args()

    main()
