import os
import shutil
import zipfile
import argparse
from glob import glob


def main():
    try:
        # Fetch all data file paths
        dataset_file_paths = sorted(glob(os.path.join(args.dir, '*')))

        # Unzip files
        for file_path in dataset_file_paths:
            # Pass directory files
            if os.path.isdir(file_path):
                continue

            # Unzip files
            if file_path.endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zip_file:
                    zip_file.extractall(path=args.dir)

            # Remove file
            os.remove(file_path)

        # Restructure dataset
        dataset_names = ["Unknown-x2", "Unknown-x4"]

        for dataset_name in dataset_names:
            new_dataset_path = os.path.join(os.path.dirname(args.dir), "DIV2K-" + dataset_name)
            degradation_operator = dataset_name.split("-")[0].lower()
            degradation_scale_factor = dataset_name.split("-")[1].capitalize()

            for split in ["train", "valid"]:
                # Define original data directory paths
                lr_path = os.path.join(args.dir, "DIV2K_" + split + "_LR_" + degradation_operator, degradation_scale_factor)
                hr_path = os.path.join(args.dir, "DIV2K_" + split + "_HR")

                # Define new data directory paths
                new_dataset_split_path = os.path.join(new_dataset_path, "training" if split == "train" else "testing")

                os.makedirs(new_dataset_split_path, exist_ok=True)
                os.makedirs(os.path.join(new_dataset_split_path, "LR"), exist_ok=True)
                os.makedirs(os.path.join(new_dataset_split_path, "HR"), exist_ok=True)

                for filename in os.listdir(hr_path):
                    basename, file_extension = os.path.splitext(filename)

                    lr_img_file_path = os.path.join(lr_path, basename + degradation_scale_factor + file_extension)
                    hr_img_file_path = os.path.join(hr_path, filename)
                    new_lr_img_file_path = os.path.join(new_dataset_split_path, "LR", filename)
                    new_hr_img_file_path = os.path.join(new_dataset_split_path, "HR", filename)

                    shutil.copy(lr_img_file_path, new_lr_img_file_path)
                    shutil.copy(hr_img_file_path, new_hr_img_file_path)

        # Delete old directories
        shutil.rmtree(args.dir)

    except Exception as e:
        print(f'An error occurred: {e}.\n' + 'If error remains download data again.')


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Format data')
    parser.add_argument('--dir', required=True, help='Path to directory with raw data', type=str)
    args = parser.parse_args()

    main()
