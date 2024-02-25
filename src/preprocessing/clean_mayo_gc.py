import shutil
import argparse
import os


def main():
    try:
        # Remove README file
        readme_file_path = os.path.join(args.dir, 'Mayo_Grand_Challenge_READ_ME.pdf')

        if os.path.exists(readme_file_path):
            os.remove(readme_file_path)

        # Remove unnecessary folders
        unnecessary_dir_paths = [
            os.path.join(args.dir, 'ACR_Phantom_Data'),
            os.path.join(args.dir, 'Ancillary_Information'),
            os.path.join(args.dir, 'Patient_Data', 'Testing_Projection_Data'),
            os.path.join(args.dir, 'Patient_Data', 'Training_Projection_Data'),
        ]

        for unnecessary_dir_path in unnecessary_dir_paths:
            shutil.rmtree(unnecessary_dir_path)

    except Exception as e:
        print(f'An error occurred: {e}.\n' + 'If error remains download data again.')


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Clean data')
    parser.add_argument('--dir', required=True, help='Path to directory with raw data', type=str)
    args = parser.parse_args()

    main()
