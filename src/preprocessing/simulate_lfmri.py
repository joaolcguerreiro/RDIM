import argparse


def main():
    # TODO
    pass


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Simulate Low-Field MRI')
    parser.add_argument('--gt-dir', required=True, help='Path to ground truth data directory', type=str)
    parser.add_argument('--save-dir', required=True, help='Path to save data directory', type=str)
    parser.add_argument('--scale-factor', default=2, help='Resolution ratio between LR and HR', type=int)
    parser.add_argument('--augmentation-factor', default=2, help='Define number of lr-hr pairs created for each data sample', type=int)
    args = parser.parse_args()

    main()
