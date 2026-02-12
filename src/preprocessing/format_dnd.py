import os
import h5py
import shutil
import argparse
import numpy as np
from glob import glob


def main():
    try:
        # Define directories
        test_dir_path = os.path.join(args.dir, 'testing')
        bounding_box_file_path = os.path.join(args.dir, 'info.mat')
        lr_dir_path = os.path.join(test_dir_path, 'LR')

        if not os.path.exists(bounding_box_file_path):
            raise FileNotFoundError('File info.mat not found')

        # Load bounding boxes
        with h5py.File(bounding_box_file_path, 'r') as f_info:
            info = f_info['info']
            bounding_boxes = info['boundingboxes']

            # Remove and create LR directory
            if os.path.exists(lr_dir_path):
                shutil.rmtree(lr_dir_path)

            os.makedirs(lr_dir_path)

            # Fetch all test files
            file_paths = sorted(file_path for file_path in glob(os.path.join(test_dir_path, '*')) if os.path.basename(file_path) != 'LR')

            # Parse files
            for img_index, file_path in enumerate(file_paths):
                if file_path.endswith('.mat'):
                    with h5py.File(file_path, 'r') as f_img:
                        lr_img = np.array(next(iter(f_img.values()))).astype(np.float32)

                    # Extract crops using bounding boxes
                    ref = bounding_boxes[0][img_index]
                    boxes = np.array(info[ref]).T

                    for bounding_box_index in range(boxes.shape[0]):
                        x0 = int(boxes[bounding_box_index, 0] - 1)
                        y0 = int(boxes[bounding_box_index, 1] - 1)
                        x1 = int(boxes[bounding_box_index, 2])
                        y1 = int(boxes[bounding_box_index, 3])

                        # Crop image
                        crop_lr_img = lr_img[:, y0:y1, x0:x1].copy()

                        # Save crop
                        with h5py.File(os.path.join(lr_dir_path, f'{img_index + 1:04d}_{bounding_box_index + 1:02d}.mat'), 'w') as f:
                            f.create_dataset('Idenoised_crop', data=crop_lr_img)

                # Remove file after processing
                os.remove(file_path)

        # Remove bounding boxes file
        os.remove(bounding_box_file_path)

    except Exception as e:
        print(f'An error occurred: {e}.\n' + 'If error remains download data again.')


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Format data')
    parser.add_argument('--dir', required=True, help='Path to directory with raw data', type=str)
    args = parser.parse_args()

    main()
