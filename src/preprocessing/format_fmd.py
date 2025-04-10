import re
import os
import shutil
import argparse
from glob import glob
from PIL import Image


def main():
    rgb_dataset_names = ["Confocal_BPAE", "TwoPhoton_BPAE", "WideField_BPAE"]

    # Merge RGB channels
    for rgb_dataset_name in rgb_dataset_names:
        for subset in ["raw", "avg2", "avg4", "avg8", "avg16", "gt"]:
            for scene in os.listdir(os.path.join(args.dir, rgb_dataset_name + "_R", subset)):
                scene_path_r = os.path.join(args.dir, rgb_dataset_name + "_R", subset, scene)
                scene_path_g = os.path.join(args.dir, rgb_dataset_name + "_G", subset, scene)
                scene_path_b = os.path.join(args.dir, rgb_dataset_name + "_B", subset, scene)

                output_scene_path = os.path.join(args.dir, rgb_dataset_name, subset, scene)
                os.makedirs(output_scene_path, exist_ok=True)

                for img_filename in os.listdir(scene_path_r):
                    if rgb_dataset_name == "WideField_BPAE" and not subset == 'gt':
                        img_id = re.search(r'(\d{4}).png$', img_filename).group(1)

                    r_file_path = os.path.join(scene_path_r, img_filename)
                    g_file_path = glob(os.path.join(scene_path_g, f"*{img_id}.png" if rgb_dataset_name == "WideField_BPAE" and not subset == 'gt' else img_filename))[0]
                    b_file_path = glob(os.path.join(scene_path_b, f"*{img_id}.png" if rgb_dataset_name == "WideField_BPAE" and not subset == 'gt' else img_filename))[0]

                    r_file = Image.open(r_file_path)
                    g_file = Image.open(g_file_path)
                    b_file = Image.open(b_file_path)
                    merged = Image.merge("RGB", (r_file, g_file, b_file))

                    save_file_path = os.path.join(output_scene_path, img_filename)
                    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
                    merged.save(save_file_path)

    # Restructure dataset
    dataset_names = ["Confocal_FISH", "Confocal_MICE", "TwoPhoton_MICE"] + rgb_dataset_names

    for dataset_name in dataset_names:
        for subset in ["raw", "avg2", "avg4", "avg8", "avg16"]:
            new_dataset_name = f"{dataset_name.replace('_', '-').replace('TwoPhoton', 'Two-Photon').replace('WideField', 'Wide-Field').replace('FISH', 'Fish').replace('MICE', 'Mice')}-{subset.capitalize().replace('Avg', 'Avg-x')}"
            new_dataset_path = os.path.join(os.path.dirname(args.dir), new_dataset_name)

            os.makedirs(os.path.join(new_dataset_path, "LR"), exist_ok=True)
            os.makedirs(os.path.join(new_dataset_path, "HR"), exist_ok=True)

            lr_path = os.path.join(args.dir, dataset_name, subset)
            hr_path = os.path.join(args.dir, dataset_name, "gt")

            for scene in os.listdir(lr_path):
                lr_scene_path = os.path.join(lr_path, scene)
                hr_scene_path = os.path.join(hr_path, scene)

                hr_img_filename = os.listdir(hr_scene_path)[0]
                hr_img_file_path = os.path.join(hr_scene_path, hr_img_filename)

                for lr_img_filename in os.listdir(lr_scene_path):
                    lr_img_file_path = os.path.join(lr_scene_path, lr_img_filename)
                    new_lr_img_file_path = os.path.join(new_dataset_path, "LR", f"{scene}_{lr_img_filename}")
                    new_hr_img_file_path = os.path.join(new_dataset_path, "HR", f"{scene}_{lr_img_filename}")

                    shutil.copy(lr_img_file_path, new_lr_img_file_path)
                    shutil.copy(hr_img_file_path, new_hr_img_file_path)

    # Delete old directories
    shutil.rmtree(args.dir)


if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Format data')
    parser.add_argument('--dir', required=True, help='Path to directory with raw data', type=str)
    args = parser.parse_args()

    main()
