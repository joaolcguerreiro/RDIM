# <span style="color: orange">Prepare Data</span>

#### Table of Contents

1) [Overview](#overview)
2) [Prepare LDCT-and-Projection-Data Dataset](#prepare-ldct-and-projection-data-dataset)
3) [Prepare BraTS 2023 Dataset](#prepare-brats-2023-dataset)
4) [Prepare FMD Dataset](#prepare-fmd-dataset)
5) [Prepare DIV2K Dataset](#prepare-div2k-dataset)
6) [Prepare SIDD Dataset](#prepare-sidd-dataset)
7) [Prepare FFHQ Dataset](#prepare-ffhq-dataset)
8) [Prepare Other Dataset](#prepare-other-dataset)

<hr style="border:1px solid gray">

#### Overview

Every low-resolution image should have a corresponding high-resolution image. The image files of each LR-HR pair should have the same filename.

```
├───data
      ├───example_dataset_01
      │      ├───training
      │      │      ├───LR
      │      │      │      └──example_0001.nii.gz
      │      │      │      └──example_0002.nii.gz
      │      │      ...    └──...
      │      │      │
      │      │      └───HR
      │      │      │      └──example_0001.nii.gz
      │      │      │      └──example_0002.nii.gz
      │      │      ...    └──...
      │      │
      │      ├───validation
      │      │      ├───LR
      │      │      └───HR
      │      │
      │      ├───testing
      │      │      ├───LR
      │      │      └───HR
      │      └───
      │
      ├───example_dataset_02
      │      ├───training
      │      │      ├───LR
      │      │      └───HR
      │      ├───validation
      │      │      ├───LR
      │      │      └───HR
      │      ├───testing
      │      │      ├───LR
      │      │      └───HR
      │      └───
      ...
```

> Note: Validation data split is optional.

<hr style="border:1px solid gray">

#### Prepare LDCT-and-Projection-Data Dataset

1) Download [LDCT-and-Projection-Data](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758026) dataset.

2) Navigate to `preprocessing` directory:
   ```
   cd src/preprocessing
   ```

3) Run clone data script to duplicate dataset into `data` directory:
   ```
   python clone_data.py --dir=<path_to_raw_ldct_pd> --dataset-name="LDCT-PD"
   ```

4) Run data cleaning script:
   ```
   python clean_ldct_pd.py --dir="../../data/LDCT-PD"
   ```

5) Run data splitting script:
   ```
   python split_data.py --dir="../../data/LDCT-PD"
   ```

6) Run data formatting script:
   ```
   python format_ldct_pd.py --dir="../../data/LDCT-PD"
   ```

> Note: The processed dataset is stored in the `data` directory, following the structure in [Overview](#overview).

> Note: It is assumed validation data split is intended. Otherwise, use flag `--val=False` in `split_data.py`.

<hr style="border:1px solid gray">

#### Prepare BraTS 2023 Dataset

1) Download [BraTS 2023](https://www.synapse.org/#!Synapse:syn51156910/wiki/621282) dataset training split.

2) Navigate to `preprocessing` directory:
   ```
   cd src/preprocessing
   ```

3) Run clone data script to duplicate dataset into `data` directory:
   ```
   python clone_data.py --dir=<path_to_raw_brats_2023> --dataset-name="BraTS-2023"
   ```

4) Run data cleaning script:
   ```
   python clean_brats_2023.py --dir="../../data/BraTS-2023"
   ```

5) Run data formatting script:
   ```
   python format_brats_2023.py --dir="../../data/BraTS-2023"
   ```

6) Run data splitting script:
   ```
   python split_data.py --dir="../../data/BraTS-2023"
   ```

7) Run data downsampling script to create LR-HR pairs:
   ```
   python simulate_lfmri.py --dir="../../data/BraTS-2023/training"
   python simulate_lfmri.py --dir="../../data/BraTS-2023/validation"
   python simulate_lfmri.py --dir="../../data/BraTS-2023/testing"
   ```

> Note: The processed dataset is stored in the `data` directory, following the structure in [Overview](#overview).

> Note: It is assumed validation data split is intended. Otherwise, use flag `--val=False` in `split_data.py` and ignore `simulate_lfmri.py` for `validation` directory.

<hr style="border:1px solid gray">

#### Prepare FMD Dataset

1) Download [FMD](https://curate.nd.edu/articles/dataset/Fluorescence_Microscopy_Denoising_FMD_dataset/24744648) dataset.

2) Navigate to `preprocessing` directory:
   ```
   cd src/preprocessing
   ```

3) Run clone data script to duplicate dataset into `data` directory:
   ```
   python clone_data.py --dir=<path_to_raw_fmd> --dataset-name="FMD"
   ```

4) Run data cleaning script:
   ```
   python clean_fmd.py --dir="../../data/FMD"
   ```

5) Run data formatting script:
   ```
   python format_fmd.py --dir="../../data/FMD"
   ```

6) Run data splitting script for each subset:
   ```
   python split_data.py --dir="../../data/Confocal-BPAE-Raw" --pair-wise=True
   python split_data.py --dir="../../data/Two-Photon-BPAE-Raw" --pair-wise=True
   python split_data.py --dir="../../data/Wide-Field-BPAE-Raw" --pair-wise=True
   ```

> Note: The processed dataset is stored in the `data` directory, following the structure in [Overview](#overview).

> Note: It is assumed validation data split is intended. Otherwise, use flag `--val=False` in `split_data.py`.

<hr style="border:1px solid gray">

#### Prepare DIV2K Dataset

1) Download [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset.

2) Navigate to `preprocessing` directory:
   ```
   cd src/preprocessing
   ```

3) Run clone data script to duplicate dataset into `data` directory:
   ```
   python clone_data.py --dir=<path_to_raw_div2k> --dataset-name="DIV2K"
   ```

4) Run data formatting script:
   ```
   python format_div2k.py --dir="../../data/DIV2K"
   ```

> Note: The processed dataset is stored in the `data` directory, following the structure in [Overview](#overview).

<hr style="border:1px solid gray">

#### Prepare SIDD Dataset

1) Download [SIDD](https://abdokamel.github.io/sidd/) dataset.

2) Navigate to `preprocessing` directory:
   ```
   cd src/preprocessing
   ```

3) Run clone data script to duplicate dataset into `data` directory:
   ```
   python clone_data.py --dir=<path_to_raw_sidd> --dataset-name="SIDD"
   ```

4) Run data cleaning script:
   ```
   python clean_sidd.py --dir="../../data/SIDD"
   ```

5) Run data splitting script:
   ```
   python split_data.py --dir="../../data/SIDD"
   ```

6) Run data formatting script:
   ```
   python format_sidd.py --dir="../../data/SIDD"
   ```

> Note: The processed dataset is stored in the `data` directory, following the structure in [Overview](#overview).

> Note: It is assumed validation data split is intended. Otherwise, use flag `--val=False` in `split_data.py`.

<hr style="border:1px solid gray">

#### Prepare FFHQ Dataset

1) Download [FFHQ](https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL?usp=drive_open) dataset training split.

2) Navigate to `preprocessing` directory:
   ```
   cd src/preprocessing
   ```

3) Run clone data script to duplicate dataset into `data` directory:
   ```
   python clone_data.py --dir=<path_to_raw_ffhq> --dataset-name="FFHQ-<task>"
   ```

4) Run data cleaning script:
   ```
   python clean_ffhq.py --dir="../../data/FFHQ-<task>"
   ```

5) Run data formatting script:
   ```
   python format_ffhq.py --dir="../../data/FFHQ-<task>"
   ```

6) Run data splitting script:
   ```
   python split_data.py --dir="../../data/FFHQ-<task>"
   ```

7) Run data corruption script to create corrupted-original pairs for a specific task (inpainting, colorization or deblurring):
   ```
   python simulate_task.py --dir="../../data/FFHQ-<task>/training" --task=<task>
   python simulate_task.py --dir="../../data/FFHQ-<task>/validation" --task=<task>
   python simulate_task.py --dir="../../data/FFHQ-<task>/testing" --task=<task>
   ```

> Note: The processed dataset is stored in the `data` directory, following the structure in [Overview](#overview).

> Note: It is assumed validation data split is intended. Otherwise, use flag `--val=False` in `split_data.py` and ignore `simulate_task.py` for `validation` directory.

<hr style="border:1px solid gray">

#### Prepare Other Dataset

Supported file formats:

* 2D Data - NifTI, PNG, JPG, JPEG, and TIFF.

* 3D Data - NifTI.

Accepted data shapes are as follows:

* 2D Data - `(H, W)` and `(H, W, C)`.

* 3D Data - `(W, H, D)` and `(W, H, D, C)`.

> Note: For NifTI the dimensions order is reversed during execution to align with PyTorch. For the other file formats the channel dimension is moved to the first position.

Every low-resolution image should have a corresponding high-resolution image. The image files of each LR-HR pair should have the same filename. The following illustrates a valid dataset structure:

```
├───example_dataset
      ├───training
      │      ├───LR
      │      │      └──example_0001.nii.gz
      │      │      └──example_0002.nii.gz
      │      ...    └──...
      │      │
      │      └───HR
      │      │      └──example_0001.nii.gz
      │      │      └──example_0002.nii.gz
      │      ...    └──...
      │
      ├───validation
      │      ├───LR
      │      └───HR
      │
      ├───testing
      │      ├───LR
      │      └───HR
      └───
```

> Note: `training`, `validation` and `testing` do not need to be in the same directory. This is a just a suggested directory structure.

---

Store the dataset in the `data` directory to align with the structure in [Overview](#overview):

* Clone data into `data` directory:
   ```
   python clone_data.py --dir=<path_to_raw_alternative_dataset> --dataset-name=<dataset_name>
   ```

> Note: This step is not mandatory. The structure in [Overview](#overview) is just a convention to ensure organization.

---

If the dataset does not include data splits or lacks LR-HR pairs, run the following scripts if necessary:

* Split data:
   ```
   python split_data.py --dir="../../data/<dataset_name>"
   ```

* Downsample data to create LR-HR pairs:
   ```
   # Simulate Low-Dose CT
   python simulate_ldct.py --dir="../../data/<dataset_name>/training"
   python simulate_ldct.py --dir="../../data/<dataset_name>/validation"
   python simulate_ldct.py --dir="../../data/<dataset_name>/testing"
   ```

   ```
   # Simulate Low-Field MRI
   python simulate_lfmri.py --dir="../../data/<dataset_name>/training"
   python simulate_lfmri.py --dir="../../data/<dataset_name>/validation"
   python simulate_lfmri.py --dir="../../data/<dataset_name>/testing"
   ```

> Note: Expects `training`, `validation`, and `testing` directories inside `<dataset_name>` directory.

> Note: It is assumed validation data split is intended. Otherwise, use flag `--val=False` in `split_data.py` and ignore `simulate_ldct.py`/`simulate_lfmri.py` for `validation` directory.

> Note: It is assumed that the dataset is stored in the `data` directory, aligned with the structure in [Overview](#overview). The aforementioned is not mandatory and the dataset directory location is arbitrary. In such case adjust the `-dir` options in the commands above.
