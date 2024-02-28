# <span style="color: orange">Prepare Data</span>

#### Table of Contents

1) [Overview](#overview)
2) [Prepare LDCT-and-Projection-Data Dataset](#prepare-ldct-and-projection-data-dataset)
4) [Prepare LIDC-IDRI Dataset](#prepare-lidc-idri-dataset)
3) [Prepare ??Low-Field MRI?? Dataset](#prepare-??Low-Field-MRI??-dataset)
5) [Prepare Other Dataset](#prepare-other-dataset)

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

> Note: Validation data is optional. Use flag `--val=False` during training...

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

<hr style="border:1px solid gray">

#### Prepare LIDC-IDRI Dataset

1) Download [LIDC-IDRI](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254) dataset.

2) Navigate to `preprocessing` directory:
   ```
   cd src/preprocessing
   ```

3) Run clone data script to duplicate dataset into `data` directory:
   ```
   python clone_data.py --dir=<path_to_raw_lidc_idri> --dataset-name="LIDC-IDRI"
   ```

4) Run data cleaning script:
   ```
   python clean_lidc_idri.py --dir="../../data/LIDC-IDRI"
   ```

5) Run data formatting script:
   ```
   python format_lidc_idri.py --dir="../../data/LIDC-IDRI"
   ```

6) Run data splitting script:
   ```
   python split_data.py --dir="../../data/LIDC-IDRI"
   ```

7) Run data downsampling script to create LR-HR pairs:
   ```
   python simulate_ldct.py --dir="../../data/LIDC-IDRI/training"
   python simulate_ldct.py --dir="../../data/LIDC-IDRI/validation"
   python simulate_ldct.py --dir="../../data/LIDC-IDRI/testing"
   ```

> Note: The processed dataset is stored in the `data` directory, following the structure in [Overview](#overview).

<hr style="border:1px solid gray">

#### Prepare ??Low-Field MRI?? Dataset

`... download_some_name.sh... clean_some_name.py... simulate_lfmri.py... split_data.py`

> Note: The processed dataset is stored in the `data` directory, following the structure in [Overview](#overview).

<hr style="border:1px solid gray">

#### Prepare Other Dataset

Supported file formats:

* NifTI;

Accepted data shapes are as follows:

* `(W, H, D)`.

* `(W, H, D, C)`.

> Note: The dimensions order is reversed during execution to align with PyTorch. 

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

> Note: It is assumed that the dataset is stored in the `data` directory, aligned with the structure in [Overview](#overview). The aforementioned is not mandatory and the dataset directory location is arbitrary. In such case adjust the `-dir` options in the commands above.
