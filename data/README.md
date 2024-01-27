#### Table of Contents

1) [Overview](#overview)
2) [Prepare ??Low-Field MRI?? Dataset](#prepare-??Low-Field-MRI??-dataset)
3) [Prepare ??Low-Dose CT?? Dataset](#prepare-??Low-Dose-CT??-dataset)
4) [Prepare LIDC-IDRI Dataset](#prepare-lidc-idri-dataset)
5) [Prepare Other Dataset](#prepare-other-dataset)

---

#### Overview

Every low-resolution image should have a corresponding high-resolution image. The pair of low-res and high-res images should have the same filename

```
├───data
      ├───example_dataset_01
      │     ├───training
      │     │      ├────LR
      │     │      │      └──example-0001.nii.gz
      │     │      │      └──example-0002.nii.gz
      │     │      ...    └──...
      │     │      │
      │     │      └───HR
      │     │      │      └──example-0001.nii.gz
      │     │      │      └──example-0002.nii.gz
      │     │      ...    └──...
      │     │
      │     ├───validation
      │     │      ├───LR
      │     │      └───HR
      │     ├───testing
      │     │      ├───LR
      │     │      └───HR
      │     └──
      │
      ├───example_dataset_02
      │     ├───training
      │     │      ├───LR
      │     │      └───HR
      │     ├───validation
      │     │      ├───LR
      │     │      └───HR
      │     ├───testing
      │     │      ├───LR
      │     │      └───HR
      │     └──
      ...
```

> Note: Validation data is optional. Use flag `--val=False` during training...

---

#### Prepare ??Low-Field MRI?? Dataset

`... download_some_name.sh... clean_some_name.py... split_data.py... simulate_lfmri.py`

---

#### Prepare ??Low-Dose CT?? Dataset

`... download_some_name.sh... clean_some_name.py... split_data.py... simulate_ldct.py`

---

#### Prepare LIDC-IDRI Dataset

1) Download [LIDC-IDRI](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254).

2) Navigate to `preprocessing` directory:
   ```
   cd src/preprocessing
   ```

3) Run data cleaning script:
   ```
   python clean_lidc_idri.py --dir=<path_to_lidc_idri>
   ```

4) Run data formatting script:
   ```
   python format_lidc_idri.py --dir=<path_to_lidc_idri>
   ```

5) Run data splitting script:
   ```
   python split_data.py --dir=<path_to_lidc_idri>
   ```

6) Run data downsampling script:
   ```
   python simulate_ldct.py --dir=<path_to_training_split>
   python simulate_ldct.py --dir=<path_to_validation_split>
   python simulate_ldct.py --dir=<path_to_testing_split>
   ```

---

#### Prepare Other Dataset

BE SURE DATA FILE FORMAT IS NIFTI... channel dim required to be first... depth dimension required even in 2D processing... accepted data shapes: (D, H, W) or (C, D, H, W)

Supported files: NifTI...

every low-resolution image should have a corresponding high-resolution image. The pair of low-res and high-res images should have the same filename

The following illustrates a valid dataset structure:

```
├───example_dataset
      ├───training
      │      ├────LR
      │      │      └──example-0001.nii.gz
      │      │      └──example-0002.nii.gz
      │      ...    └──...
      │      │
      │      └───HR
      │      │      └──example-0001.nii.gz
      │      │      └──example-0002.nii.gz
      │      ...    └──...
      │
      ├───validation
      │      ├───LR
      │      └───HR
      ├───testing
      │      ├───LR
      │      └───HR
      └──
```

> Note: `training`, `validation` and `testing` do not need to be in the same directory. This is a just a suggested directory structure.