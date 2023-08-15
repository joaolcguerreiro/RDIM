# SRMeDiff

Repository ....

TODO: MAKE GIF WITH SAMPLE EVERY CHECKPOINT AND ALSO DISPLAY CHART FOR SAMPLE PIXEL DISTRIBUTION VS GROUND TRUTH PIXEL DISTRIBUTION
SINCE WE SHOW THE EVOLUTION OF ONLY ONE SAMPLE, WE ONLY SHOW THE PIXEL DISTRIBUTION OF THAT SAMPLE AND NOT THE GLOBAL PIXEL DISTRIBUTION

...

---

### Quick Inference

1) ....
   * Using your own dataset
     * Make sure your dataset is aligned with **[Data Requirements]()** section.

   * Using LDCT Dataset
     * Follow **[Download LDCT Dataset]()** and **[Prepare LDCT Dataset]()** sections.

2) Follow **[Setup Environment]()** section.

3) Run the following command:
   ```
    python src/scripts/srmediff/inference.py --checkpoint=<path_to_checkpoint> --lr-dir=<path_to_lr_dir> --save-dir=<path_to_save_dir> [--batch-size=<batch_size>] [--num-workers=<num_workers>]
   ```

<hr style="border:2px solid gray">

# Setup Environment

1) Create environment:
   ```
   conda create -n srmediff python=3.10.11
   ```
   ```
   conda activate srmediff
   ```

2) Install Python dependencies (Python 3.10 is recommended):
   ```
   pip install -r requirements.txt
   ```

3) Install layout engine of Graphviz properly (https://github.com/mert-kurttutan/torchview).
   ```
   # For macOS
   brew install graphviz
   ```
   ```
   # For CentOS
   sudo yum install graphviz
   ```
   ```
   # For Debian-based Linux (e.g. Ubuntu)
   apt-get install graphviz
   ```
   ```
   # For Windows
   choco install graphviz
   ```

<hr style="border:2px solid gray">

# Data Requirements

BE SURE DATA FILE FORMAT IS NIFTI... IF DICOM THEN CONVERT BEFORE ????... channel dim required to be first... accepted data shapes: (W, H, D) or (C, W, H, D) (WHERE H, W, D can interchange and C>0)... (for video super-resolution a valid shape is (C, T, W, H)... where T, W, H can interchange)

every low-resolution image should have a corresponding high-resolution image. The pair of low-res and high-res images should have the same filename

Supported files: NifTI...

<hr style="border:2px solid gray">

# Download LDCT Dataset

Follow this steps in case you want to use the LDCT dataset.

1) Download requires the [NBIA Data Retriever](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images).

2) Download data with Manifest File.
   - **macOS** and **Windows**
     - Double click manifest file in `srmediff/data/manifest.tcia` to open it;
     - Select `Descriptive Directory Name` option;
     - Click `Browse` and select destiny directory for downloaded files. We recommend `srmediff/data` directory;
     - Click `Start`;
   - **Linux**
     - Run command:
       ```
       # For CentOS
       /opt/NBIADataRetriever/NBIADataRetriever --cli <abolute_path_to_project>/srmediff/data/manifest.tcia -d <abolute_path_to_project>/srmediff/data
       ```
       ```
       # For Ubuntu
       /opt/nbia-data-retriever/nbia-data-retriever --cli <abolute_path_to_project>/srmediff/data/manifest.tcia -d <abolute_path_to_project>/srmediff/data
       ```
     - Help: Follow *"Running the NBIA Data Retriever on Linux"* section in https://wiki.cancerimagingarchive.net/display/NBIA/NBIA+Data+Retriever+Command-Line+Interface+Guide. `<location>/<manifest file name>.tcia` and `<parent location>/<download directory>` are expected to be the absolute paths. The remaining command-line options are not considered.

<hr style="border:2px solid gray">

# Prepare LDCT Dataset

0) Navigate to `preprocessing` directory:
   ```
   cd src/preprocessing
   ```
   
1) Run data cleaning script:
   ```
   python clean_ldct.py
   ```
   
2) Run data formatting script:
   ```
   python format_ldct.py
   ```
   
3) Run data splitting script:
   ```
   python split_data.py
   ```

(DELETE FROM HERE WHEN LDCT IS WORKING!!!) **LIDC-IDRI Dataset**

0) Navigate to `preprocessing` directory:
   ```
   cd src/preprocessing
   ```

1) Run data cleaning script:
   ```
   python clean_lidc_idri.py
   ```

2) Run data formatting script:
   ```
   python format_lidc_idri.py
   ```

3) Run data splitting script:
   ```
   python split_data.py
   ```

4) Run data downsampling script:
   ```
   python downsample_data.py
   ```

<hr style="border:2px solid gray">

# Prepare Alternative Dataset

In case you want to use your own dataset:

1) VALIDATION DATA IS OPTIONAL... BUT YOU CAN RUN SPLIT SCRIPT [OPTIONAL] IF YOU WANT TO SPLIT YOUR TRAINING DATA (USE --DIR FLAG TO SPECIFY DIRECTORY PATH)... validation and testing splits are not MANDATORY.

2) BE SURE DATA FILE FORMAT IS NIFTI... IF DICOM THEN CONVERT BEFORE ????... channel dim required to be first... accepted data shapes: (W, H, D) or (C, W, H, D) (WHERE H, W, D can interchange and C>0)... (for video super-resolution a valid shape is (C, T, W, H)... where T, W, H can interchange)

3) UPDATE CONFIG FILE WITH DATA PATHS

4) YOU CAN SIMULATE THE DEGRADATIONS WITH DOWNSAMPLE_DATA SCRIPT... [OPTIONAL]

---

The following illustrates a valid directory structure:

```
├───data
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

Note: This aforementioned structure is not mandatory. Also, the validation and testing splits are not MANDATORY.

<hr style="border:2px solid gray">

# Configuration File

Global configuration files are in configuration/ directory... (in case of trainig resume a copy of this files is done to load original parameters) 

#### Batch Size

- sdfhjsdfdsj

---

#### Gradient Accumulation Steps

- Strategy for memory-efficiency. Effectively simulates a batch size of K x BATCH_SIZE, while using the memory required for a batch size of BATCH_SIZE, e.g., if K=2 and BATCH_SIZE=4, then essentially we have a batch size of 8.

<hr style="border:2px solid gray">

# Experiment Tracking

Everytime you train a model.... a experiments folder is built (except if training was resumed, thereby it is used the folder of the prior run)... the global configuration file is saved to resume training.

### Directory Structure

the structure is the following....
<model_name>
   <date_time>
      config.yaml
      <checkpoints>
      <figures>
      <logs>
      <results>

---

### Folder Description

config.yaml ... copy of global configuration file used in the experiment. Loaded in case of training resuming.

checkpoints/ .... Contains saved checkpoints of the model during training

figures/ ....

logs/ ....

results/ .... Contains the output of running the trained model on the test set. The results are typically generated by running the trained model on the test set using the test.py script.

<hr style="border:2px solid gray">

# Run Model

In the following instructions, replace `<model>` by one of the options below:

* `esrgan`

* `srdiff`

* `srmediff`

---

### Train and Validation

1) go to configuration/ and adjust configuration parameters needed....

2) Navigate to `src/scripts/<model>/`

3) Run ...
   ```
   # Default execution
   python train.py
   ```

   ```
   # With patch-based training (Recommended)
   python train.py --patch_size=<patch_size>
   ```

   ```
   # Without Validation
   python train.py --val=False
   ```

Note: If `patch_size="[-1, -1, -1]"` then no cropping is employed (more computationally expensive)... Also, for `patch_size` equal or smaller than `"[8, 8, 8]"` the `batch_size` is required to be greater than `1`.

---

### Load Pre-trained

1) go to `configuration/<model>/config.yaml` and update pre-trained path key parameter of model....

2) Navigate to `src/scripts/<model>/`

3) Run ...
   ```
   python train.py
   ```

---

### Resume Training

1) go to `experiments/<model>` and copy experiment path

2) Navigate to `src/scripts/<model>/`

3) Run ...
   ```
   python train.py --resume=<experiment_path>
   ```

    Note: Load pre-trained differs from resume training since in pre-training only the model weights are loaded. If you simultaneously resume a training the checkpoint weights will override the pre-trained weights.

---

### Test

1) go to `experiments/<model>` and copy experiment path

2) Navigate to `src/scripts/<model>/`

3) Run ...
   ```
   python test.py --exp=<experiment_path> --epoch=<epoch>
   ```

---

### Inference

1) Get path of a checkpoint file...

2) Navigate to `src/scripts/<model>/`

3) Run ...
   ```
    python inference.py --checkpoint=<path_to_checkpoint> --lr-dir=<path_to_lr_dir> --save-dir=<path_to_save_dir> [--batch-size=<batch_size>] [--num-workers=<num_workers>]
   ```

<hr style="border:2px solid gray">

# Visualize Results

Image visualization file...

0) Navigate to `visualization` directory:
   ```
   cd visualization
   ```

1) Run visualization script:
   ```
   python visualization.py --mode=<mode> --lr-dir=<lr_dir> --hr-dir=<hr_dir> [--checkpoint=<checkpoint>] [--augmentation=<bool>] [--plane=<plane>] [--dim=<dimension>] [--volume=<volume_index>] [--slice=<slice_index>] [--batch-size=<batch_size>] [--num-workers=<num_workers>] [--labels=<bool>]
   ```

---

#### Visualization Mode

`--mode`

| Option            | Description                                                             |
|-------------------|-------------------------------------------------------------------------|
| `'gt_scans'`      | Display ground-truth scan counterpart                                   |
| `'lr_scans'`      | Display low-resolution scan counterpart                                 |
| `'sr_scans'`      | Display super-resolution scan counterpart                               |
| `'lr_gt_scans'`   | Display low-resolution and ground truth scans jointly |
| `'compare_scans'` | Display low-resolution, super-resolution and ground truth scans jointly |

---

#### Low-Resolution Data Directory <span style="color: orange">[SEMI-OPTIONAL]</span>

`--lr-dir`

| Option | Description                                                                |
|--------|----------------------------------------------------------------------------|
| `str`  | Path to low-resolution data directory                                   |

Required when mode is `lr_scans`, `sr_scans`, `lr_gt_scans` or `compare_scans`

---

#### High-Resolution Data Directory <span style="color: orange">[SEMI-OPTIONAL]</span>

`--hr-dir`

| Option | Description                                                                |
|--------|----------------------------------------------------------------------------|
| `str`  | Path to high-resolution data directory                                   |

Required when mode is `gt_scans`, `lr_gt_scans` or `compare_scans`

---

#### Model Checkpoint <span style="color: orange">[OPTIONAL]</span>

`--checkpoint`

| Option | Description                                                                | Default |
|--------|----------------------------------------------------------------------------|:-------:|
| `str`  | Path to model checkpoint file                                   |  None   |

---

#### Data Augmentation <span style="color: orange">[OPTIONAL]</span>

`--augmentation`

| Option | Description                                                 | Default |
|--------|-------------------------------------------------------------|:-------:|
| `bool` | Apply augmentation techniques to the visualization data |  False  |

---

#### Target Pixel Window for Visualization <span style="color: orange">[OPTIONAL]</span>

`--pixel-range`

| Option     | Description                                                                |   Default    |
|------------|----------------------------------------------------------------------------|:------------:|
| `sequence` | Window range of pixel values (min, max). For multi-channel scaling provide ((min...), (max...))                                   | (None, None) |

---

#### View Plane <span style="color: orange">[OPTIONAL]</span>

`--plane`

| Option          | Description | Default |
|-----------------| --- |:-------:|
| `'axial'` | Your description here |    ✓     |
| `'sagittal'` | Your description here |         |
| `'coronal'` | Your description here |         |

---

#### Visualization Spacial Dimension <span style="color: orange">[OPTIONAL]</span>

`--dim`

| Option          | Description | Default |
|-----------------| --- |:-------:|
| `'2D'` | Your description here |    ✓     |
| `'3D'` | Your description here |         |

---

#### Volume Index <span style="color: orange">[OPTIONAL]</span>

`--volume`

| Option | Description                                                                  | Default |
|--------|------------------------------------------------------------------------------|:-------:|
| `int`  | Non-negative integer (n > 0) referencing volume. (-1 to display all volumes) |   -1    |

---

#### Slice Index <span style="color: orange">[OPTIONAL]</span>

`--slice`

| Option | Description | Default |
|--------| --- |:-------:|
| `int`  | ... |   -1    |

---

#### Batch Size <span style="color: orange">[OPTIONAL]</span>

`--batch-size`

| Option | Description | Default |
|--------| --- |:-------:|
| `int`  | ... |    4    |

---

#### Data Loading Subprocesses <span style="color: orange">[OPTIONAL]</span>

`--num-workers`

| Option | Description | Default |
|--------| --- |:-------:|
| `int`  | ... |    4    |

---

#### Image Labels <span style="color: orange">[OPTIONAL]</span>

`--labels`

| Option | Description | Default |
|--------| --- |:-------:|
| `bool` | ... |  True   |

<hr style="border:2px solid gray">
