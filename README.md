# <span style="color: orange">VoxRDiff</span>

Repository ....

TODO: MAKE GIF WITH SAMPLE EVERY CHECKPOINT AND ALSO DISPLAY CHART FOR SAMPLE PIXEL DISTRIBUTION VS GROUND TRUTH PIXEL DISTRIBUTION
SINCE WE SHOW THE EVOLUTION OF ONLY ONE SAMPLE, WE ONLY SHOW THE PIXEL DISTRIBUTION OF THAT SAMPLE AND NOT THE GLOBAL PIXEL DISTRIBUTION

...

<hr style="border:2px solid orange">

# <span style="color: orange">Quick Inference</span>

#### 3D Reconstruction

   ```python
   import torch
   from monai.inferers.inferer import SlidingWindowInferer
   
   # Set patch size (D, H, W)
   patch_size = (32, 32, 32)
   
   # Define inferer (Optional)
   inferer = SlidingWindowInferer(roi_size=patch_size, overlap=0.25, mode='gaussian')
   
   # Two random volumes (batch size of 2) with 1 channel and size of 128x128x128 (D, H, W). Mean is 0 and STD is 1.
   volumes = torch.randn(2, 1, 128, 128, 128)
   
   # Avoid HTTP Error 403
   torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
   
   # Load model
   model = torch.hub.load('???', '???', pretrained=True)
   model.eval()
   
   # Generated volumes. Model expects z-score normalized input volumes.
   gen_volumes = model(volumes) if not inferer else inferer(inputs=volumes, network=model)
   ```

<hr style="border:1px solid gray">

#### 2D Reconstruction

   ```python
   import torch
   from monai.inferers.inferer import SlidingWindowInferer
   
   # Set patch size (D, H, W) = (1, H, W)
   patch_size = (1, 32, 32)
   
   ... # Same as 3D Reconstruction
   
   # Two random images (batch size of 2) with 1 channel and size of 128x128 (H, W). Mean is 0 and STD is 1.
   volumes = torch.randn(2, 1, 128, 128)
   
   # Add required depth dimension (1, H, W)
   volumes = volumes.unsqueeze(dim=-3)
   
   ... # Same as 3D Reconstruction
   
   # Load model
   model = torch.hub.load('???', '???', pretrained=True)
   model.eval()

   ... # Same as 3D Reconstruction

   # Remove depth dimension
   gen_volumes = gen_volumes.squeeze(dim=-3)
   ```

<hr style="border:1px solid gray">

#### Feature Extraction

   ```python
   import torch
   from monai.inferers.inferer import SlidingWindowInferer
   
   ... # Same as Reconstruction
   
   # Load model
   model = torch.hub.load('???', '???', pretrained=True)
   model.forward = model.extract_features
   model.eval()
   
   # Extract features. Model expects z-score normalized input volumes.
   features = model(volumes) if not inferer else inferer(inputs=volumes, network=model)
   ```

<hr style="border:2px solid orange">

# <span style="color: orange">Usage</span>

### Setup Environment

1) Clone GitHub repository:

   ```
   git clone https://github.com/??????
   cd VoxRDiff
   ```

2) Create environment:
   ```
   conda create -n srmediff python=3.10.11
   conda activate srmediff
   ```

3) Install Python dependencies (Python 3.10 is recommended):
   ```
   pip install -r requirements.txt
   ```

4) Install layout engine of Graphviz properly (https://github.com/mert-kurttutan/torchview). [REMOVE THIS]
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

<hr style="border:1px solid gray">

### Prepare Data

1) See [data/README.md](data/README.md) for details.

2) After preparing the data, navigate to `configs/<model>/config.yaml` and adjust data paths....

<hr style="border:1px solid gray">

### Run Model

In the following instructions, replace `<model>` by one of the options below:

* `esrgan`

* `srmediff`

---

#### Train and Validation

1) Navigate to `configs/<model>/config.yaml` and adjust configuration parameters needed....

2) Navigate to `src/scripts/<model>/`

3) Run ...
   ```
   # Default execution
   python train.py
   ```

   ```
   # With patch-based training (Recommended)
   python train.py --lr-patch-size=<lr_patch_size>
   ```

   ```
   # Without Validation
   python train.py --val=False
   ```

Note: If `lr_patch_size="[-1, -1, -1]"` then no cropping is employed (more computationally expensive)... Also, for `lr_patch_size` equal or smaller than `"[8, 8, 8]"` the `batch_size` is required to be greater than `1` because of discriminator batch norm.
... Also, for SRMeDiff the dimensions of the `lr_patch_size` are recommended to be powers of 2...

---

#### Load Pre-trained

1) See [pretrained_models/README.md](pretrained_models/README.md) to download available pre-trained models... or use checkpoint path from a previous experiment...

2) Navigate to `configs/<model>/config.yaml` and update pre-trained path parameters of model....

3) Navigate to `src/scripts/<model>/`

4) Run ...
   ```
   python train.py
   ```

---

#### Resume Train

1) Navigate to `experiments/<model>` and copy experiment path

2) Navigate to `src/scripts/<model>/`

3) Run ...
   ```
   python train.py --resume=<experiment_path>
   ```

    Note: Load pre-trained differs from resume training since in pre-training only the model weights are loaded and the training is reset... If you resume training the last epoch checkpoint weights will be prioritized over the pre-trained weights...

---

#### Test

1) Navigate to `experiments/<model>` and copy experiment path

2) Navigate to `src/scripts/<model>/`

3) Run ...
   ```
   python test.py --exp=<experiment_path> --epoch=<epoch>
   ```
   
Note: Results are in <experiment_path/results>... additionally check <experiment_path/logs/test> and <experiment_path/figures/test> for more details...

---

#### Inference

1) Get path of a checkpoint file...

2) Navigate to `src/scripts/<model>/`

3) Run ...
   ```
    python inference.py --checkpoint=<path_to_checkpoint> --lr-dir=<path_to_lr_dir> --save-dir=<path_to_save_dir> [--batch-size=<batch_size>] [--num-workers=<num_workers>]
   ```

<hr style="border:1px solid gray">

### Experiment Tracking (??MAYBE DELETE THIS??)

Everytime you train a model.... an experiment folder is built (except if training was resumed, thereby it is used the folder of the prior run)... the global configuration file is stored in the experiment folder to fetch the original parameters during training resume.

#### Directory Structure

the structure is the following....
<model_name>
   <date_time>
      <configs>
      <checkpoints>
      <figures>
      <logs>
      <results>

---

#### Folder Description

configs/... Contains copy of global configuration file used in the experiment. Loaded in case of training resuming.

checkpoints/ .... Contains saved checkpoints of the model during training

figures/ ....

logs/ ....

results/ .... Contains the output of running the trained model on the test set. The results are typically generated by running the trained model on the test set using the test.py script.

<hr style="border:1px solid gray">

### Visualize Results

Image visualization file...

0) Navigate to `visualization` directory:
   ```
   cd visualization
   ```

1) Run visualization script:
   ```
   python visualization.py --mode=<mode> --lr-dir=<lr_dir | None> --sr-dir=<sr_dir | None> --gt-dir=<gt_dir | None> [--augmentation=<bool>] [--plane=<plane>] [--dim=<dimension>] [--volume=<volume_index>] [--slice=<slice_index>] [--batch-size=<batch_size>] [--num-workers=<num_workers>] [--labels=<bool>]
   ```

---

#### Visualization Mode <span style="color: orange">[REQUIRED]</span>

`--mode`

| Option          | Description                                               |
|-----------------|-----------------------------------------------------------|
| `'lr_scans'`    | Display low-resolution scan counterpart                   |
| `'sr_scans'`    | Display super-resolution scan counterpart                 |
| `'gt_scans'`    | Display ground truth scan counterpart                     |
| `'lr_sr_scans'` | Display low-resolution and super-resolution scans jointly |
| `'lr_gt_scans'` | Display low-resolution and ground truth scans jointly     |
| `'sr_gt_scans'` | Display super-resolution and ground truth scans jointly   |

---

#### Low-Resolution Data Directory <span style="color: orange">[CONDITIONAL]</span>

`--lr-dir`

| Option | Description                            |
|--------|----------------------------------------|
| `str`  | Path to low-resolution data directory  |

Required when mode is `lr_scans`, `lr_sr_scans` or `lr_gt_scans`

---

#### Super-Resolution Data Directory <span style="color: orange">[CONDITIONAL]</span>

`--sr-dir`

| Option | Description                             |
|--------|-----------------------------------------|
| `str`  | Path to super-resolution data directory |

Required when mode is `sr_scans`, `lr_sr_scans` or `sr_gt_scans`

---

#### Ground Truth Data Directory <span style="color: orange">[CONDITIONAL]</span>

`--gt-dir`

| Option | Description                         |
|--------|-------------------------------------|
| `str`  | Path to ground truth data directory |

Required when mode is `gt_scans`, `lr_gt_scans` or `sr_gt_scans`

---

#### Data Augmentation <span style="color: orange">[OPTIONAL]</span>

`--augmentation`

| Option | Description                                               | Default |
|--------|-----------------------------------------------------------|:-------:|
| `bool` | Apply augmentation techniques to the visualization data   |  False  |

---

#### Target Pixel Window for Visualization <span style="color: orange">[OPTIONAL]</span>

`--pixel-range`

| Option     | Description                                                                                     |   Default    |
|------------|-------------------------------------------------------------------------------------------------|:------------:|
| `sequence` | Window range of pixel values (min, max). For multi-channel scaling provide ((min...), (max...)) | (None, None) |

---

#### View Plane <span style="color: orange">[OPTIONAL]</span>

`--plane`

| Option       | Description           | Default |
|--------------|-----------------------|:-------:|
| `'axial'`    | Your description here |    ✓    |
| `'coronal'`  | Your description here |         |
| `'sagittal'` | Your description here |         |

---

#### Visualization Spacial Dimension <span style="color: orange">[OPTIONAL]</span>

`--dim`

| Option   | Description           | Default |
|----------|-----------------------|:-------:|
| `'2D'`   | Your description here |    ✓    |
| `'3D'`   | Your description here |         |

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
|--------|-------------|:-------:|
| `int`  | ...         |   -1    |

---

#### Batch Size <span style="color: orange">[OPTIONAL]</span>

`--batch-size`

| Option | Description | Default |
|--------|-------------|:-------:|
| `int`  | ...         |    2    |

---

#### Data Loading Subprocesses <span style="color: orange">[OPTIONAL]</span>

`--num-workers`

| Option | Description | Default |
|--------|-------------|:-------:|
| `int`  | ...         |    4    |

---

#### Image Labels <span style="color: orange">[OPTIONAL]</span>

`--labels`

| Option | Description | Default |
|--------|-------------|:-------:|
| `bool` | ...         |  True   |

<hr style="border:2px solid orange">

# <span style="color: orange">Citation</span>

If you find this project useful, please consider citing the paper:
```
@article{
...
}
```

<hr style="border:2px solid orange">

# <span style="color: orange">License</span>

This project is licensed under MIT license. See [LICENSE](LICENSE) for details.
