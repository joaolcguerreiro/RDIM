# <span style="color: orange">Visualize Scans</span>

#### Table of Contents

1) [Run Visualization Script](#run-visualization-script)
2) [Script Arguments Overview](#script-arguments-overview)

<hr style="border:1px solid gray">

#### Run Visualization Script

1) Run script:
   ```
   python visualization.py --mode=<mode> [--lr-dir=<lr_dir>] [--gen-dir=<gen_dir>] [--gt-dir=<gt_dir>] [--augmentation=<bool>] [--num-spatial-dims=<num_spatial_dimensions>] [--plane=<plane>] [--visualization-dim=<visualization_dimension>] [--volume=<volume_index>] [--slice=<slice_index>] [--batch-size=<batch_size>] [--num-workers=<num_workers>] [--labels=<bool>]
   ```

<hr style="border:1px solid gray">

#### Script Arguments Overview

Detailed definition and description of all available visualization script arguments.

---

- Visualization Mode <span style="color: orange">[REQUIRED]</span>

`--mode`

| Option           | Description                                           |
|------------------|-------------------------------------------------------|
| `'lr_scans'`     | Display low-resolution scan counterpart               |
| `'gen_scans'`    | Display generated scan counterpart                    |
| `'gt_scans'`     | Display ground truth scan counterpart                 |
| `'lr_gen_scans'` | Display low-resolution and generated scans jointly    |
| `'lr_gt_scans'`  | Display low-resolution and ground truth scans jointly |
| `'gen_gt_scans'` | Display generated and ground truth scans jointly      |

---

- Low-Resolution Data Directory <span style="color: orange">[CONDITIONAL]</span>

`--lr-dir`

| Option | Description                            |
|--------|----------------------------------------|
| `str`  | Path to low-resolution data directory  |

Required when mode is `lr_scans`, `lr_gen_scans` or `lr_gt_scans`

---

- Super-Resolution Data Directory <span style="color: orange">[CONDITIONAL]</span>

`--gen-dir`

| Option | Description                      |
|--------|----------------------------------|
| `str`  | Path to generated data directory |

Required when mode is `gen_scans`, `lr_gen_scans` or `gen_gt_scans`

---

- Ground Truth Data Directory <span style="color: orange">[CONDITIONAL]</span>

`--gt-dir`

| Option | Description                         |
|--------|-------------------------------------|
| `str`  | Path to ground truth data directory |

Required when mode is `gt_scans`, `lr_gt_scans` or `gen_gt_scans`

---

- Data Augmentation <span style="color: orange">[OPTIONAL]</span>

`--augmentation`

| Option | Description                                               | Default |
|--------|-----------------------------------------------------------|:-------:|
| `bool` | Apply augmentation techniques to the visualization data   |  False  |

---

- Target Pixel Window for Visualization <span style="color: orange">[OPTIONAL]</span>

`--pixel-range`

| Option     | Description                                                                                     |   Default    |
|------------|-------------------------------------------------------------------------------------------------|:------------:|
| `sequence` | Window range of pixel values (min, max). For multi-channel scaling provide ((min...), (max...)) | (None, None) |

---

- Number of Spatial Dimensions <span style="color: orange">[OPTIONAL]</span>

`--num-spatial-dims`

| Option | Description                              | Default |
|--------|------------------------------------------|:-------:|
| `int`  | Number of spatial dimensions in the data |    3    |

---

- View Plane <span style="color: orange">[OPTIONAL]</span>

`--plane`

| Option       | Description           | Default |
|--------------|-----------------------|:-------:|
| `'axial'`    | Your description here |    ✓    |
| `'coronal'`  | Your description here |         |
| `'sagittal'` | Your description here |         |

---

- Visualization Spacial Dimension <span style="color: orange">[OPTIONAL]</span>

`--visualization-dim`

| Option   | Description           | Default |
|----------|-----------------------|:-------:|
| `'2D'`   | Your description here |    ✓    |
| `'3D'`   | Your description here |         |

---

- Volume Index <span style="color: orange">[OPTIONAL]</span>

`--volume`

| Option | Description                                                                  | Default |
|--------|------------------------------------------------------------------------------|:-------:|
| `int`  | Non-negative integer (n > 0) referencing volume. (-1 to display all volumes) |   -1    |

---

- Slice Index <span style="color: orange">[OPTIONAL]</span>

`--slice`

| Option | Description              | Default |
|--------|--------------------------|:-------:|
| `int`  | Slice index (-1 for all) |   -1    |

---

- Batch Size <span style="color: orange">[OPTIONAL]</span>

`--batch-size`

| Option | Description                    | Default |
|--------|--------------------------------|:-------:|
| `int`  | Number of samples to visualize |    2    |

---

- Data Loading Subprocesses <span style="color: orange">[OPTIONAL]</span>

`--num-workers`

| Option | Description                                  | Default |
|--------|----------------------------------------------|:-------:|
| `int`  | Number of subprocesses used for data loading |    4    |

---

- Image Labels <span style="color: orange">[OPTIONAL]</span>

`--labels`

| Option | Description          | Default |
|--------|----------------------|:-------:|
| `bool` | Display image labels |  True   |
