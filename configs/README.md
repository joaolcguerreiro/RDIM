# <span style="color: orange">Configuration Guide</span>

#### Table of Contents

1) [Overview](#overview)
2) [Data Configuration](#data-configuration)
3) [Model Configuration](#model-configuration)
4) [Learning Stage Configuration](#learning-stage-configuration)
5) [Evaluation Stage Configuration](#evaluation-stage-configuration)

<hr style="border:1px solid gray">

#### Overview

The `config.yaml` file defines all configurations related to:

* Dataset paths and loading
* Model architecture and diffusion parameters
* Training hyperparameters
* Evaluation settings

<hr style="border:1px solid gray">

#### Data Configuration

Defines dataset paths and data loading settings.

| Parameter                     | Description                                                                | Default |
|-------------------------------|----------------------------------------------------------------------------|:-------:|
| `DATA/TRAINING/LR_DIR_PATH`   | Path to Low-Resolution training data directory                             |    -    |
| `DATA/TRAINING/HR_DIR_PATH`   | Path to High-Resolution training data directory                            |    -    |
| `DATA/VALIDATION/LR_DIR_PATH` | Path to Low-Resolution validation data directory                           |    -    |
| `DATA/VALIDATION/HR_DIR_PATH` | Path to High-Resolution validation data directory                          |    -    |
| `DATA/TESTING/LR_DIR_PATH`    | Path to Low-Resolution testing data directory                              |    -    |
| `DATA/TESTING/HR_DIR_PATH`    | Path to High-Resolution testing data directory                             |    -    |
| `NUM_WORKERS`                 | Number of CPU workers for data loading                                     |    8    |
| `GPU_IDS`                     | List of GPU device IDs, e.g., [0,1] (~ defaults to all available devices). |    ~    |

> Note: Validation data split is optional and is only used during the training stage. Other splits are required depending on the phase being executed (training or testing).

> Note: See [data/README.md](data/README.md) for an illustration of a valid dataset directory structure. This convention is not mandatory.

<hr style="border:1px solid gray">

#### Model Configuration

Contains architecture and diffusion-related parameters.

* General Model Parameters

| Parameter                | Description                                        | Default |
|--------------------------|----------------------------------------------------|:-------:|
| `NUM_SPATIAL_DIMENSIONS` | Number of spatial dimensions (e.g., 3 for 3D data) |    3    |
| `SCALE_FACTOR`           | Upsampling scale factor (1 for denoising tasks)    |    1    |
| `IN_CHANNELS`            | Number of input channels                           |    1    |
| `OUT_CHANNELS`           | Number of output channels                          |    1    |
| `LR_CONDITION_ENCODING`  | Apply LR condition encoding                        |  False  |

---

* Gaussian Diffusion Parameters

| Parameter                    | Description                                                                                                                |  Default   |
|------------------------------|----------------------------------------------------------------------------------------------------------------------------|:----------:|
| `NUM_TIMESTEPS`              | Number of diffusion timesteps during training (Size of forward chain)                                                      |    100     |
| `NUM_SAMPLING_TIMESTEPS`     | Number of diffusion timesteps during sampling (Size of reverse chain)                                                      |     1      |
| `SAMPLING_TIMESTEP_SCHEDULE` | Schedule type for sampling timesteps                                                                                       |  'LINEAR'  |
| `BETA_SCHEDULE`              | Schedule type for beta schedule                                                                                            | 'CIRCULAR' |
| `GAMMA`                      | Diffusion hyperparameter that controls the strength of the variance                                                        |    0.0     |
| `ETA`                        | Diffusion hyperparameter that enables interpolation between deterministic (η=0.0) and stochastic (η=1.0) reverse processes |    0.0     |

---

* Noise Predictor Parameters

| Parameter             | Description                                           |          Default           |
|-----------------------|-------------------------------------------------------|:--------------------------:|
| `NUM_INNER_FEATURES`  | Number of base features inside the U-Net-like network |            128             |
| `FEATURE_MULTIPLIERS` | Feature growth multipliers per U-Net-like level       |        [1, 2, 4, 8]        |
| `NUM_RB_BLOCKS`       | Number of residual blocks                             |             2              |
| `NUM_RB_CONV_BLOCKS`  | Number of convolutional layers per residual block     |             2              |
| `ATTENTION_LEVELS`    | Apply attention at specific levels                    | [False, False, True, True] |
| `NUM_ATTENTION_HEADS` | Number of attention heads                             |             16             |
| `NUM_GROUPS`          | Number of groups for group normalization              |             16             |
| `PRE_TRAINED_PATH`    | Path to pretrained noise predictor weights [OPTIONAL] |                            |

> Note: The path for pre-trained weights is optional. Fill it if you want to finetune or benefit from saved models.

---

* LR Encoder Parameters

| Parameter                     | Description                                        | Default |
|-------------------------------|----------------------------------------------------|:-------:|
| `NUM_FEATURES`                | Number of base features inside network             |   64    |
| `NUM_RRDB_BLOCKS`             | Number of residual-in-residual dense blocks (RRDB) |    1    |
| `NUM_RRDB_DB_BLOCKS`          | Number of dense blocks per RRDB block              |    2    |
| `NUM_RRDB_DB_CONV_BLOCKS`     | Number of convolutional layers per dense block     |    3    |
| `NUM_RRDB_DB_GROWTH_FEATURES` | Number of growth features in dense block           |   32    |
| `PRE_TRAINED_PATH`            | Path to pretrained LR encoder weights [OPTIONAL]   |         |

> Note: The path for pre-trained weights is optional. Fill it if you want to finetune or benefit from saved models.

---

* Sampling Timestep Schedule Options

| Option        | Description                 | Default |
|---------------|-----------------------------|:-------:|
| `'LINEAR'`    | Linear timestep schedule    |    ✓    |
| `'GEOMETRIC'` | Geometric timestep schedule |         |

---

* Beta Schedule Options

| Option                             | Description                            | Parameter `P` Default | Default |
|------------------------------------|----------------------------------------|:---------------------:|:-------:|
| `'GENERAL_SIGMOID'`                | Sigmoid-like β-schedule                |           2           |         |
| `'GENERAL_SIGMOID_ANTIDERIVATIVE'` | Sigmoid-like antiderivative β-schedule |          10           |         |
| `'CIRCULAR'`                       | Circular β-schedule                    |           5           |    ✓    |
| `'PARABOLIC'`                      | Parabolic β-schedule                   |           2           |         |

<hr style="border:1px solid gray">

#### Learning Stage Configuration

Defines training parameters and optimization settings.

* Noise Predictor Training Parameters

| Parameter                         | Description                                                   |  Default   |
|-----------------------------------|---------------------------------------------------------------|:----------:|
| `EPOCHS`                          | Number of training epochs                                     |    1000    |
| `BATCH_SIZE`                      | Mini-batch size                                               |     16     |
| `NUM_GRADIENT_ACCUMULATION_STEPS` | Number of gradient accumulation steps before optimizer update |     1      |
| `LR_PATCH_SIZE`                   | Low-resolution patch size ([-1, -1, -1] for full images)      | [128, 128] |
| `REPEAT_SAMPLE_FACTOR`            | Number of times each sample is repeated per epoch             |     32     |
| `AUGMENTATION`                    | Apply data augmentation                                       |   False    |
| `PERCEPTUAL_LOSS`                 | Image quality metric for perceptual loss                      |  'LPIPS'   |
| `FIX_LR_ENCODER`                  | Freeze LR encoder during noise predictor training             |    True    |
| `SAVE_INTERVAL`                   | Epoch interval to save checkpoints                            |    100     |

> Note: Patch-based training is not employed when `LR_PATCH_SIZE=[-1, -1, -1]`. In this scenario the training is done on full images, which is computationally more expensive.

> Note: For denoising tasks, the `LR_PATCH_SIZE` equals the high-resolution (HR) patch-size, because `SCALE_FACTOR=1`.

---

* Perceptual Loss Options

| Option    | Description                   | Default |
|-----------|-------------------------------|:-------:|
| `'LPIPS'` | Learned perceptual similarity |    ✓    |
| `'SSIM'`  | Structural similarity         |         |

---

* Noise Predictor Optimizer Parameters

| Parameter       | Description       | Default |
|-----------------|-------------------|:-------:|
| `LEARNING_RATE` | Learning rate     | 1.0e-4  |
| `WEIGHT_DECAY`  | Weight decay      |    0    |
| `BETA_1`        | Adam β₁ parameter |   0.9   |
| `BETA_2`        | Adam β₂ parameter |  0.999  |

---

* Noise Predictor Scheduler Parameters

| Parameter   | Description           | Default |
|-------------|-----------------------|:-------:|
| `ETA_MIN`   | Minimum learning rate | 1.0e-9  |

---

* Noise Predictor Loss Weights

| Parameter         | Description                              | Default |
|-------------------|------------------------------------------|:-------:|
| `DIFFUSION_LOSS`  | Weight to control diffusion/content loss |   1.0   |
| `PERCEPTUAL_LOSS` | Weight to control perceptual loss        | 5.0e-2  |

---

* LR Encoder Pre-training Parameters

| Parameter                         | Description                                                   | Default |
|-----------------------------------|---------------------------------------------------------------|:-------:|
| `EPOCHS`                          | Number of training epochs (-1 to skip pre-training)           |   -1    |
| `BATCH_SIZE`                      | Mini-batch size                                               |   16    |
| `NUM_GRADIENT_ACCUMULATION_STEPS` | Number of gradient accumulation steps before optimizer update |    1    |
| `REPEAT_SAMPLE_FACTOR`            | Number of times each sample is repeated per epoch             |   32    |

---

* LR Encoder Optimizer Parameters

| Parameter       | Description       | Default |
|-----------------|-------------------|:-------:|
| `LEARNING_RATE` | Learning rate     | 1.0e-4  |
| `WEIGHT_DECAY`  | Weight decay      |    0    |
| `BETA_1`        | Adam β₁ parameter |   0.9   |
| `BETA_2`        | Adam β₂ parameter |  0.999  |

---

* LR Encoder Scheduler Parameters

| Parameter   | Description  | Default |
|-------------|--------------|:-------:|
| `STEP_SIZE` | Step size    |   200   |
| `GAMMA`     | Decay factor |   0.5   |

---

* LR Encoder Loss Weights

| Parameter         | Description                       | Default |
|-------------------|-----------------------------------|:-------:|
| `CONTENT_LOSS`    | Weight to control content loss    |   1.0   |
| `PERCEPTUAL_LOSS` | Weight to control perceptual loss | 1.0e-2  |

<hr style="border:1px solid gray">

#### Evaluation Stage Configuration

Settings for validation and testing stages.

* Validation Parameters

| Parameter                         | Description                                               | Default |
|-----------------------------------|-----------------------------------------------------------|:-------:|
| `BATCH_SIZE`                      | Number of samples processed concurrently during inference |    1    |
| `PATCH_SLIDING_WINDOW_BATCH_SIZE` | Number of sliding windows processed concurrently          |   16    |

---

* Testing Parameters

| Parameter                         | Description                                               | Default |
|-----------------------------------|-----------------------------------------------------------|:-------:|
| `BATCH_SIZE`                      | Number of samples processed concurrently during inference |    1    |
| `PATCH_SLIDING_WINDOW_BATCH_SIZE` | Number of sliding windows processed concurrently          |   16    |
