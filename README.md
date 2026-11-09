# 3D Thermal Stress Prediction with Mamba

Core model, training loss and evaluation code for 3D von Mises stress prediction. Prepare your own data using the format below; datasets and trained weights are not included.

Run all commands from the repository root.

## 1. Install the environment

Use Linux with an NVIDIA GPU and a CUDA toolkit compatible with PyTorch.

```bash
conda env create -n iccad26-release -f environment.yml
conda activate iccad26-release
python -m pip install ninja
export CUDA_HOME=/path/to/your/cuda-toolkit
bash scripts/install_extensions.sh
```

Use the bundled custom Mamba and causal-conv1d extensions. If you already have the original environment, simply run `conda activate iccad26` and skip installation. The environment file comes from that setup; installation on a clean machine has not yet been verified.

## 2. Prepare your data

Split complete models into **train / validation / test before creating patches**. All patches from one model must stay in the same subset. Use globally unique `model_*` names and retain stable design IDs in metadata, if available.

Each subset directory directly contains model folders:

```text
data/train/model_train_a/
  input_features.zarr/     # float32, shape (16, D, H, W)
  vonmises.zarr/           # float32, shape (1, D, H, W)

data/val/model_val_a/      # same file layout
data/test/model_test_a/    # same file layout
```

Input channels use zero-based indices:

| Channel | Feature |
|---|---|
| 0, 1 | x, y coordinates |
| 2 | z coordinate |
| 3 | Temperature |
| 4 | Unused; omitted by preprocessing |
| 5–10 | Six material parameters; detailed order/units to be documented |
| 11 | Thickness |
| 12 | Material-interface mask |
| 13 | Interface feature `exp(-d)`, where `d` is the nearest interface distance |
| 14, 15 | Temperature gradients in x and y |

The loader converts the 16 raw channels into 15 model inputs. Supply channel 13 already transformed as `exp(-d)`. Keep units and channel order consistent across subsets, and provide finite stress labels for training and evaluation. Geometry/simulation-to-Zarr conversion is performed separately.

For faster loading, prepare patches under each model folder:

```text
patches_40x512x512_stride10x256x256/
  input_patches.zarr/      # (N, 16, 40, 512, 512)
  vonmises_patches.zarr/   # (N, 1, 40, 512, 512)
  patches_metadata.json
```

Keep the default patch size `(40,512,512)` and stride `(10,256,256)` for the original setup. Metadata fields are `num_patches`, `d_starts`, `h_starts`, `w_starts` and `patch_shape_dhw`. Store patches in the Cartesian-product order of the starts, with w changing fastest. Include terminal patches for non-divisible dimensions; automatic slicing without prepared patches does not add them. Keep full volumes for statistics and evaluation.

## 3. Configure paths and settings

Set your subset paths in [configs/data.yaml](configs/data.yaml):

```yaml
TRAIN_DATA_DIR: ./data/train
VAL_DATA_DIR: ./data/val
DATA_DIR: ./data/test
```

Other settings are separated by purpose:

| File | Settings |
|---|---|
| [configs/train.yaml](configs/train.yaml) | Batch size, epochs, learning rate, loss and output directory |
| [configs/model.yaml](configs/model.yaml) | Model architecture |
| [configs/evaluate.yaml](configs/evaluate.yaml) | Checkpoint, GPU and test output directory |

## 4. Compute statistics and train

```bash
# Compute preprocessing statistics using training models only.
python compute_transform_stats.py

# Train on one GPU.
CUDA_VISIBLE_DEVICES=0 python train.py
```

Statistics are saved to `TRAIN_DATA_DIR/transform_stats.json`. Training, validation and testing reuse this file. The default stress scale is the sampled training P70; recompute statistics if you change the training data or percentile settings.

For multiple GPUs, use this training command instead:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py
```

Weights and logs are saved under `./outputs/experiment_TIMESTAMP/` by default. `model/best_model.pth` is selected by the lowest validation loss.

## 5. Test a trained model

Replace the checkpoint path with the file produced by your training run:

```bash
python evaluate.py --checkpoint ./outputs/experiment_TIMESTAMP/model/best_model.pth --gpu 0
```

Keep the model configuration and training statistics used for that checkpoint. Evaluation reads `DATA_DIR`, reconstructs the input sampling grid and compares predictions with the full stress labels. Results are saved to `./outputs/evaluation/test_results.json` by default. Check that all intended test models appear in the results. Add `--plot_scatter` to save scatter plots.

Run `python train.py --help` or `python evaluate.py --help` for command-line options.

Model implementation: [stress_mamba/model.py](stress_mamba/model.py). Training loss: [stress_mamba/utils.py](stress_mamba/utils.py).

## Acknowledgements

We thank [SegMamba](https://github.com/ge-xing/SegMamba), [MONAI](https://github.com/Project-MONAI/MONAI), [Mamba](https://github.com/state-spaces/mamba) and [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) for their open-source implementations.

The MONAI-derived model and bundled Mamba retain [Apache-2.0](third_party/mamba/LICENSE); causal-conv1d retains [BSD-3-Clause](third_party/causal-conv1d/LICENSE). Original copyright notices are preserved. The license for the author's original code is pending.
