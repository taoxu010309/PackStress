# Bundled custom BiMamba-v3

This is the supplied custom Mamba 1.0.1 source used by this project's 3D model. It is not the stock PyPI package. The constructor requires `bimamba_type="v3"` and compatible `nslices`; see `stress_mamba/model.py` in the repository root for its actual use.

Use the existing `iccad26` environment, or follow [the environment guide](../../README.md#1-install-the-environment) and run `bash scripts/install_extensions.sh` from the repository root. The setup script builds these sources; stock wheels, stock Mamba examples and language-model checkpoints are not a substitute for this custom implementation.

The core uses the fused full-sequence v3 path. Streaming/generation helpers and other inherited APIs are not part of the thermal-stress workflow; they are retained for package imports and source provenance. This distribution contains no upstream assets, benchmark suites or pretrained models.

Provenance references: [Mamba](https://github.com/state-spaces/mamba), [Mamba paper by Albert Gu and Tri Dao](https://arxiv.org/abs/2312.00752), and [SegMamba](https://github.com/ge-xing/SegMamba). The exact commit and full local modification history were not available in the supplied source snapshot. Original file headers, [LICENSE](LICENSE) and [AUTHORS](AUTHORS) are retained. This guide replaces the inherited stock-package instructions for this bundled distribution.
