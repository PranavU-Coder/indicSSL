# MSM-MAE Pre-training

[MSM-MAE](https://github.com/nttcslab/msm-mae) (Masked Spectrogram Modeling using Masked Autoencoders) is the predecessor to M2D. It applies the MAE framework to log-mel spectrograms, learning general-purpose audio representations by masking and reconstructing portions of the input spectrogram in a self-supervised manner.

M2D improves upon MSM-MAE by computing the reconstruction loss in feature space (via a momentum encoder) rather than in input space, yielding substantially better downstream performance. MSM-MAE weights are provided here for reproducibility and comparison purposes.

See also: [Masked Spectrogram Modeling using Masked Autoencoders for Learning General-purpose Audio Representation](https://proceedings.mlr.press/v166/niizumi22a.html) (HEAR 2021 NeurIPS Challenge, PMLR 2022).

## Pre-training

The weight distributed in this repository is trained for 300 epochs, consistent with the M2D training setup. Follow the data preparation steps in [data/README.md](../data/README.md) before running.

```shell
# Training command
OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 -m mae_train_audio --input_size 80x608 --patch_size 16x16 --epochs 300 --batch_size 512 --save_freq 50 --seed 7 --data_path your/ssd/data
```

> **Note:** Replace `your/ssd/data` with the path to your LMS data directory. Placing data on fast storage (SSD recommended) significantly speeds up training. If `--data_path` is omitted, the `data/` directory at the repository root is used.
