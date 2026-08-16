# indicSSL

This repository contains a controlled comparison of masked-modeling self-supervised objectives for low-resource Indic speech.

> This repository is a *controlled comparison / reproducibility study*,  the contribution is the matched experimental protocol, the low-resource Indic setting, and the empirical finding below.

---

## Abstract

Three self-supervised masked-modeling objectives: **masked reconstruction** , **masked-only latent prediction**  & **whole-input latent prediction** are trained under *identical* conditions (same ViT-S/16 encoder, same log-mel front-end, same 50k-steps, same frozen-probe evaluation) and evaluated on Indic **language identification (LID)** and **speaker identification (SID)** using the IndicSUPERB / Kathbath benchmark.

**Under matched conditions, masked reconstruction outperforms both latent-prediction objectives on both tasks**, by margins that far exceed seed noise. Spectrograms lack the trivial local pixel-redundancy that gives signal reconstruction an advantage over the objectives, so in audio the reconstruction target's local structure (formants, harmonics) is itself the task-relevant signal.

---

## Results

All numbers are `n=3` seeds (42, 1, 2), read directly from downstream prediction files on the Kathbath valid set (`n=32176`).

| System    | Objective                        | LID (mean ± sd) | SID (mean ± sd) |
| --------- | -------------------------------- | --------------- | --------------- |
| IndicMAE  | masked reconstruction            | **64.51 ± 0.28** | **85.58 ± 0.44** |
| M2D       | masked-only latent prediction    | 61.09 ± 0.66    | 80.54 ± 0.44    |
| IndicJEPA | whole-input latent prediction    | 56.10 ± 3.74    | 77.77 ± 3.10    |

<details>
<summary>Per-seed accuracies</summary>

| System    | LID s42 / s1 / s2       | SID s42 / s1 / s2       |
| --------- | ----------------------- | ----------------------- |
| IndicMAE  | 64.37 / 64.83 / 64.32   | 85.78 / 85.90 / 85.05   |
| M2D       | 61.84 / 60.78 / 60.65   | 81.05 / 80.24 / 80.32   |
| IndicJEPA | 58.14 / 51.79 / 58.38   | 78.47 / 74.33 / 80.51   |

</details>

---

## Repository structure

```
indicJEPA/                      
├── src/indicJEPA/             
│   ├── ajepa.py                 #   ViTEncoder, Block, sincos pos embed, AudioJEPA
│   ├── mae.py                   #   IndicMAE (ViTEncoder + MAEDecoder), norm_pix_loss=True
│   ├── train.py                 #   IndicJEPA trainer (config-driven)
│   ├── train_mae.py             #   IndicMAE trainer (config-driven)
│   └── indic.py, prep.py        #   Indic dataset 
├── IndicSUPERB/                 # s3prl-based downstream eval 
│   ├── s3prl/                   #   vendored s3prl; active wrapper slot: upstream/example/expert.py
│   ├── result/downstream/       
│   ├── utilities/               #   m4a to 16 kHz wav (structure.py), ASR preprocessing
│   ├── expert_m2d.py            #   M2D s3prl wrapper
│   └── smoke_mae.py, smoke_wrapper.py, chunk_lid_train.py
├── m2d/                         # M2D pretraining (separate venv)
├── configs/                     # persistent run configs
├── logs/                        # per-seed logs & prediction files for review
│   ├── pretrain/{indicmae,m2d,indicjepa}/
│   └── downstream/<run>/
├── data/                        # Kathbath + IndicVoices       (git-ignored)
│   └── install.sh               # dataset installation for anyone who needs it
├── expert.py                    # IndicJEPA s3prl wrapper (reads target_encoder)
├── pyproject.toml, uv.lock, .python-version   # uv-managed project
├── CODE_OF_CONDUCT.md, CONTRIBUTING.md, .pre-commit-config.yaml
├── LICENSE
└── README.md
```

---

## Installation

Two isolated virtual environments are used and **must not be mixed**:

| venv                          | used for                                                        |
| ----------------------------- | -------------------------------------------------------------- |
| `IndicSUPERB/.venv-superb`    | s3prl, all downstream evals, IndicMAE + IndicJEPA training, analysis |
| `m2d/.venv`                   | M2D pretraining **only** (requires `timm==0.9.2`)              |

```bash
# downstream / MAE / JEPA environment
python -m venv IndicSUPERB/.venv-superb
source IndicSUPERB/.venv-superb/bin/activate
pip install -e IndicSUPERB/s3prl

# M2D environment (note the timm pin, the repo default of 0.4.5 is not correct)
python -m venv m2d/.venv
source m2d/.venv/bin/activate
pip install -r m2d/requirements.txt
pip install 'timm==0.9.2'
```

Every fresh shell should have the project root exported to avoid any import issue during runtime:

```bash
export INDICJEPA_ROOT=~/Projects/indicJEPA
```

While this naming convention may seem really odd, this was because the study's initial aim was to introduce IndicJEPA however since then the objective has shifted, codebase has had minimal changes as it is quite extensive and making any changes can severely impact the ability for any reproduction and so on.

Use `tmux` if possible, completely optional.

---

## Data

### Kathbath (downstream: LID / SID)

Fetch and extract the IndicSUPERB / Kathbath audio with the included script:

```bash
bash data/install.sh --dest $INDICJEPA_ROOT/data          # full clean split (~92 GB)
bash data/install.sh  --dest $INDICJEPA_ROOT/data --no-train   # eval-only (skips 85 GB train tar)
```

Then convert m4a to 16 kHz wav (needs `ffmpeg`):

```bash
bash data/install.sh  --dest $INDICJEPA_ROOT/data --convert --superb-dir $INDICJEPA_ROOT/IndicSUPERB
```

The downstream code reads from `$INDICJEPA_ROOT/data/kb_data_clean_wav`.

### IndicVoices (pretraining)

Pretraining uses ~52 h of IndicVoices (**31,883 clips**, 12 languages). The M2D pipeline expects a CSV manifest and pre-computed log-mel spectrograms:

```
m2d/data/files_indicvoices.csv
m2d/data/indicvoices_lms/*.npy
```

---

## Method

All three systems share the **same encoder**

**Shared encoder** is a `ViTEncoder` (in `src/indicJEPA/ajepa.py`): 384-dim embed, depth 12, 6 heads, patch 16×16, MLP ratio 4.0, **no cls token**, 2D sincos positional embedding, ~21.4 M parameters. Verified identical across all three checkpoints (MAE 21.44 M, JEPA context encoder 21.44 M, M2D blocks 21.29 M). Masked encoding is supported via `keep_indices` (visible-patches-only).

**Input**  is a 128×256 log-mel (128 mel bands × 256 time bins), 16 kHz, 5.0 s clips, patch 16×16 → grid 8×16 = 128 patches.

| System    | Prediction target                          | Decoder / predictor        | EMA teacher | Mask ratio                  |
| --------- | ------------------------------------------ | -------------------------- | ----------- | --------------------------- |
| IndicMAE  | raw mel pixels (MSE, `norm_pix_loss=True`) | 192-dim × 6 (**2.82 M**)   | no          | 0.4–0.6 (per batch)         |
| IndicJEPA | latent of masked patches; target sees whole input | 192-dim × 6 (**2.84 M**) | yes         | 0.4–0.6 (Audio-JEPA convention) |
| M2D       | latent of masked patches; target sees only masked | 512-dim × 8 (**25.22 M**) | yes         | 0.7, `masking_start_epoch=50` |

---

## Pretraining

**IndicMAE** (from `.venv-superb`):

```bash
cd $INDICJEPA_ROOT && source IndicSUPERB/.venv-superb/bin/activate
cp runs/indicmae_s42/config.yaml configs/mae_sN.yaml
sed -i 's/seed: 42/seed: N/; s/run_name: null/run_name: indicmae_sN/' configs/mae_sN.yaml
python src/indicJEPA/train_mae.py --config configs/mae_sN.yaml 2>&1 | tee configs/mae_sN.log
# should see params 21.44M / 2.82M, loss ~0.4 (bf16-normalised, confirms norm_pix active) else something went woefully wrong
```

**IndicJEPA** (from `.venv-superb`, config-driven, same pattern):

```bash
python src/indicJEPA/train.py --config configs/jepa_sN.yaml 2>&1 | tee configs/jepa_sN.log
```

**M2D** (from `m2d/.venv`):

```bash
cd $INDICJEPA_ROOT/m2d && source .venv/bin/activate
python train_audio.py --model m2d_vit_small --data_path data --csv_main data/files_indicvoices.csv \
  --input_size 128x256 --patch_size 16x16 --batch_size 64 --epochs 100 --warmup_epochs 20 \
  --blr 0.0003 --weight_decay 0.05 --clip_grad 3.0 --mask_ratio 0.7 --loss_fn norm_mse \
  --ema_decay 0.99999 --ema_decay_init 0.99995 --masking_start_epoch 50 --eval_after 50 --save_freq 20 \
  --seed N --output_dir m2d_vit_small-128x256p16x16p16k-seedN-bs64-e100 --log_dir <same>
```

---

## Downstream evaluation

Frozen encoder & a fixed lightweight linear probe (mean-pooled features), held **identical across all methods** so differences reflect the encoder, not probe capacity. Tasks: **LID** (12 languages) and **SID** (229 speakers) on the Kathbath valid set.

The active s3prl wrapper slot is `s3prl/upstream/example/expert.py` **please copy the correct wrapper in before each run**:

| System    | Wrapper to copy into the active slot                          |
| --------- | ------------------------------------------------------------ |
| IndicMAE  | `s3prl/upstream/example/expert_mae_load.py`                  |
| IndicJEPA | `expert.py` (repo root; reads `target_encoder`)             |
| M2D       | `IndicSUPERB/expert_m2d.py`                                  |

```bash
cd $INDICJEPA_ROOT/IndicSUPERB
cp s3prl/upstream/example/expert_mae_load.py s3prl/upstream/example/expert.py   # taking IndicMAE as an example
source .venv-superb/bin/activate && export INDICJEPA_ROOT=~/Projects/indicJEPA
CK=$INDICJEPA_ROOT/checkpoints/indicmae_sN_final.pt
DATA=$INDICJEPA_ROOT/data/kb_data_clean_wav

# LID (default projector)
python s3prl/run_downstream.py -n indicmae_lid_sN -m train -u customized_upstream -k $CK \
  -d indic_lid -c s3prl/downstream/indic_lid/config.yaml \
  -o "config.downstream_expert.datarc.file_path=$DATA,,config.runner.total_steps=50000,,config.downstream_expert.datarc.num_workers=16"

# SID (projector_dim=256)
python s3prl/run_downstream.py -n indicmae_sid_sN -m train -u customized_upstream -k $CK \
  -d indic_sid -c s3prl/downstream/indic_sid/config.yaml \
  -o "config.downstream_expert.datarc.file_path=$DATA,,config.runner.total_steps=50000,,config.downstream_expert.datarc.num_workers=16,,config.downstream_expert.modelrc.projector_dim=256"
```

> If you re-launch into an existing run dir, `rm -rf result/downstream/<run_name>` first otherwise causes a "0 files found" / class-count crash.

---

## Pretrained checkpoints

Consolidated under `checkpoints/`

| File                              | System / seed          |
| --------------------------------- | ---------------------- |
| `indicmae_s{42,1,2}_final.pt`     | IndicMAE               |
| `m2d_s{42,1,2}_final.pth`         | M2D                    |
| `indicjepa_s{42,1,2}_final.pt`    | IndicJEPA              |

---

## Citations

[WIP]

---

## License

Code in this repository is released under the **MIT License**

Vendored and derived components retain their own licenses, particularly s3prl, the Audio-MAE / M2D reference implementations, and the Kathbath dataset (released by AI4Bharat under CC0).

---

## Acknowledgements

Thanks to AI4Bharat for the IndicSUPERB / Kathbath benchmark, to the s3prl project for the downstream evaluation framework, and to the authors of Audio-MAE, M2D, and Audio-JEPA whose objectives are compared here.

## References

- Audio-MAE (Masked Autoencoders that Listen) — Huang et al., 2022 — [arXiv:2207.06405](https://arxiv.org/abs/2207.06405)
- M2D (Masked Modeling Duo) — Niizumi et al., 2022 — [arXiv:2210.14648](https://arxiv.org/abs/2210.14648)
- M2D: Towards a Universal Audio Pre-training Framework — Niizumi et al., 2024 — [arXiv:2404.06095](https://arxiv.org/abs/2404.06095)
- I-JEPA (the vision intuition inverted here) — Assran et al., 2023 — [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)
- A-JEPA (Joint-Embedding Predictive Architecture Can Listen) — Fei et al., 2023 — [arXiv:2311.15830](https://arxiv.org/abs/2311.15830)
- Audio-JEPA — Tuncay et al., 2025 — [arXiv:2507.02915](https://arxiv.org/abs/2507.02915)
- WavJEPA (raw-waveform JEPA) — Yuksel et al., 2025 — [arXiv:2509.23238](https://arxiv.org/abs/2509.23238)
- IndicSUPERB / Kathbath — Javed et al., 2022 — [arXiv:2208.11761](https://arxiv.org/abs/2208.11761)