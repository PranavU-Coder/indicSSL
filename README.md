# indicSSL

This repository contains a controlled comparison of masked-modeling self-supervised objectives for low-resource Indic speech.

> This repository is a *controlled comparison / reproducibility study*,  the contribution is the matched experimental protocol, the low-resource Indic setting, and the empirical finding below.

---

## Abstract

Self-supervised masked modeling has split into two families: objectives that reconstruct the masked input signal, and objectives that predict a learned latent representation of it. In vision, latent prediction is widely regarded as the stronger choice, on the grounds that reconstruction spends capacity on low-level detail. This repository tests whether that ordering transfers to low-resource speech.

Three masked-modeling objectives: **masked spectrogram reconstruction** (IndicMAE), **masked-only latent prediction** (M2D) & **whole-input, I-JEPA-style latent prediction** (IndicJEPA) are pretrained on ~60 h of unlabeled speech in 12 Indic languages, holding fixed the encoder (ViT-S/16), the pretraining budget (50k steps), a shared log-mel front-end, and a lightweight probe on the frozen encoder. They are evaluated on IndicSUPERB / Kathbath **language identification (LID)** and **speaker identification (SID)**.

---

## Results

Frozen-encoder probe accuracy (%), `n=3` pretraining seeds (42, 1, 2), each probed once per task. All numbers are computed directly from the probe's per-utterance prediction files on the Kathbath valid set (`n=32,176`), using the final probe checkpoint (step 50k, so no checkpoint was selected on the evaluation set).

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

The weakest IndicMAE run (64.32 LID, 85.05 SID) is above the strongest run of either latent objective (61.84 LID, 81.05 SID, both M2D).

</details>

---

## Repository structure

```
indicJEPA/                      
├── src/indicJEPA/             
│   ├── ajepa.py                 #   ViTEncoder, Block, sincos pos embed, AudioJEPA
│   ├── mae.py                   #   IndicMAE (ViTEncoder & MAEDecoder), norm_pix_loss=True
│   ├── train.py                 #   IndicJEPA trainer (config-driven)
│   ├── train_mae.py             #   IndicMAE trainer (config-driven)
│   ├── indic.py                 #   IndicVoices streaming to cached log-mel tensors
│   └── prep.py                  #   shared mel-parameter solver (compute_mel_params)
├── IndicSUPERB/                 # s3prl-based downstream eval 
│   ├── s3prl/                   #   vendored s3prl; active wrapper slot: upstream/example/expert.py
│   ├── result/downstream/       
│   ├── utilities/               #   m4a to 16 kHz wav (structure.py), ASR preprocessing
│   ├── expert_m2d.py            #   M2D s3prl wrapper
│   └── smoke_mae.py, smoke_wrapper.py, chunk_lid_train.py
├── m2d/                         # M2D pretraining (separate venv)
├── configs/                     # default.yaml & config loader / overrides
├── checkpoints/                 # final pretrained checkpoints (git-ignored)
├── runs/                        # per-run training outputs (git-ignored)
├── logs/                        # per-seed logs & prediction files for review
│   ├── pretrain/{indicmae,m2d,indicjepa}/
│   └── downstream/<run>/
├── data/                        # Kathbath & IndicVoices       (git-ignored)
│   └── install.sh               # dataset installation for anyone who needs it
├── expert.py                    # IndicJEPA s3prl wrapper (reads target_encoder)
├── tests/                       # pytest suite (run by CI with ruff format --check)
├── .github/                     # CI workflows, issue / PR templates
├── pyproject.toml, uv.lock, .python-version   # uv-additions
├── CODE_OF_CONDUCT.md, CONTRIBUTING.md
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

The repository and package are still named `indicJEPA` because the study originally set out to introduce IndicJEPA. The objective has since shifted to a controlled comparison, but the code was deliberately left largely unchanged so that the reported runs stay reproducible.

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

Pretraining uses ~60 h of IndicVoices (**31,883 utterances**): 5 h from each of the 12 languages covered by IndicSUPERB (Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Sanskrit, Tamil, Telugu, Urdu), so pretraining and evaluation span the same languages. Utterances average 6.8 s (median 5.9 s). IndicMAE and IndicJEPA read cached log-mel tensors built by streaming IndicVoices from Hugging Face (needs `huggingface-cli login` or `--token`):

```bash
python src/indicJEPA/indic.py --output_dir ./cache/indicvoices --config configs/default.yaml
```

This writes `cache/indicvoices/dev/*.pt`, which is the path set by `data.dataset_path` in `configs/default.yaml`.

The M2D pipeline instead expects a CSV manifest and pre-computed log-mel spectrograms:

```
m2d/data/files_indicvoices.csv
m2d/data/indicvoices_lms/*.npy
```

---

## Method

All three systems share the **same encoder**, which is the controlled variable of the study.

**Shared encoder** is a `ViTEncoder` (in `src/indicJEPA/ajepa.py`): 384-dim embed, depth 12, 6 heads, patch 16×16, MLP ratio 4.0, fixed 2D sincos positional embedding, ~21.4 M parameters. IndicMAE and IndicJEPA have **no cls token**; M2D carries one internally (its weights require it), but it is dropped before pooling, so all three are evaluated on patch tokens only. Verified from the saved checkpoints: the MAE encoder, JEPA context encoder and M2D online encoder each have 21.44 M parameters, M2D differing only by the 768 parameters of its cls token and that token's positional embedding (counts include the fixed sincos tables). Masked encoding is supported via `keep_indices` (visible-patches-only).

**Input** is a 128×256 log-mel (128 mel bands × 256 time frames), 16 kHz. Each utterance is converted to log-mel once, and a random 5.0 s window is cropped from it at every step. Utterances shorter than 5.0 s (42% of the pretraining set) are end-padded. Patch 16×16 → grid 8×16 = 128 patches. M2D trains on the same cached spectrograms (converted only in file format). Two per-method differences remain: IndicMAE / IndicJEPA pad with the utterance's minimum log-mel value and normalise each window by its own mean/std, whereas M2D pads with zero and uses fixed dataset-level statistics stored in its checkpoint.

| System    | Prediction target                          | Target encoder sees | Decoder / predictor        | EMA teacher | Mask ratio          |
| --------- | ------------------------------------------ | ------------------- | -------------------------- | ----------- | ------------------- |
| IndicMAE  | per-patch normalised mel patches (MSE, `norm_pix_loss=True`) | — | 192-dim × 6 (**2.82 M**)   | no          | U(0.4, 0.6) per batch |
| IndicJEPA | latents of masked patches (squared ℓ2)    | whole input         | 192-dim × 6 (**2.84 M**)   | yes, τ cosine 0.996 → 1.0 | U(0.4, 0.6) per batch |
| M2D       | standardised latents of masked patches (cosine) | masked patches only | 512-dim × 8 (**25.68 M**) | yes, τ linear 0.99995 → 0.99999 | 0.7 fixed |

- **Capacity matching:** IndicMAE and IndicJEPA are matched in head capacity (within 0.03 M) and share the same masking distribution, so for this pair the only substantive difference is the target (signal vs. latent). This is the controlled comparison. M2D's decoder is ~9× larger, which is **not** controlled for.
- **Masking:** all three use unstructured random masks from the first step (IndicJEPA uses single random masks rather than I-JEPA's multi-block masking). M2D keeps its reference default of 0.7. Its authors report speech tasks peaking at 0.6, but that gain (1.8 SID / 0.2 LID points) is smaller than IndicMAE's lead over M2D here. The M2D script's `masking_start_epoch=50` controls only the hint schedule of its weighted-masking variants, which are not enabled.
- **EMA targets:** IndicJEPA's target encoder starts as a copy of the online encoder, and the EMA covers all of it. In M2D the target's Transformer blocks and final LayerNorm are randomly initialised and are the only averaged weights; the patch / positional embeddings and cls token are shared with the online encoder.
- **Optimisation:** IndicMAE and IndicJEPA use AdamW (β = (0.9, 0.95), wd 0.05), batch size 64, lr 3e-4 with 1k-step linear warmup from 1e-6 then cosine decay to 0, grad clip 1.0, bf16. M2D keeps its reference recipe: the same AdamW settings but no decay on biases / norms, blr 3e-4 scaled to 7.5e-5 absolute, a 20-epoch (~10k-step) warmup then cosine decay, clip 3.0, fp16.

---

## Pretraining

**IndicMAE** (from `.venv-superb`):

```bash
cd $INDICJEPA_ROOT && source IndicSUPERB/.venv-superb/bin/activate
python src/indicJEPA/train_mae.py --config configs/default.yaml
# should see params 21.44M / 2.82M, loss ~0.4 (bf16-normalised, confirms norm_pix active) else something went woefully wrong

# other seeds (reported: 42, 1, 2)
python src/indicJEPA/train_mae.py --config configs/default.yaml --seed 1
```

> `train_mae.py` takes only `--config` and an optional `--seed`. Data path, checkpoint dir and seed come from `configs/default.yaml` (`data.dataset_path`, `paths.checkpoint_dir`, `training.seed`).

**IndicJEPA** (from `.venv-superb`):

```bash
python src/indicJEPA/train.py --config configs/default.yaml \
  --data_dir ./cache/indicvoices --output_dir ./runs/indicjepa_sN \
  --override training.seed=N 2>&1 | tee runs/indicjepa_sN.log

# resume from a checkpoint
python src/indicJEPA/train.py --config configs/default.yaml \
  --data_dir ./cache/indicvoices --output_dir ./runs/indicjepa_sN \
  --resume ./runs/indicjepa_sN/ckpt_step20000.pt
```

> Unlike `train_mae.py`, `train.py` requires `--data_dir` and `--output_dir`; any config key can be overridden with `--override key.subkey=value`.

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

All heads are discarded and the encoder is frozen (for IndicJEPA the **EMA target encoder** is evaluated). Each utterance is split into consecutive 5 s windows (the last zero-padded), each converted and normalised as in pretraining, and the 128 patch tokens per window are averaged over the 8 frequency rows → 16 frames per window. Following SUPERB, the probe learns a softmax-weighted sum over the outputs of all 12 blocks (taken before the final LayerNorm), mean-pools over time, and classifies:

- **LID** (12-way): benchmark default, one 128-unit ReLU hidden layer → linear.
- **SID** (229-way): 256-dim linear projection → linear (so the probe is linear in the layer-weighted features).

The probe is trained for 50k steps with the IndicSUPERB downstream recipes and is **identical across all methods** for a given task, so differences reflect the encoder, not the probe. Both tasks are scored by accuracy on the Kathbath valid set.

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

- MAE (Masked Autoencoders Are Scalable Vision Learners) — He et al., 2022 — [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
- Audio-MAE (Masked Autoencoders that Listen) — Huang et al., 2022 — [arXiv:2207.06405](https://arxiv.org/abs/2207.06405)
- M2D (Masked Modeling Duo) — Niizumi et al., 2022 — [arXiv:2210.14648](https://arxiv.org/abs/2210.14648)
- M2D: Towards a Universal Audio Pre-training Framework — Niizumi et al., 2024 — [arXiv:2404.06095](https://arxiv.org/abs/2404.06095)
- I-JEPA (the vision intuition inverted here) — Assran et al., 2023 — [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)
- A-JEPA (Joint-Embedding Predictive Architecture Can Listen) — Fei et al., 2023 — [arXiv:2311.15830](https://arxiv.org/abs/2311.15830)
- Audio-JEPA — Tuncay et al., 2025 — [arXiv:2507.02915](https://arxiv.org/abs/2507.02915)
- WavJEPA (raw-waveform JEPA) — Yuksel et al., 2025 — [arXiv:2509.23238](https://arxiv.org/abs/2509.23238)
- IndicSUPERB / Kathbath — Javed et al., AAAI 2023 — [arXiv:2208.11761](https://arxiv.org/abs/2208.11761)
- IndicVoices — Javed et al., Findings of ACL 2024 — [ACL Anthology](https://aclanthology.org/2024.findings-acl.639/)
- Are GANs Created Equal? A Large-Scale Study — Lucic et al., 2018 — [arXiv:1711.10337](https://arxiv.org/abs/1711.10337)
- A Metric Learning Reality Check — Musgrave et al., 2020 — [arXiv:2003.08505](https://arxiv.org/abs/2003.08505)
