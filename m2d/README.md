<table>
  <tr> <td align="center" width="50%">
      <img src="examples/image-key-visual-m2d.jpg" alt="key_visual_M2D" width="95%"><br>
      Masked Modeling Duo (M2D)
    </td>
    <td align="center" width="50%">
      <img src="examples/image-key-vis-m2d-clap.jpg" alt="key_visual_M2D-CLAP" width="73%"><br>
      M2D-CLAP
    </td> </tr>
</table>

# Masked Modeling Duo (M2D) & M2D-CLAP

This repository provides demo implementations of our paper "[Masked Modeling Duo: Towards a Universal Audio Pre-training Framework](https://ieeexplore.ieee.org/document/10502167)", "[M2D-CLAP: Exploring General-purpose Audio-Language Representations Beyond CLAP](https://ieeexplore.ieee.org/document/11168481)", "[Masked Modeling Duo: Learning Representations by Encouraging Both Networks to Model the Input](https://arxiv.org/abs/2210.14648)", and so on.

> 🌟 **Looking for the best general-purpose audio model?** [M2D-CLAP](clap/README.md) achieves state-of-the-art performance on audio tagging, zero-shot classification, and audio-language tasks — try it instantly in [Colab](http://colab.research.google.com/github/nttcslab/m2d/blob/master/examples/Colab_M2D-CLAP_ESC-50_ZS.ipynb).

## Quick Start

| Description | Notebook |
|:------------|:---------|
| Audio tagging example (M2D) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) examples/Colab_M2D_example_Tagging.ipynb](http://colab.research.google.com/github/nttcslab/m2d/blob/master/examples/Colab_M2D_example_Tagging.ipynb) |
| Zero-shot ESC-50 classification with M2D-CLAP | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) examples/Colab_M2D-CLAP_ESC-50_ZS.ipynb](http://colab.research.google.com/github/nttcslab/m2d/blob/master/examples/Colab_M2D-CLAP_ESC-50_ZS.ipynb) |
| Audio feature visualization example with M2D-CLAP | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) examples/Colab_M2D-CLAP_ESC-50_VizualizeEmbs.ipynb](http://colab.research.google.com/github/nttcslab/m2d/blob/master/examples/Colab_M2D-CLAP_ESC-50_VizualizeEmbs.ipynb) |

The example below uses **M2D-CLAP**, our recommended model. You can load it and encode audio in just a few lines:

```python
# 🔊 Load model
from examples.portable_m2d import PortableM2D  # portable_m2d: a simple one-file loader
model = PortableM2D('m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth')

# 🎵 Prepare input (three 10-s waveforms, range [-1., 1.])
import torch
batch_audio = 2 * torch.rand((3, 10 * 16000)) - 1.0

# 📐 Encode → frame-level features
frame_level = model(batch_audio)
print(frame_level.shape)  # torch.Size([3, 63, 3840])

# 📦 Aggregate → clip-level features
clip_level = torch.mean(frame_level, dim=1)
print(clip_level.shape)  # torch.Size([3, 3840])
```


## Pre-trained/Fine-tuned Weights

### AudioSet pre-trained weights

| Description | Recommendation | Weight | Fur-PT Ready | AS2M mAP |
|:------------|:---------------|:-------|:-------------:|:--------:|
| M2D-CLAP_2025 ⭐ | **Recommended.** Best for CLAP / audio tagging (AT) / sound event detection (SED). | [m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025](https://github.com/nttcslab/m2d/releases/download/v0.5.0/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025.zip) | ✅ | 0.490 |
| M2D-CLAP_2024, additionally fine-tuned on AS2M | 2nd Best for AT/SED. (Encoder only) | [m2d_clap_vit_base-80x1001p16x16-240128_AS-FT_enconly](https://github.com/nttcslab/m2d/releases/download/v0.3.0/m2d_clap_vit_base-80x1001p16x16-240128_AS-FT_enconly.zip) | N/A | 0.485 |
| M2D-AS fine-tuned on AS2M | 3rd best for AT/SED. (Encoder only) | [m2d_as_vit_base-80x1001p16x16-240213_AS-FT_enconly](https://github.com/nttcslab/m2d/releases/download/v0.3.0/m2d_as_vit_base-80x1001p16x16-240213_AS-FT_enconly.zip) | N/A | 0.485 |
| M2D/0.7 fine-tuned on AS2M | 4th best for AT/SED. (Encoder only) | [m2d_vit_base-80x1001p16x16-221006-mr7_as_46ab246d](https://github.com/nttcslab/m2d/releases/download/v0.3.0/m2d_vit_base-80x1001p16x16-221006-mr7_as_46ab246d.zip) | N/A | 0.479 |
| M2D/0.7 | General-purpose transfer learning and further pre-training. | [m2d_vit_base-80x608p16x16-221006-mr7](https://github.com/nttcslab/m2d/releases/download/v0.1.0/m2d_vit_base-80x608p16x16-221006-mr7.zip) | ✅ | - |
| M2D/0.7 | General-purpose transfer learning. (Encoder only) | [m2d_vit_base-80x608p16x16-221006-mr7_enconly](https://github.com/nttcslab/m2d/releases/download/v0.1.0/m2d_vit_base-80x608p16x16-221006-mr7_enconly.zip) | N/A | - |
| M2D/0.7 | General-purpose transfer learning. (Encoder only) | [m2d_vit_base-80x608p16x16-220930-mr7_enconly](https://github.com/nttcslab/m2d/releases/download/v0.1.0/m2d_vit_base-80x608p16x16-220930-mr7_enconly.zip) | N/A | - |
| M2D/0.7 (t.f. 40ms) | General-purpose transfer learning and further pre-training w/ finer time frame. | [m2d_vit_base-80x200p16x4-230529](https://github.com/nttcslab/m2d/releases/download/v0.1.0/m2d_vit_base-80x200p16x4-230529.zip) | ✅ | - |
| M2D-X/0.7 (η= 0.3) | The best ICBHI 2017 model in Section IV-E on the [TASLP paper](https://ieeexplore.ieee.org/document/10502167). | [m2d_x_icbhi](https://github.com/nttcslab/m2d/releases/download/v0.2.0/m2d_x_icbhi.tgz) | N/A | - |
| M2D/0.6 | General-purpose transfer learning and further pre-training. | [m2d_vit_base-80x608p16x16-221006-mr6](https://github.com/nttcslab/m2d/releases/download/v0.1.0/m2d_vit_base-80x608p16x16-221006-mr6.zip) | ✅ | - |
| M2D-CLAP_2024 (Older) | General-purpose transfer learning and further pre-training, especially when application data is closer to the AudioSet ontology. | [m2d_clap_vit_base-80x608p16x16-240128](https://github.com/nttcslab/m2d/releases/download/v0.1.0/m2d_clap_vit_base-80x608p16x16-240128.zip) | ✅ | - |
| M2D-AS | General-purpose transfer learning and further pre-training, especially when application data is closer to the AudioSet ontology. | [m2d_as_vit_base-80x608p16x16-240213](https://github.com/nttcslab/m2d/releases/download/v0.1.0/m2d_as_vit_base-80x608p16x16-240213.zip) | ✅ | - |
| [MSM-MAE](https://github.com/nttcslab/msm-mae)/0.75 | Predecessor to M2D; for reproducibility or comparison. | [msm_mae_vit_base-80x608p16x16-220924-mr75](https://github.com/nttcslab/m2d/releases/download/v0.1.0/msm_mae_vit_base-80x608p16x16-220924-mr75.zip) | ✅ | - |

| Description | Recommendation | Weight | Fur-PT Ready | AS2M mAP |
|:--------------|:----------------|:------------|:------:|:--------:|
| M2D-AS fine-tuned on AS2M@32kHz | Best for audio tagging (AT) / sound event detection (SED) at 32 kHz. | [m2d_as_vit_base-80x1001p16x16p32k-240413_AS-FT_enconly](https://github.com/nttcslab/m2d/releases/download/v0.3.0/m2d_as_vit_base-80x1001p16x16p32k-240413_AS-FT_enconly.zip) | N/A | 0.480 |
| M2D-AS@32kHz | General-purpose transfer learning at 32 kHz. (Encoder only) | [m2d_as_vit_base-80x608p16x16p32k-240413_enconly](https://github.com/nttcslab/m2d/releases/download/v0.1.0/m2d_as_vit_base-80x608p16x16p32k-240413_enconly.zip) | N/A | - |

### LibriSpeech pre-trained weights

| Description | Recommendation | Weight | Fur-PT Ready | AS2M mAP |
|:--------------|:----------------|:------------|:------:|:--------:|
| M2D-S/0.6 6-s input | Speech transfer learning and further pre-training. | [m2d_s_vit_base-80x608p80x2-230220](https://github.com/nttcslab/m2d/releases/download/v0.2.0/m2d_s_vit_base-80x608p80x2-230220.zip) | ✅ | - |
| M2D-S/0.6 5-s input | Speech transfer learning and further pre-training. | [m2d_s_vit_base-80x512p80x2-230301](https://github.com/nttcslab/m2d/releases/download/v0.2.0/m2d_s_vit_base-80x512p80x2-230301.zip) | ✅ | - |
| M2D-S/0.6 4-s input | Speech transfer learning and further pre-training. | [m2d_s_vit_base-80x400p80x2-230201](https://github.com/nttcslab/m2d/releases/download/v0.2.0/m2d_s_vit_base-80x400p80x2-230201.zip) | ✅ | - |


## Application Resources

👉 [**Application Guide (alpha) is available.**](app/Guide_app.md) -- Our guidelines may provide useful information on how to plan further pre-train your models.
<figure>
  <a href="app/Guide_app.md"><img src="examples/image-AppGuideChart.png" alt="A guide chart", width="30%"></a>
</figure>

- [👉 **Resources for M2D-CLAP (General-purpose Audio-Language Representation)**](clap/README.md).
- [👉 **Resources for M2D-X medical applications (ICBHI2017/SPRSound), further pre-training examples**](app/icbhi_sprs/README_ICBHI_SPRS.md).
- [👉 **Resources for M2D medical application (CirCor DigiScope heart sound)**](app/circor/README.md).
- [👉 **Resources for M2D-AS (M2D-X specialized in AudioSet)**](audioset/README.md).
- [👉 **Resources for M2D-S (M2D-X specialized in Speech)**](speech/README.md).
- [👉 **Resources for M2D on respiratory sound analysis (OPERA benchmark)**](https://github.com/nttcslab/eval-audio-repr/blob/main/plugin/OPERA/REAEDME_OPERA.md) — Pre-training and evaluation resources for respiratory sounds using M2D, hosted in the EVAR repository (see our [Interspeech 2025 paper](https://www.isca-archive.org/interspeech_2025/niizumi25_interspeech.html)).
- [👉 **Resources for M2D on music understanding (MARBLE benchmark)**](https://github.com/nttcslab/eval-audio-repr/blob/main/plugin/MARBLE/REAEDME_MARBLE.md) — Integration of M2D with the MARBLE music benchmark, hosted in the EVAR repository (covered in the [M2D-CLAP 2025 paper](https://ieeexplore.ieee.org/document/11168481)).
- [👉 **MSM-MAE pre-training (predecessor to M2D)**](examples/MSM-MAE_Pretraining.md) — Pre-training guide for [MSM-MAE](https://github.com/nttcslab/msm-mae), the model that M2D builds upon. Provided for reproducibility and comparison.

A schematic illustration of M2D-X further pre-training:
<figure>
  <img src="examples/image-M2D-further-PT.svg" alt="A schematic illustration of M2D-X further pre-training", width="40%">
</figure>

## 1. Setup

The repository is based on the codes from [facebookresearch/mae](https://github.com/facebookresearch/mae), and we patch our changes on these files.

1. Download external source files and apply a patch.

    ```sh
    git clone https://github.com/nttcslab/m2d.git
    cd m2d
    curl -o util/lars.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/lars.py
    curl -o util/lr_decay.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/lr_decay.py
    curl -o util/lr_sched.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/lr_sched.py
    curl -o util/misc.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/misc.py
    curl -o util/analyze_repr.py https://raw.githubusercontent.com/daisukelab/general-learning/master/SSL/analyze_repr.py
    curl -o m2d/pos_embed.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py
    curl -o train_audio.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_pretrain.py
    curl -o speech/train_speech.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_pretrain.py
    curl -o audioset/train_as.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_pretrain.py
    curl -o clap/clap_only.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_pretrain.py
    curl -o clap/train_clap.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_pretrain.py
    curl -o mae_train_audio.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_pretrain.py
    curl -o m2d/engine_pretrain_m2d.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/engine_pretrain.py
    curl -o m2d/models_mae.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/models_mae.py
    curl -o m2d/timm_layers_pos_embed.py https://raw.githubusercontent.com/huggingface/pytorch-image-models/e9373b1b925b2546706d78d25294de596bad4bfe/timm/layers/pos_embed.py
    patch -p1 < patch_m2d.diff
    ```

2. Install external modules listed on [requirements.txt](requirements.txt).

    ```sh
    pip install -r requirements.txt
    ```

## 2. Evaluating M2D

We use the [EVAR](https://github.com/nttcslab/eval-audio-repr) for our evaluation.

### 2-1. Setup EVAR

[EVAR](https://github.com/nttcslab/eval-audio-repr) is an evaluation package for audio representations used by our research papers such as [BYOL-A](https://ieeexplore.ieee.org/document/9944865).

The following steps set up EVAR.

1. In the folder of your copy of the M2D repository, clone the EVAR repository and prepare basic items.

    ```sh
    git clone https://github.com/nttcslab/eval-audio-repr.git evar
    cd evar
    curl https://raw.githubusercontent.com/daisukelab/general-learning/master/MLP/torch_mlp_clf2.py -o evar/utils/torch_mlp_clf2.py
    curl https://raw.githubusercontent.com/daisukelab/sound-clf-pytorch/master/for_evar/sampler.py -o evar/sampler.py
    curl https://raw.githubusercontent.com/daisukelab/sound-clf-pytorch/master/for_evar/cnn14_decoupled.py -o evar/cnn14_decoupled.py
    cd ..
    ```

2. Set up downstream task datasets according to [Preparing-datasets.md](https://github.com/nttcslab/eval-audio-repr/blob/main/Preparing-datasets.md). The following is an example for setting up CREMA-D dataset.

    ```sh
    cd evar
    python evar/utils/download_cremad.py downloads/cremad
    python prepare_wav.py downloads/cremad work/16k/cremad 16000
    cd ..
    ```

### 2-2. Linear Evaluation

Once you set up EVAR, you can evaluate your models as follows.

- For evaluating a model with an absolute path `/your/path/to/model.pth`.

    ```sh
    cd evar
    python lineareval.py config/m2d.yaml cremad weight_file=/your/path/to/model.pth
    ```

- If you want to save GPU memory, set a smaller batch size as follows. This example sets it as 16.

    ```sh
    cd evar
    python lineareval.py config/m2d.yaml cremad batch_size=16,weight_file=/your/path/to/model.pth
    ```

We used the `all_eval.sh` script to evaluate on all downstream tasks.

### 2-3. Fine-tuning

We have fine-tuned our models using the scripts in the `util` folder.

The following examples fine-tune on each downstream task three times with seed 42. Replace `/your/path/to/m2d_vit_base-80x608p16x16-221006-mr7` with your actual model path.

```sh
cd evar
bash <path/to/m2d>/util/ft-as2m.sh /your/path/to/m2d_vit_base-80x608p16x16-221006-mr7 3 42 300  # AudioSet 2M
bash <path/to/m2d>/util/ft-as0k.sh /your/path/to/m2d_vit_base-80x608p16x16-221006-mr7 3 42 300  # AudioSet 20K
bash <path/to/m2d>/util/ft-esc50.sh /your/path/to/m2d_vit_base-80x608p16x16-221006-mr7 3 42 300  # ESC-50
bash <path/to/m2d>/util/ft-spc.sh /your/path/to/m2d_vit_base-80x608p16x16-221006-mr7 3 42 300  # Speech Commands
bash <path/to/m2d>/util/ft-vc1.sh /your/path/to/m2d_vit_base-80x608p16x16-221006-mr7 3 42 300  # VoxCeleb1
```

#### NOTE: Please set your data path in `util/ft-as2m.sh`

The `ft-as2m.sh` requires the path to your log-mel spectrogram AudioSet samples in .npy. Update it with your data path before running.

## 3. Pre-training From Scratch

### 3-1. Prepare pre-training data samples

The pre-trainer (e.g., `train_audio.py` for audio) loads data from the `data` folder by default (`--data_path`), using a list of samples in a CSV `data/files_audioset.csv` by default (`--csv_main`).
Follow the steps in [data/README.md](data/README.md).

The following is an example using the [FSD50K](https://arxiv.org/abs/2010.00475) dataset.

1. Preprocess .wav files into log-mel spectrogram .npy files. The following converts from a source folder `/your/local/fsd50k/FSD50K.dev_audio` to a new folder `data/fsd50k_lms`.

    ```sh
    python wav_to_lms.py /your/local/fsd50k/FSD50K.dev_audio data/fsd50k_lms
    ```

2. Create a CSV file that will be used as a list of pre-training samples, containing a single column `file_name`. The following example creates `files_f_s_d_5_0_k.csv`.

    ```sh
    echo file_name > data/files_f_s_d_5_0_k.csv
    (cd data && find fsd50k_lms/FSD50K.dev_audio -name "*.npy") >> data/files_f_s_d_5_0_k.csv
    ```

Example of created folder structure:

    data/
        files_f_s_d_5_0_k.csv
        fsd50k_lms/
            FSD50K.dev_audio/
                2931.npy
                408195.npy
                    :

### 3-2. Start pre-training

Once your data is ready, start pre-training as follows.

```sh
python train_audio.py --csv_main data/files_f_s_d_5_0_k.csv
```

### 3-3. Evaluation during and after the training

The training loop automatically evaluates the pre-trained model.

- During pre-training, `train_audio.py` runs a script called `quick_eval.sh` as a sub-process. You can edit `quick_eval.sh` for your purposes.
- When the pre-training is finished, the final evaluation script `all_eval.sh` is executed.

### 3-4. Complete pre-training command lines

The command lines for pre-training full-performance models follow:

```sh
# M2D
OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 -m train_audio --input_size 80x608 --patch_size 16x16 --epochs 300 --batch_size 512 --accum_iter 1 --save_freq 50 --seed 3 --model m2d_vit_base --csv_main data/files_audioset.csv --data_path /path/to/your/data --loss_off 0.
# M2D-AS
OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 -m audioset.train_as --input_size 80x608 --patch_size 16x16 --epochs 300 --batch_size 512 --accum_iter 1 --save_freq 50 --seed 3 --data_path /path/to/your/data --loss_off 1.
```

> **Note:** Replace `/path/to/your/data` with the path to your LMS data directory. Placing data on fast storage (SSD recommended) significantly speeds up training. If `--data_path` is omitted, the `data/` directory at the repository root is used.

Example logs are available: [example_logs.zip](https://github.com/nttcslab/m2d/releases/download/v0.1.0/example_logs.zip).

We explain the details in the [Guide_app.md](app/Guide_app.md).

For other model variants, see also:
- [M2D-CLAP pre-training](clap/README.md) — multi-stage training for audio-language representation
- [MSM-MAE pre-training](examples/MSM-MAE_Pretraining.md) — predecessor to M2D

## 4. Other Pre-trained/fine-tuned Weights

Please find all pre-trained/fine-tuned weights published on the [releases](https://github.com/nttcslab/m2d/releases).

## 5. License

See [LICENSE.pdf](LICENSE.pdf) for details.

## Citations

If you find our M2D or M2D-CLAP useful in your research, please consider citing our papers.

```BibTeX
@article{niizumi2025m2d-clap,
    author  = {Niizumi, Daisuke and Takeuchi, Daiki and Yasuda, Masahiro and Nguyen, Binh Thien and Ohishi, Yasunori and Harada, Noboru},
    journal = {IEEE Access}, 
    title   = {{M2D-CLAP: Exploring General-purpose Audio-Language Representations Beyond CLAP}}, 
    year    = {2025},
    volume  = {13},
    pages   = {163313-163330},
    doi={10.1109/ACCESS.2025.3611348}}

@article{niizumi2024m2dx,
    title   = {{Masked Modeling Duo: Towards a Universal Audio Pre-training Framework}},
    author  = {Daisuke Niizumi and Daiki Takeuchi and Yasunori Ohishi and Noboru Harada and Kunio Kashino},
    journal = {IEEE/ACM Trans. Audio, Speech, Language Process.},
    year    = {2024},
    volume  = {32},
    pages   = {2391-2406},
    url     = {https://ieeexplore.ieee.org/document/10502167},
    doi     = {10.1109/TASLP.2024.3389636}}

@inproceedings{niizumi2024m2d-clap,
    title   = {{M2D-CLAP: Masked Modeling Duo Meets CLAP for Learning General-purpose Audio-Language Representation}},
    author  = {Daisuke Niizumi and Daiki Takeuchi and Yasunori Ohishi and Noboru Harada and Masahiro Yasuda and Shunsuke Tsubaki and Keisuke Imoto},
    booktitle={Interspeech},
    year    = {2024},
    pages   = {57--61},
    doi     = {10.21437/Interspeech.2024-29}}

@inproceedings{niizumi2023m2d,
    title   = {{Masked Modeling Duo: Learning Representations by Encouraging Both Networks to Model the Input}},
    author  = {Daisuke Niizumi and Daiki Takeuchi and Yasunori Ohishi and Noboru Harada and Kunio Kashino},
    booktitle={ICASSP}, 
    year    = {2023},
    url     = {https://ieeexplore.ieee.org/document/10097236},
    doi     = {10.1109/ICASSP49357.2023.10097236}}

@inproceedings{niizumi2023m2d4speech,
    title   = {{Masked Modeling Duo for Speech: Specializing General-Purpose Audio Representation to Speech using Denoising Distillation}},
    author  = {Daisuke Niizumi and Daiki Takeuchi and Yasunori Ohishi and Noboru Harada and Kunio Kashino},
    year    = {2023},
    booktitle={Interspeech},
    pages   = {1294--1298},
    doi     = {10.21437/Interspeech.2023-221}}

@inproceedings{niizumi2024embc,
    title   = {{Exploring Pre-trained General-purpose Audio Representations for Heart Murmur Detection}},
    author  = {Niizumi, Daisuke and Takeuchi, Daiki and Ohishi, Yasunori and Harada, Noboru and Kashino, Kunio},
    booktitle={EMBC},
    year    = {2024},
    pages   = {1-4},
    doi     = {10.1109/EMBC53108.2024.10782479}}
```

## Acknowledgements

- Our code is based on the [MAE PyTorch/GPU re-implementation](https://github.com/facebookresearch/mae) of the paper [Masked Autoencoders Are Scalable Vision Learners](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html).
- We use [nnAudio](https://ieeexplore.ieee.org/document/9174990) ([KinWaiCheuk/nnAudio](https://github.com/KinWaiCheuk/nnAudio)) for converting raw audio into log-mel spectrogram.

We appreciate these publicly available implementations and all the modules our experiments heavily depend on!

## References

- MAE: *[Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick "Masked Autoencoders Are Scalable Vision Learners," Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 16000-16009](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html).*
- FSD50K: *[Eduardo Fonseca and Xavier Favory and Jordi Pons and Frederic Font and Xavier Serra, “FSD50K: an Open Dataset of Human-Labeled Sound Events,” in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 30, pp. 829-852, 2022](https://ieeexplore.ieee.org/document/9645159).*
- MSM-MAE: *[Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Noboru Harada, and Kunio Kashino "Masked Spectrogram Modeling using Masked Autoencoders for Learning General-purpose Audio Representation," HEAR: Holistic Evaluation of Audio Representations (NeurIPS 2021 Competition), PMLR 166:1-24, 2022](https://proceedings.mlr.press/v166/niizumi22a.html).*
