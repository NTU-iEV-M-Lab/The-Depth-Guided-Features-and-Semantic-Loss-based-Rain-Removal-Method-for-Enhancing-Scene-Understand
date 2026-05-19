[![Paper](https://img.shields.io/badge/Paper-ScienceDirect-red)](https://www.sciencedirect.com/science/article/pii/S0957417425042769)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.eswa.2025.130661-blue)](https://doi.org/10.1016/j.eswa.2025.130661)
[![TF](https://img.shields.io/badge/TensorFlow-2.13-orange)](#installation)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

# Improved Visibility for Monocular Camera-based Perception under Rain with Fog
### An Efficient Vision Transformer Approach with Depth-Guided Spatial Feature Transform (DG-SFT) and Semantic Loss

Official Keras/TensorFlow implementation of the ESWA paper:
**“Improved Visibility for Monocular Camera-based Perception in Autonomous Driving Systems under Rain with Fog: An Efficient Vision Transformer Approach.”**

## Abstract
Adverse weather degrades image quality and undermines the reliability of monocular camera-based perception in autonomous driving. Most existing deraining methods primarily remove rain streaks while overlooking fog-like effects, which further obscure distant scene details and harm downstream perception. This work presents an efficient Vision Transformer (ViT)-based restoration framework that addresses both rain streaks and fog without compromising perception accuracy. The proposed model introduces a Depth-Guided Spatial Feature Transform (DG-SFT) block that leverages depth information predicted by a lightweight CNN decoder and is designed based on a mathematical rain model to mitigate distance-dependent haze. In addition, we propose a semantic loss that constrains the discrepancy between segmentation outputs of the original and restored images. Experiments on RainCityscapes and real-world rainy images show improved PSNR/SSIM over prior ViT- and CNN-based baselines, while achieving 7.86 ms inference latency, supporting latency-critical deployment.

## Model Architecture

<p align="center">
  <img src="image/README/Architecture.png" width="95%">
</p>
<p align="center"><em>Overview of the proposed network.</em></p>

The model has three parts:

- **PoolFormer encoder** (`model/poolformer.py`) — a lightweight ViT backbone whose self-attention is replaced by parameter-free spatial pooling; it produces a four-stage feature pyramid.
- **Depth decoder head** (`model/depth_head.py`) — a lightweight CNN decoder that predicts an inverse-depth map.
- **De-weather decoder head** (`model/weather_head.py`) — restores the clean image; its **DG-SFT** block uses the predicted depth to apply a learned spatial scale and shift, correcting distance-dependent rain/haze.

Training optimizes a joint objective (paper Eq. 9–13):

```
total_loss = depth_loss + (l1_loss + perceptual_loss) + semantic_loss
```

all terms equally weighted. The **semantic loss** is the cross-entropy between SegFormer predictions on the ground-truth and restored images.

## Repository Structure

```
.
├── train.py            # Train the model on RainCityscapes
├── evaluate.py         # Evaluate PSNR/SSIM/depth-L1 (+ optional image export)
├── tf_FLOPs.py         # Report the model FLOPs
├── model/              # PoolFormer encoder + depth / de-weather heads
├── utils/              # Dataset loader, losses, metrics, plotting, callbacks
├── requirements.txt
└── image/README/       # Figures used in this README
```

## Installation

Requires Python 3.9–3.11 and TensorFlow 2.13.

```bash
git clone https://github.com/<your-account>/DG-SFT.git
cd DG-SFT
pip install -r requirements.txt
```

> **Note.** Training downloads two pretrained models automatically on first run: the VGG16 ImageNet weights (perceptual loss) and the SegFormer checkpoint `nvidia/segformer-b0-finetuned-cityscapes-512-1024` (semantic loss). An internet connection is required for the first run.

## Dataset Preparation

The model is trained on **RainCityscapes**, derived from the [Cityscapes](https://www.cityscapes-dataset.com/) dataset. Arrange the data under `data/` as follows:

```
data/
├── rain_cityscapes/
│   └── leftImg8bit_rain/{train,val}/<city>/*.png    # synthetic rain+fog images
├── cityscapes/
│   ├── leftImg8bit/{train,val}/<city>/*.png         # clean reference images
│   └── gtFine/{train,val}/<city>/*_gtFine_labelTrainIds.png
└── disparity/{train,val}/<city>/*_disparity.png     # normalized inverse-depth maps
```

- The clean images and `gtFine` labels come from Cityscapes.
- `data/disparity/` holds normalized inverse-depth maps derived from the Cityscapes disparity maps (scaled to `[0, 1]`); these serve as the depth ground truth.
- Dataset paths can be adjusted via the arguments of `utils/dataset.RainCityscapes`.

## Training

Train with the default settings:

```bash
python train.py
```

All hyperparameters can be overridden from the command line:

```bash
python train.py --epochs 150 --batch-size 4 --lr 3e-4 \
    --weight-decay 1e-4 --warmup 0 --seed 2 \
    --checkpoint-dir checkpoints --gpu-memory-limit 20480
```

| Argument | Default | Description |
|---|---|---|
| `--epochs` | `150` | Number of training epochs |
| `--batch-size` | `4` | Mini-batch size |
| `--lr` | `3e-4` | Peak learning rate for the cosine schedule |
| `--weight-decay` | `1e-4` | Weight decay for the Lion optimizer |
| `--warmup` | `0` | Warm-up fraction of total training steps |
| `--seed` | `2` | Random seed |
| `--checkpoint-dir` | `checkpoints` | Root directory for saved checkpoints |
| `--gpu-memory-limit` | `20480` | Per-GPU memory cap (MB) |

Checkpoints are written to `<checkpoint-dir>/seed<seed>/`:

- `model.keras` — the full training model (use this with `evaluate.py`).
- `weather.keras` — the exported inference-only sub-model.

## Evaluation

Compute PSNR / SSIM / depth-L1 on the RainCityscapes validation split:

```bash
python evaluate.py --checkpoint checkpoints/seed2/model.keras
```

Add `--save-images` to also export the predicted (derained) images:

```bash
python evaluate.py --checkpoint checkpoints/seed2/model.keras \
    --save-images --output-dir results
```

> **Pretrained weights are not distributed.** Train the model yourself with
> `train.py` and pass the resulting `model.keras` checkpoint to `evaluate.py`.

## Results

The proposed method improves PSNR/SSIM over prior ViT- and CNN-based baselines on RainCityscapes while keeping inference latency low (7.86 ms). It also preserves downstream perception (semantic segmentation) on restored images. See the paper for the complete quantitative tables.

**Comparison on RainCityscapes**
<p align="center">
  <img src="image/README/1732081399839.png" width="95%">
</p>
<p align="center">
  <img src="image/README/2026-02-23%20144923.png" width="80%">
</p>

**Analysis on the downstream task**
<p align="center">
  <img src="image/README/1732081441406.png" width="95%">
</p>
<p align="center">
  <img src="image/README/1231256876867.png" width="95%">
</p>

## Citation

If you find this work useful, please cite the paper:

```bibtex
@article{huang2025improvedvisibility,
  title   = {Improved Visibility for Monocular Camera-based Perception in Autonomous Driving Systems under Rain with Fog: An Efficient Vision Transformer Approach},
  author  = {Huang, Yao-Jiun and Li, Kang},
  journal = {Expert Systems with Applications},
  year    = {2025},
  doi     = {10.1016/j.eswa.2025.130661},
  publisher = {Elsevier}
}
```

## License

Released under the [Apache License 2.0](LICENSE).

## Acknowledgements

- The [Cityscapes](https://www.cityscapes-dataset.com/) dataset and its RainCityscapes extension.
- [SegFormer](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-512-1024) (via Hugging Face Transformers) for the semantic loss.
- VGG16 ImageNet features for the perceptual loss.
