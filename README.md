# FMAFusion

Frequency-guided Multi-scale Alignment Fusion for food nutritional estimation.

## Project Structure

```
FMAFusion/
├── dataloader/
│   ├── dataloader.py    DataLoader with deterministic preprocessing
│   └── dataset.py       Nutrition5K RGB-D dataset reader
├── model/
│   ├── __init__.py
│   ├── dual_swin.py       Dual Swin-Base encoder → FMAFusion → Head
│   ├── fma_fusion.py      FMAFusion + CrossLevelHighFreqGate + FMAFusionHead
│   ├── loss.py            NormalizedMAELoss
│   └── swin_backbone.py   Swin Transformer backbone (from microsoft)
├── train.py                Training script
├── evaluate.py             Evaluation script (MAE / PMAE)
├── logs/                   Training logs & checkpoints
├── weights/                Pretrained Swin weights
├── dataset/
│   └── nutrition5k_dataset/
│       ├── dish_nutrition_values.csv       # ground truth labels
│       ├── dish_ingredients.csv
│       ├── ingredients_metadata.csv
│       ├── imagery/
│       │   ├── realsense_overhead/
│       │   │   └── <dish_id>/
│       │   │       ├── rgb.png
│       │   │       ├── depth_color.png
│       │   │       └── depth_raw.png
│       │   ├── train.csv                   # train split
│       │   └── test.csv                    # test split
├── README.md
└── requirements.txt
```

## Setup

```bash
# dataset
mkdir dataset
# download Nutrition5K to dataset/nutrition5k_dataset/
# then split
python process_data/split_dataset.py

# pretrained weights
mkdir weights
# download swin_base_patch4_window7_224_22k.pth to weights/
```
## Usage

```bash
# train
python train.py --image_size 224 --epochs 150 --b 32 --lr 0.00004 \
    --swin_weights_rgb ./weights/swin_base_patch4_window7_224_1k.pth \
    --exp_tag FMAFusion

# evaluate
python evaluate.py --weights logs/<exp>/best_model.pth --image_size 224
```
## Requirements

torch≥2.0, torchvision, timm, opencv-python, tqdm, numpy, pillow
