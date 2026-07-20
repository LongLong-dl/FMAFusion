import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    from dataset import Nutrition_RGBD_CSV
except ImportError:
    from dataloader.dataset import Nutrition_RGBD_CSV


def seed_worker(worker_id):
    """Deterministic DataLoader worker init for reproducibility (Dataset and Setup)."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_DataLoader(args):
    """Build train/test DataLoaders with ImageNet normalization (Dataset and Setup)."""
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Data augmentation: Resize(274×324) → CenterCrop(224×224)
    transform = transforms.Compose([
        transforms.Resize((args.size + 50, args.size + 100)),
        transforms.CenterCrop((args.size, args.size)),
        transforms.ToTensor(),
        norm,
    ])

    imagery_root = os.path.join(args.data_root, 'imagery')
    train_csv = os.path.join(imagery_root, 'train.csv')
    test_csv = os.path.join(imagery_root, 'test.csv')

    trainset = Nutrition_RGBD_CSV(imagery_root, train_csv, transform=transform)
    testset = Nutrition_RGBD_CSV(imagery_root, test_csv, transform=transform)

    train_loader = DataLoader(trainset,
                              batch_size=args.b,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              worker_init_fn=seed_worker,
                              drop_last=True)
    test_loader = DataLoader(testset,
                             batch_size=args.b,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             worker_init_fn=seed_worker)

    return train_loader, test_loader


if __name__ == '__main__':
    class Args:
        data_root = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'nutrition5k_dataset')
        size = 224
        b = 4
        num_workers = 0

    train_loader, test_loader = get_DataLoader(Args)
    print(f"Train batches: {len(train_loader)}")
    print(f"Test  batches: {len(test_loader)}")

    batch = next(iter(train_loader))
    img_rgb, dish_id, cal, mass, fat, carb, protein, img_depth = batch
    print(f"RGB: {img_rgb.shape}  Depth: {img_depth.shape}")

    print("\n=== All DataLoader tests passed ===")
