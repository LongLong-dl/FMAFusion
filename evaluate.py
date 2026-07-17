import argparse
import json
import os

import numpy as np
import torch

from dataloader.dataloader import get_DataLoader
from model.dual_swin import DualSwin

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_NAMES = ['calories', 'mass', 'fat', 'carb', 'protein']


class _Args:
    def __init__(self, size, batch_size, num_workers):
        self.size = size
        self.b = batch_size
        self.num_workers = num_workers
        self.data_root = os.path.join(os.path.dirname(__file__),
                                      'dataset', 'nutrition5k_dataset')


def evaluate(weights_path, image_size=224, batch_size=32, num_workers=8):
    print(f'Loading model: image_size={image_size}')
    print(f'Weights: {weights_path}')

    model = DualSwin(
        image_size=image_size,
        swin_weights_rgb=None,
        dropout=0.1,
    ).to(device)

    state = torch.load(weights_path, map_location=device)
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.eval()
    print(f'Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}')

    args = _Args(size=image_size, batch_size=batch_size, num_workers=num_workers)
    _, test_loader = get_DataLoader(args)

    per_target_abs_sum = [0.0] * 5
    per_target_label_sum = [0.0] * 5
    count = 0

    with torch.no_grad():
        for batch in test_loader:
            img_rgb, _, cal, mass, fat, carb, protein, img_rgbd = batch
            rgb = img_rgb.to(device)
            depth = img_rgbd.to(device)
            labels = torch.stack([cal, mass, fat, carb, protein], dim=1).to(device)

            pred, _ = model(rgb, depth)
            err = (pred - labels).abs()

            for k in range(5):
                per_target_abs_sum[k] += err[:, k].sum().item()
                per_target_label_sum[k] += labels[:, k].abs().sum().item()
            count += labels.size(0)

    print(f'\n{"=" * 60}')
    print(f'  Evaluation Results — Test Set ({count} samples)')
    print(f'{"=" * 60}')
    print(f'  {"Target":<12} {"MAE":>8} {"PMAE(%)":>10}')
    print(f'  {"-" * 32}')

    total_mae = 0.0
    results = {}
    for k, name in enumerate(TARGET_NAMES):
        mae = per_target_abs_sum[k] / count
        mean_abs_label = per_target_label_sum[k] / count
        pmae = (mae / mean_abs_label) * 100 if mean_abs_label > 0 else 0.0
        total_mae += mae
        results[name] = {'MAE': round(mae, 2), 'PMAE': round(pmae, 2)}
        print(f'  {name:<12} {mae:>8.2f} {pmae:>9.2f}%')

    avg_mae = total_mae / 5
    avg_pmae = sum(r['PMAE'] for r in results.values()) / 5
    print(f'  {"-" * 32}')
    print(f'  {"AVERAGE":<12} {avg_mae:>8.2f} {avg_pmae:>9.2f}%')
    print(f'{"=" * 60}')

    return results, avg_mae, avg_pmae


def main():
    p = argparse.ArgumentParser(description='FMAFusion Evaluation')
    p.add_argument('--weights', type=str, required=True)
    p.add_argument('--image_size', type=int, default=224, choices=[224, 384])
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=8)

    cli = p.parse_args()

    results, avg_mae, avg_pmae = evaluate(
        weights_path=cli.weights,
        image_size=cli.image_size,
        batch_size=cli.batch_size,
        num_workers=cli.num_workers,
    )

    out_dir = os.path.dirname(cli.weights)
    out_path = os.path.join(out_dir, 'eval_result.json')
    with open(out_path, 'w') as f:
        json.dump({
            'weights': cli.weights,
            'per_target': results,
            'avg_mae': round(avg_mae, 2),
            'avg_pmae': round(avg_pmae, 2),
        }, f, indent=2)
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
