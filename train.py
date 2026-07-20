import argparse
import json
import os
import random
import numpy as np
import torch
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm

from dataloader.dataloader import get_DataLoader
from model.dual_swin import DualSwin
from model.loss import NormalizedMAELoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Args:
    def __init__(self, **kw):
        self.size = kw.get('size', 224)
        self.b = kw.get('b', 16)
        self.num_workers = kw.get('num_workers', 8)
        self.data_root = os.path.join(os.path.dirname(__file__),
                                      'dataset', 'nutrition5k_dataset')


def evaluate_epoch(model, loader, criterion):
    model.eval()
    total_loss, total_mae = 0.0, 0.0
    with torch.no_grad():
        for batch in loader:
            img_rgb, _, cal, mass, fat, carb, protein, img_rgbd = batch
            rgb, depth = img_rgb.to(device), img_rgbd.to(device)
            labels = torch.stack([cal, mass, fat, carb, protein], dim=1).to(device)
            op, _aux = model(rgb, depth)
            loss = criterion(op, labels)
            total_loss += loss.item()
            total_mae += (op - labels).abs().mean().item()
    model.train()
    n = len(loader)
    return total_loss / n, total_mae / n


def per_target_mae(model, loader):
    model.eval()
    sums = [0.0] * 5
    counts = 0
    with torch.no_grad():
        for batch in loader:
            img_rgb, _, cal, mass, fat, carb, protein, img_rgbd = batch
            rgb, depth = img_rgb.to(device), img_rgbd.to(device)
            labels = torch.stack([cal, mass, fat, carb, protein], dim=1).to(device)
            op, _ = model(rgb, depth)
            err = (op - labels).abs()
            for k in range(5):
                sums[k] += err[:, k].sum().item()
            counts += labels.size(0)
    model.train()
    targets = ['calories', 'mass', 'fat', 'carb', 'protein']
    result = {}
    for i, t in enumerate(targets):
        result[t] = sums[i] / counts
    return result


def main(exp_tag="FMAFusion",
         image_size=224,
         swin_weights_rgb=None,
         swin_weights_depth=None,
         learning_rate=1e-4,
         weight_decay=1e-5,
         beta1=0.9,
         beta2=0.999,
         epochs=200,
         batch_size=16,
         num_workers=8,
         dropout=0.1,
         early_stop_patience=30,
         ):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = exp_tag.replace(" ", "_").replace("/", "_")
    log_dir = os.path.join(os.path.dirname(__file__), "logs", f"{tag}_{ts}")
    os.makedirs(log_dir, exist_ok=True)

    print(f'=== FMAFusion: {exp_tag} ===')
    print(f'image_size={image_size}  batch={batch_size}')
    print(f'lr={learning_rate}  wd={weight_decay}  '
          f'beta1={beta1}  beta2={beta2}  epochs={epochs}  '
          f'dropout={dropout}')
    print(f'swin_weights_rgb={swin_weights_rgb}')
    print(f'swin_weights_depth={swin_weights_depth}')

    model = DualSwin(
        image_size=image_size,
        swin_weights_rgb=swin_weights_rgb,
        swin_weights_depth=swin_weights_depth,
        dropout=dropout,
    ).to(device)

    n_params = count_params(model)
    print(f'Trainable Parameters: {n_params:,}  |  Log: {log_dir}')

    args = Args(size=image_size, b=batch_size, num_workers=num_workers)
    train_loader, test_loader = get_DataLoader(args)

    criterion = NormalizedMAELoss().to(device)
    param_groups = [
        {'params': model.encoder_rgb.parameters(), 'lr': learning_rate,
         'betas': (beta1, beta2), 'weight_decay': weight_decay},
        {'params': model.encoder_depth.parameters(), 'lr': learning_rate,
         'betas': (beta1, beta2), 'weight_decay': weight_decay},
        {'params': list(model.fma_fusion.parameters()) + list(model.head.parameters()),
         'lr': learning_rate, 'betas': (beta1, beta2), 'weight_decay': weight_decay},
    ]
    optimizer = optim.AdamW(param_groups)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=0.000025,
    )

    best_mae = float('inf')
    best_epoch = 0
    no_improve_count = 0

    cfg = {
        "experiment": exp_tag,
        "model": {
            "name": "DualSwin_FMAFusion",
            "image_size": image_size,
            "total_params": n_params,
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "device": str(device),
        },
        "optimizer": {"type": "AdamW", "lr": learning_rate,
                      "weight_decay": weight_decay,
                      "beta1": beta1, "beta2": beta2},
        "scheduler": "CosineAnnealingLR",
        "loss": "NormalizedMAELoss",
        "regularization": {
            "dropout": dropout,
            "early_stop_patience": early_stop_patience,
        },
        "swin_weights": {
            "rgb": swin_weights_rgb,
            "depth": swin_weights_depth,
        },
        "start_time": ts,
    }
    with open(os.path.join(log_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    log_path = os.path.join(log_dir, "train_log.jsonl")

    for epoch in range(epochs):
        model.train()
        epoch_loss, epoch_mae = 0.0, 0.0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            img_rgb, _, cal, mass, fat, carb, protein, img_rgbd = batch
            rgb, depth = img_rgb.to(device), img_rgbd.to(device)
            labels = torch.stack([cal, mass, fat, carb, protein], dim=1).to(device)

            optimizer.zero_grad()
            op, _aux = model(rgb, depth)
            loss = criterion(op, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mae += (op.detach() - labels).abs().mean().item()

        scheduler.step()
        train_loss = epoch_loss / len(train_loader)
        train_mae = epoch_mae / len(train_loader)
        lr_now = scheduler.get_last_lr()[0]

        test_loss, test_mae = evaluate_epoch(model, test_loader, criterion)

        print(f'Epoch {epoch+1}/{epochs}  '
              f'Train L={train_loss:.4f}  Train MAE={train_mae:.2f}  '
              f'Test MAE={test_mae:.2f}  '
              f'LR={lr_now:.6f}')

        rec = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "train_mae": round(train_mae, 6),
            "test_loss": round(test_loss, 6),
            "test_mae": round(test_mae, 6),
            "lr": lr_now,
            "timestamp": datetime.now().isoformat(),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

        if test_mae < best_mae:
            best_mae = test_mae
            best_epoch = epoch + 1
            no_improve_count = 0
            torch.save(model.state_dict(),
                       os.path.join(log_dir, 'best_model.pth'))
            print(f'  >>> New best! Epoch={best_epoch}  Test MAE={best_mae:.2f}')
        else:
            no_improve_count += 1

        if early_stop_patience > 0 and no_improve_count >= early_stop_patience:
            print(f'\nEarly stopping at epoch {epoch+1} '
                  f'(no improvement for {early_stop_patience} epochs).')
            break

    model.load_state_dict(
        torch.load(os.path.join(log_dir, 'best_model.pth'), map_location=device))
    print(f'\n=== Per-target MAE on test set ===')
    pt = per_target_mae(model, test_loader)
    for k, v in pt.items():
        print(f'  {k}: {v:.2f}')

    print(f'\nTraining finished. Best Test MAE: {best_mae:.2f} at epoch {best_epoch}')
    with open(os.path.join(log_dir, "final_summary.json"), "w") as f:
        json.dump({
            "best_val_mae": best_mae,
            "best_epoch": best_epoch,
            "per_target_test_mae": pt,
        }, f, indent=2)


if __name__ == '__main__':
    set_seed(42)
    p = argparse.ArgumentParser(description='FMAFusion Training')
    p.add_argument('--image_size', type=int, default=224, choices=[224, 384])
    p.add_argument('--swin_weights_rgb', type=str, default=None)
    p.add_argument('--swin_weights_depth', type=str, default=None)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--b', type=int, default=32, help='batch size')
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--lr', type=float, default=0.00004)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--beta1', type=float, default=0.9)
    p.add_argument('--beta2', type=float, default=0.999)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--early_stop_patience', type=int, default=30)
    p.add_argument('--exp_tag', type=str, default=None)

    cli = p.parse_args()
    exp_tag = cli.exp_tag or f"FMAFusion_swin{cli.image_size}"

    main(exp_tag=exp_tag,
         image_size=cli.image_size,
         swin_weights_rgb=cli.swin_weights_rgb,
         swin_weights_depth=cli.swin_weights_depth,
         learning_rate=cli.lr,
         weight_decay=cli.weight_decay,
         beta1=cli.beta1,
         beta2=cli.beta2,
         epochs=cli.epochs,
         batch_size=cli.b,
         num_workers=cli.num_workers,
         dropout=cli.dropout,
         early_stop_patience=cli.early_stop_patience)
