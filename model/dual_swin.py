import torch
import torch.nn as nn

from .swin_backbone import (swin_base_patch4_window7_224_in22k,
                            swin_base_patch4_window12_384_in22k,
                            load_swin_weights)
from .fma_fusion import FMAFusion, FMAFusionHead

_SWIN_CHANNELS = (128, 256, 512, 1024)


class DualSwin(nn.Module):
    """Dual Swin-Base encoder → FMAFusion → Late-Pool Head (Method, Figure 2).
    Two independent Swin-Base backbones (ImageNet-1K pretrained, unshared weights)
    extract 4-level feature pyramids (Method, Eq.1).
    """
    def __init__(self,
                 image_size: int = 224,
                 swin_weights_rgb: str | None = None,
                 swin_weights_depth: str | None = None,
                 dropout: float = 0.1):
        super().__init__()

        swin_fn = (swin_base_patch4_window12_384_in22k if image_size == 384
                   else swin_base_patch4_window7_224_in22k)

        self.encoder_rgb = _build_swin_branch(swin_fn, swin_weights_rgb, 'RGB', image_size)
        wpath = swin_weights_depth or swin_weights_rgb
        self.encoder_depth = _build_swin_branch(swin_fn, wpath, 'Depth', image_size)

        self.image_size = image_size

        # FMAFusion module (FMAFusion Module): frequency-domain multi-scale fusion
        self.fma_fusion = FMAFusion(
            in_dims=_SWIN_CHANNELS,
            common_dim=256,
            output_dim=1024,
            low_radius=2,
            dropout=dropout,
            verbose=True,
        )

        # Late-Pool Fusion Head (Late-Pool Fusion Head)
        self.head = FMAFusionHead(
            in_dim=1024, hidden_dim=512, num_targets=5, dropout=dropout,
        )

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor):
        # Dual-stream feature extraction (Method)
        rgb_feats = self.encoder_rgb(rgb)
        depth_feats = self.encoder_depth(depth)

        # FMAFusion: cross-modal frequency-domain fusion (FMAFusion Module)
        fused = self.fma_fusion(rgb_feats, depth_feats)
        # Late-Pool Head → 5 nutrition predictions (Late-Pool Fusion Head)
        pred, attn_map = self.head(fused)

        return pred, {'attn_map': attn_map}


def _build_swin_branch(swin_fn, weights_path: str | None, tag: str,
                       image_size: int):
    model = swin_fn(num_classes=0)
    if weights_path:
        load_swin_weights(model, weights_path)
    print(f'  [DualSwin] Swin-Base {tag} loaded '
          f'(in22k, {image_size}px, weights={weights_path})')
    return model
