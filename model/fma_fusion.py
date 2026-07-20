import torch
import torch.nn as nn
import torch.nn.functional as F


def _create_low_mask(size: int, low_radius: int, device: torch.device):
    """Create a rectangular low-frequency mask L (FMAFusion Module Stage B, Eq.2 context)."""
    mask = torch.zeros(size, size, device=device)
    center = size // 2
    r = min(low_radius, center)
    mask[center - r:center + r + 1, center - r:center + r + 1] = 1.0
    return mask


class LowFreqWeightGenerator(nn.Module):
    """Sample-adaptive weight generator (FMAFusion Module Stage B, Eq.2).
    GAP + MLP → 8 scalar weights per sample.
    """
    def __init__(self, common_dim: int = 256, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(common_dim * 8, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 8),
        )
        self.temp = nn.Parameter(torch.tensor(1.0))  # learnable temperature τ

    def forward(self, F_list: list):
        B = F_list[0].shape[0]
        stats = [self.pool(f).view(B, -1) for f in F_list]
        cat = torch.cat(stats, dim=1)
        logits = self.mlp(cat)
        weights = F.softmax(logits * self.temp, dim=-1)
        return weights.view(B, 8, 1, 1, 1)


class ModalFusion(nn.Module):
    """Per-level RGB–Depth high-frequency fusion (FMAFusion Module Stage C, Eq.4).
    Sigmoid-gated weighted sum with learnable α_i per level.
    """
    def __init__(self, dim: int = 256):
        super().__init__()
        self.rgb_weight = nn.Parameter(torch.tensor(0.5))  # σ(α_i)

    def forward(self, mag_rgb: torch.Tensor, mag_depth: torch.Tensor):
        w = torch.sigmoid(self.rgb_weight)
        return w * mag_rgb + (1 - w) * mag_depth


class CrossLevelHighFreqGate(nn.Module):
    """Cross-level high-frequency channel gating (Cross-Level High-Frequency Gate, Eq.5–7).
    Aggregates S1–S3 GAP descriptors → MLP → channel gate g → S4 enhancement.
    """
    def __init__(self, dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def forward(self, high_mags_1to3: list, mag_high_s4: torch.Tensor):
        pooled = []
        for hm in high_mags_1to3:
            pooled.append(self.pool(hm).flatten(1))
        cat = torch.cat(pooled, dim=1)           # Eq.6: z = [GAP(M¹); GAP(M²); GAP(M³)]
        gate = self.mlp(cat)                      # Eq.7: g = Sigmoid(MLP(z))
        gate = gate.view(gate.shape[0], gate.shape[1], 1, 1)
        enhanced = mag_high_s4 * gate              # M⁴_rgb ⊙ H ⊙ g
        enhanced = enhanced + self.refine(enhanced) # + Conv3×3 residual (Eq.7)
        return enhanced


class FMAFusion(nn.Module):
    """FMAFusion module (FMAFusion Module): frequency-domain multi-scale fusion.
    All cross-modal, cross-scale aggregation performed within magnitude spectrum.
    """
    def __init__(self,
                 in_dims: tuple = (128, 256, 512, 1024),
                 common_dim: int = 256,
                 output_dim: int = 1024,
                 low_radius: int = 2,
                 dropout: float = 0.1,
                 verbose: bool = True):
        super().__init__()

        self.common_dim = common_dim
        self.output_dim = output_dim
        self.low_radius = low_radius
        self.verbose = verbose
        self._first_forward = True

        # Stage A: 1×1 projection to unified dimension d=256 (FMAFusion Module Stage A)
        self.stage_proj_rgb = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dims[i], common_dim, 1, bias=False),
                nn.BatchNorm2d(common_dim),
                nn.GELU(),
            ) for i in range(4)
        ])
        self.stage_proj_depth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dims[i], common_dim, 1, bias=False),
                nn.BatchNorm2d(common_dim),
                nn.GELU(),
            ) for i in range(4)
        ])

        # Stage B: learnable low-frequency weight generator (FMAFusion Module Stage B)
        self.low_weight_gen = LowFreqWeightGenerator(common_dim, dropout=dropout)

        # Stage C: per-level ModalFusion (FMAFusion Module Stage C, Eq.4)
        self.modal_fusions = nn.ModuleList([
            ModalFusion(common_dim) for _ in range(4)
        ])

        # Cross-level high-frequency gate (Cross-Level High-Frequency Gate, Eq.5–7)
        self.cross_level_gate = CrossLevelHighFreqGate(dim=common_dim, hidden_dim=128)

        # High-frequency refinement conv (FMAFusion Module Stage C)
        self.high_refine = nn.Sequential(
            nn.Conv2d(common_dim, common_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(common_dim),
            nn.GELU(),
        )

        # Stage D: output projection (FMAFusion Module Stage D)
        self.output_proj = nn.Sequential(
            nn.Conv2d(common_dim, output_dim, 1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.GELU(),
        )

        # Stage D: spatial residual bypass (FMAFusion Module Stage D)
        s4_in_dim = in_dims[3] * 2  # cat(RGB⁴, Depth⁴)
        self.residual_proj = nn.Sequential(
            nn.Conv2d(s4_in_dim, output_dim, 1, bias=False),
            nn.BatchNorm2d(output_dim),
        )

        self.out_norm = nn.LayerNorm(output_dim)

    def forward(self, rgb_feats: list, depth_feats: list):
        device = rgb_feats[0].device

        target_h, target_w = rgb_feats[3].shape[2], rgb_feats[3].shape[3]

        # Stage A: upsample S1–S3 to S4 resolution (7×7) (FMAFusion Module Stage A)
        rgb_aligned = []
        depth_aligned = []
        for i in range(4):
            if i < 3:
                rgb_aligned.append(F.interpolate(
                    rgb_feats[i], size=(target_h, target_w),
                    mode='bilinear', align_corners=False))
                depth_aligned.append(F.interpolate(
                    depth_feats[i], size=(target_h, target_w),
                    mode='bilinear', align_corners=False))
            else:
                rgb_aligned.append(rgb_feats[i])
                depth_aligned.append(depth_feats[i])

        # Stage A: 1×1 projection to common dim d=256 (FMAFusion Module Stage A)
        F_rgb = []
        F_depth = []
        for i in range(4):
            F_rgb.append(self.stage_proj_rgb[i](rgb_aligned[i]))
            F_depth.append(self.stage_proj_depth[i](depth_aligned[i]))

        # Stage A: FFT → magnitude + phase (FMAFusion Module Stage A)
        X_rgb = []
        X_depth = []
        for i in range(4):
            X_rgb.append(torch.fft.fftshift(
                torch.fft.fft2(F_rgb[i].to(torch.float32)), dim=(-2, -1)))
            X_depth.append(torch.fft.fftshift(
                torch.fft.fft2(F_depth[i].to(torch.float32)), dim=(-2, -1)))

        mag_rgb = [torch.abs(X) for X in X_rgb]
        mag_depth = [torch.abs(X) for X in X_depth]
        phase4 = torch.angle(X_rgb[3])  # keep only S4 RGB phase; discard the rest (FMAFusion Module Stage A)

        # Stage B: low-frequency mask L (radius r=2) (FMAFusion Module Stage B)
        spatial_size = target_h
        mask_low = _create_low_mask(spatial_size, self.low_radius, device)
        mask_low = mask_low.view(1, 1, spatial_size, spatial_size)
        mask_high = 1.0 - mask_low

        # Stage B: sample-adaptive weights via GAP + MLP (FMAFusion Module Stage B)
        F_all = F_rgb + F_depth
        mag_all = mag_rgb + mag_depth
        low_weights = self.low_weight_gen(F_all)

        # Stage B: weighted low-frequency aggregation (Eq.3)
        mag_low = torch.zeros_like(mag_all[0])
        for i in range(8):
            mag_low = mag_low + low_weights[:, i] * mag_all[i] * mask_low

        # Stage C: per-level ModalFusion on high frequencies (FMAFusion Module Stage C, Eq.4)
        fused_high = []
        for i in range(4):
            fh = self.modal_fusions[i](
                mag_rgb[i] * mask_high, mag_depth[i] * mask_high)
            fused_high.append(fh)

        # Stage C: cross-level high-frequency gating (Cross-Level High-Frequency Gate, Eq.5–7)
        mag_high = self.cross_level_gate(
            [fused_high[0], fused_high[1], fused_high[2]], mag_rgb[3] * mask_high)
        mag_high = self.high_refine(mag_high)

        # Stage D: combine low + high → reconstruct with S4 RGB phase (FMAFusion Module Stage D)
        mag_fused = mag_low + mag_high
        X_fused = mag_fused * torch.exp(1j * phase4)

        X_fused = torch.fft.ifftshift(X_fused, dim=(-2, -1))
        F_fused = torch.fft.ifft2(X_fused).real

        F_out = self.output_proj(F_fused)

        # Stage D: spatial residual bypass (S4 RGB + S4 Depth → 1×1) (FMAFusion Module Stage D)
        s4_cat = torch.cat([rgb_aligned[3], depth_aligned[3]], dim=1)
        residual = self.residual_proj(s4_cat)
        F_out = F_out + residual

        # Stage D: LayerNorm (FMAFusion Module Stage D)
        F_out = F_out.permute(0, 2, 3, 1)
        F_out = self.out_norm(F_out)
        F_out = F_out.permute(0, 3, 1, 2)

        if self.verbose and self._first_forward:
            self._first_forward = False
            self._print_shapes(rgb_feats, depth_feats, rgb_aligned, depth_aligned,
                               F_rgb, F_depth, X_rgb, X_depth, mag_rgb, mag_depth,
                               mask_low, low_weights, mag_low, mag_high,
                               mag_fused, F_fused, F_out, residual)

        return F_out

    def _print_shapes(self, rgb_feats, depth_feats, rgb_aligned, depth_aligned,
                      F_rgb, F_depth, X_rgb, X_depth, mag_rgb, mag_depth,
                      mask_low, low_weights, mag_low, mag_high,
                      mag_fused, F_fused, F_out, residual):
        input_size = rgb_feats[0].shape[-1]
        print(f'\n[FMAFusion] Shape trace (input={input_size}px, '
              f'low_radius={self.low_radius}):')
        print(f'  {"Stage":<6} {"RGB raw":<18} {"Depth raw":<18} '
              f'{"F_rgb":<18} {"F_depth":<18} {"X_rgb":<18} {"X_depth":<18}')
        for i in range(4):
            def s(t): return str(list(t.shape))
            print(f'  S{i+1}    {s(rgb_feats[i]):<18} {s(depth_feats[i]):<18} '
                  f'{s(F_rgb[i]):<18} {s(F_depth[i]):<18} '
                  f'{s(X_rgb[i]):<18} {s(X_depth[i]):<18}')
        print(f'  mask_low: {list(mask_low.shape)}  '
              f'low_weights(8): {list(low_weights.shape)}')
        print(f'  mag_low:  {list(mag_low.shape)}  '
              f'mag_high:  {list(mag_high.shape)}')
        print(f'  mag_fused:{list(mag_fused.shape)}  '
              f'F_fused: {list(F_fused.shape)}')
        print(f'  F_out:    {list(F_out.shape)}  '
              f'residual:  {list(residual.shape)}')
        print(f'  [FMAFusion] Final output: {list(F_out.shape)}\n')


class FMAFusionHead(nn.Module):
    """Late-Pool Fusion Head (Late-Pool Fusion Head): spatial attention before GAP.
    Applies attention map A → reweight → GAP → shared MLP → 5 experts.
    """
    def __init__(self, in_dim: int = 1024, hidden_dim: int = 512,
                 num_targets: int = 5, dropout: float = 0.1):
        super().__init__()
        # Spatial attention: Conv1×1 → BN → GELU → Conv1×1 → Sigmoid
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 4, 1, bias=False),
            nn.BatchNorm2d(in_dim // 4),
            nn.GELU(),
            nn.Conv2d(in_dim // 4, 1, 1),
            nn.Sigmoid(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Shared encoder (universal nutrition representation)
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 5 per-target experts: calories, mass, fat, carb, protein
        self.experts = nn.ModuleList([
            nn.Linear(hidden_dim // 2, 1) for _ in range(num_targets)
        ])

    def forward(self, fused_feat: torch.Tensor):
        attn_map = self.spatial_attn(fused_feat)
        x_weighted = fused_feat * attn_map
        x_pooled = self.gap(x_weighted).flatten(1)
        shared_feat = self.shared(x_pooled)
        preds = [expert(shared_feat) for expert in self.experts]
        pred = torch.cat(preds, dim=1)
        return pred, attn_map


if __name__ == '__main__':
    print("=" * 60)
    print("FMAFusion — Unit Test")
    print("=" * 60)

    B = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def make_feats(H, W):
        return [
            torch.randn(B, 128, H // 4, W // 4, device=device),
            torch.randn(B, 256, H // 8, W // 8, device=device),
            torch.randn(B, 512, H // 16, W // 16, device=device),
            torch.randn(B, 1024, H // 32, W // 32, device=device),
        ]

    print(f'\n--- FMAFusion (CrossLevelHighFreqGate) ---')
    fma = FMAFusion(verbose=True).to(device)
    n = sum(p.numel() for p in fma.parameters())
    print(f'  FMAFusion params: {n:,}')

    rgb, depth = make_feats(224, 224), make_feats(224, 224)
    fused = fma(rgb, depth)
    print(f'  Output: {list(fused.shape)}')

    head = FMAFusionHead().to(device)
    pred, attn = head(fused)
    print(f'  Pred (FMAFusionHead): {list(pred.shape)}  '
          f'AttnMap: {list(attn.shape)}')

    (pred.sum() + fused.sum()).backward()
    print(f'  Gradient OK')

    print(f'\n{"=" * 60}')
    print('All tests passed!')
    print('=' * 60)
