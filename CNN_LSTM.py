import torch
import torch.nn as nn

def _best_gn_groups(ch):
    """Dynamically select the best group number for GroupNorm."""
    for g in (16, 8, 5, 4, 3, 2, 1):
        if ch % g == 0:
            return g
    return 1

class _DW_PW_Block(nn.Module):
    """
        Depthwise-Pointwise Convolution Block.
        A lightweight convolutional block used to process spatial features.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw  = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        self.gn1 = nn.GroupNorm(_best_gn_groups(in_ch), in_ch)
        self.act1= nn.ReLU()
        self.pw  = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
        self.act2= nn.ReLU()
        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.dw(x);  x = self.gn1(x); x = self.act1(x)
        x = self.pw(x);  x = self.act2(x)
        if self.proj is not None:
            identity = self.proj(identity)
        return x + identity


# Satellite Embedding Encoder
class AEFEncoderLite(nn.Module):
    """
        Encoder for Satellite Embeddings (SE).

        This module processes the fixed-size patches extracted via mask-aware ROIAlign.
        It uses a CNN backbone followed by Spatial Pyramid Pooling (SPP) to generate
        a fixed-length representation vector regardless of the original basin scale.

        Input:  (B, 65, K, K) -> 64 channels (Satellite) + 1 channel (Mask)
        Output: (B, attribute_embed)
    """
    def __init__(self, in_channels, mid_channels, attr_dim, mlp_drop):
        super().__init__()
        # CNN Backbone for feature extraction
        self.block0 = _DW_PW_Block(in_channels, mid_channels)
        self.block1 = _DW_PW_Block(mid_channels, mid_channels)

        # Squeeze-and-Excitation block for channel attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels, mid_channels // 4, 1), nn.ReLU(),
            nn.Conv2d(mid_channels // 4, mid_channels, 1), nn.Sigmoid()
        )
        self.reduce = nn.Conv2d(mid_channels, 32, kernel_size=1, bias=False)

        # Spatial Pyramid Pooling (SPP)
        # Captures multi-scale context by pooling at different grid sizes
        self.spp_sizes_max = (1, 2, 4)
        self.spp_sizes_avg = (1, 2)
        self.poolers_max = nn.ModuleList([nn.AdaptiveMaxPool2d((s, s)) for s in self.spp_sizes_max])
        self.poolers_avg = nn.ModuleList([nn.AdaptiveAvgPool2d((s, s)) for s in self.spp_sizes_avg])

        tiles_max = sum(s * s for s in self.spp_sizes_max)
        tiles_avg = sum(s * s for s in self.spp_sizes_avg)
        out_dim = 32 * (tiles_max + tiles_avg)

        # Final Projection
        self.readout = nn.Sequential(
            nn.Dropout(mlp_drop),
            nn.Linear(out_dim, attr_dim),
        )

    def forward(self, aef_in):
        x = self.block1(self.block0(aef_in))
        x = self.reduce(x * self.se(x))
        feats = [p(x).flatten(1) for p in list(self.poolers_max) + list(self.poolers_avg)]
        return self.readout(torch.cat(feats, dim=1))


# Main Model: LSTM with Dual Encoders
class LSTMRunoffModel(nn.Module):
    """
        Regional Rainfall-Runoff Model with Dual Encoders (Static + Satellite).

        This model fuses traditional static catchment attributes with deep satellite
        embeddings to generate a basin-specific representation ($z_{attr}$).
        This representation conditions the LSTM decoder for streamflow prediction.

        Args:
            - input_size: Number of dynamic forcing variables (V).
            - static_size: Number of static attributes (S).
            - attribute_embed: Dimension of the latent basin representation.
            - use_static: Whether to use the static attribute branch.
            - use_aef: Whether to use the satellite embedding branch.
    """
    def __init__(self, input_size, static_size, hidden_size, attribute_embed, num_layers=1, dropout=0.25,
                 use_static=True,  use_aef=True, aef_in_channels=65, aef_mid_channels=128,
                 attribute_embed_drop=0.00):
        super().__init__()
        self.use_static = use_static and static_size > 0
        self.use_aef    = use_aef
        self.attribute_embed = attribute_embed

        # Flag to check if we are fusing two branches
        self.is_fusion_mode = self.use_static and self.use_aef

        # Module 1: Static Attribute Encoder
        # Encodes tabular static data (S) into a latent vector.
        if self.use_static:
            mid_s = max(attribute_embed // 2, static_size)
            self.static_encoder = nn.Sequential(
                nn.Linear(static_size, mid_s),
                nn.ReLU(),
                nn.Dropout(attribute_embed_drop),
                nn.Linear(mid_s, attribute_embed)
            )
        else:
            self.static_encoder = None

        # Module 2: Satellite Embedding Encoder
        # Encodes spatial satellite data (Xb, Mb) into a latent vector.
        if self.use_aef:
            self.aef_encoder = AEFEncoderLite(
                in_channels  = aef_in_channels,
                mid_channels = aef_mid_channels,
                attr_dim     = attribute_embed,
                mlp_drop     = attribute_embed_drop,
            )
        else:
            self.aef_encoder = None

        # Module 3: Adaptive Gating (Scale Balancing)
        # Applies learnable scaling factors to dynamically balance the magnitude of features
        # from different modalities to prevent one branch from dominating the fusion
        if self.is_fusion_mode:
            self.static_gate = nn.Parameter(torch.tensor(0.3, dtype=torch.float32))
            self.aef_gate    = nn.Parameter(torch.tensor(0.3, dtype=torch.float32))
        else:
            self.static_gate = None
            self.aef_gate    = None

        # Module 4: Feature Fusion
        # Fuses the two representations into a single basin descriptor.
        if self.is_fusion_mode:
            self.fuse = nn.Linear(attribute_embed * 2, attribute_embed, bias=True)
        else:
            self.fuse = None

        # Module 5: LSTM Decoder & Head
        # Temporal prediction module conditioned on the basin descriptor.
        fused_dim = attribute_embed
        self.lstm = nn.LSTM(
            input_size=input_size + fused_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, max(hidden_size // 4, 1)),
            nn.ReLU(),
            nn.Linear(max(hidden_size // 4, 1), 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Custom weight initialization for stability."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                hidden = param.shape[0] // 4
                param.data[hidden:2*hidden].add_(1.0)
        last = list(self.head)[-1]
        if isinstance(last, nn.Linear):
            nn.init.xavier_uniform_(last.weight)
            if last.bias is not None: nn.init.zeros_(last.bias)

    def _check_aef_triplet(self, Xb, XM, Mb):
        """Validate shapes of the satellite triplet (Features, Masked Features, Mask)."""
        B = Xb.shape[0]
        if Xb.shape != XM.shape:
            raise ValueError(f"Shape mismatch Xb/XM: {Xb.shape} vs {XM.shape}")
        if not (Mb.shape[0] == B and Mb.shape[1] == 1 and Mb.ndim == 4):
            raise ValueError(f"Mb shape must be (B,1,H,W), got {Mb.shape}")
        if Mb.shape[-2:] != Xb.shape[-2:]:
            raise ValueError(f"Mb spatial size must match Xb: {Mb.shape[-2:]} vs {Xb.shape[-2:]}")

    def _encode_static(self, x_static):
        z = self.static_encoder(x_static)
        if self.static_gate is not None:
            # z = z / z.norm(p=2, dim=1, keepdim=True).clamp_min(1e-4)
            z = z * self.static_gate.to(dtype=z.dtype, device=z.device)
        return z

    def _encode_aef_triplet(self, Xb, XM, Mb):
        self._check_aef_triplet(Xb, XM, Mb)
        aef_in = torch.cat([Xb, Mb], dim=1)
        z = self.aef_encoder(aef_in)
        if self.aef_gate is not None:
            # z = z / z.norm(p=2, dim=1, keepdim=True).clamp_min(1e-4)
            z = z * self.aef_gate.to(dtype=z.dtype, device=z.device)
        return z

    def forward(self, x_seq, x_static, aef_triplet):
        """
            x_seq: (B, T, V) - Dynamic forcing time series.
            x_static: (B, S) - Static basin attributes.
            aef_triplet: (Xb, XM, Mb) - Satellite embedding tensors.
        """
        B, T, V = x_seq.shape
        device = x_seq.device
        dtype  = x_seq.dtype
        parts = []
        z_static = None
        z_aef = None

        # Encode Static Attributes
        if self.use_static:
            z_static = self._encode_static(x_static.to(device=device, dtype=dtype))
            parts.append(z_static)

        # Encode Satellite Embeddings
        if self.use_aef:
            Xb, XM, Mb = aef_triplet
            z_aef = self._encode_aef_triplet(
                Xb.to(device=device, dtype=dtype, non_blocking=True),
                XM.to(device=device, dtype=dtype, non_blocking=True),
                Mb.to(device=device, dtype=dtype, non_blocking=True),
            )
            parts.append(z_aef)

        # Fuse Representations
        if self.is_fusion_mode:
            z_attr = torch.cat([z_static, z_aef], dim=1)
            z_attr = self.fuse(z_attr)
        else:
            # Pass through the single active branch directly
            z_attr = parts[0]

        # Temporal Broadcasting (Conditioning)
        # Replicate the static basin representation for every time step
        z_b = z_attr.unsqueeze(1).expand(B, T, z_attr.shape[-1])
        x_cat = torch.cat([x_seq, z_b], dim=-1)

        # LSTM Decoding
        out, _ = self.lstm(x_cat)
        y = self.head(out[:, -1, :])

        return y
