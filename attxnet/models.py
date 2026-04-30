"""
Lightweight backbone + attention module + classifier.
Backbones: ResNet18, MobileNetV3-Small, EfficientNet-B0 (via timm).
Attention: CBAM, Coordinate Attention (CA), or None.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ──────────────────────────── Attention Modules ────────────────────────────


class ChannelAttention(nn.Module):
    """Channel attention sub-module of CBAM."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg_pool = x.mean(dim=[2, 3])
        max_pool = x.amax(dim=[2, 3])
        attn = torch.sigmoid(self.fc(avg_pool) + self.fc(max_pool))
        return x * attn.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """Spatial attention sub-module of CBAM."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., ECCV 2018)."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x


class CoordinateAttention(nn.Module):
    """Coordinate Attention (Hou et al., CVPR 2021)."""

    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.conv1 = nn.Conv2d(channels, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(mid, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(mid, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        # pool along spatial dims
        x_h = x.mean(dim=3, keepdim=True).permute(0, 1, 3, 2)   # (b,c,1,h)
        x_w = x.mean(dim=2, keepdim=True)                         # (b,c,1,w)
        y = torch.cat([x_h, x_w], dim=3)                          # (b,c,1,h+w)
        y = y.permute(0, 1, 3, 2)                                 # (b,c,h+w,1)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = y.split([h, w], dim=2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w.permute(0, 1, 3, 2)).permute(0, 1, 3, 2))
        return x * a_h * a_w


ATTENTION_REGISTRY = {
    "cbam": CBAM,
    "ca": CoordinateAttention,
    "none": None,
}


# ──────────────────────────── Model Builder ────────────────────────────


class CrackClassifier(nn.Module):
    """
    Lightweight backbone + optional attention + binary classification head.

    Args:
        backbone_name: timm model name.
        attention: 'cbam', 'ca', or 'none'.
        pretrained: use ImageNet pre-trained weights.
        num_classes: output classes (default 2 for crack/non-crack).
        drop_rate: dropout before final FC.
    """

    BACKBONE_MAP = {
        "resnet18": "resnet18",
        "mobilenetv3": "mobilenetv3_small_100",
        "efficientnet": "efficientnet_b0",
    }

    def __init__(
        self,
        backbone_name: str = "mobilenetv3",
        attention: str = "cbam",
        pretrained: bool = True,
        num_classes: int = 2,
        drop_rate: float = 0.2,
    ):
        super().__init__()
        timm_name = self.BACKBONE_MAP.get(backbone_name, backbone_name)
        self.backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features

        attn_cls = ATTENTION_REGISTRY.get(attention)
        if attn_cls is not None:
            self.attn = attn_cls(feat_dim)
        else:
            self.attn = nn.Identity()

        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, num_classes),
        )
        self.backbone_name = backbone_name
        self.attention_name = attention
        self._feat_dim = feat_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before pooling (for Grad-CAM)."""
        return self.backbone.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.backbone.forward_features(x)  # (B, C, H, W) or (B, C)
        if feat_map.dim() == 4:
            feat_map = self.attn(feat_map)
            feat = feat_map.mean(dim=[2, 3])
        else:
            feat = feat_map
        return self.head(feat)

    def get_cam_target_layer(self):
        """Return the last spatial layer within forward_features() for Grad-CAM.

        MobileNetV3: forward_features = conv_stem→bn1→blocks  (conv_head is in forward_head)
        EfficientNet: forward_features = conv_stem→bn1→blocks→conv_head→bn2
        ResNet18:     forward_features = conv1→bn1→...→layer4
        """
        if hasattr(self.backbone, 'layer4'):
            return self.backbone.layer4[-1]
        if hasattr(self.backbone, 'blocks'):
            return self.backbone.blocks[-1]
        raise RuntimeError(f"Cannot determine CAM target layer for {self.backbone_name}")

    def __repr__(self):
        return (f"CrackClassifier(backbone={self.backbone_name}, "
                f"attention={self.attention_name}, feat_dim={self._feat_dim})")


def build_model(backbone: str = "mobilenetv3", attention: str = "cbam",
                pretrained: bool = True, num_classes: int = 2) -> CrackClassifier:
    return CrackClassifier(backbone, attention, pretrained, num_classes)
