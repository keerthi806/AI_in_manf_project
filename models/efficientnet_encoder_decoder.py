import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2


class EfficientNetB2Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
        features = efficientnet_b2(weights=weights).features

        # Grayscale input stem.
        self.stem = features[0]
        old_conv = self.stem[0]
        conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        if pretrained:
            with torch.no_grad():
                conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        self.stem[0] = conv

        # EfficientNet-B2 stages 1..5.
        self.blocks = nn.Sequential(*list(features.children())[1:6])

    def forward(self, image):
        image = self.stem(image)
        outputs = []
        for i, block in enumerate(self.blocks):
            image = block(image)
            if i in (0, 1, 2, 4):
                outputs.append(image)
        return outputs


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class EffNetEncDec(nn.Module):
    def __init__(self, pretrained=True, embed_dim=256, num_heads=4,
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.name = "efficientnet_encoder_decoder"
        self.embed_dim = embed_dim

        self.backbone = EfficientNetB2Backbone(pretrained)
        self.token_proj = nn.Conv2d(120, embed_dim, 1)

        self.pos_embed = nn.Parameter(torch.randn(1, 256, embed_dim) * 0.02)
        self.type_embed = nn.Parameter(torch.randn(2, 1, embed_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers, enable_nested_tensor=False
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 3, 128, 1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )
        self.low_skip = nn.Sequential(
            nn.Conv2d(240, 128, 1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )
        self.skip1 = nn.Sequential(
            nn.Conv2d(32, 48, 1),
            nn.GroupNorm(8, 48),
            nn.GELU(),
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(48, 64, 1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.skip3 = nn.Sequential(
            nn.Conv2d(96, 64, 1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )

        self.up1 = UpBlock(128, 96, 64)
        self.up2 = UpBlock(96, 64, 64)
        self.up3 = UpBlock(64, 48, 48)
        self.up4 = UpBlock(48, 32)
        self.head = nn.Conv2d(32, 2, 3, padding=1)

    def forward(self, initial, deformed):
        i1, i2, i3, i5 = self.backbone(initial)
        d1, d2, d3, d5 = self.backbone(deformed)

        t1 = self.token_proj(i5).flatten(2).transpose(1, 2)
        t2 = self.token_proj(d5).flatten(2).transpose(1, 2)
        t1 = t1 + self.pos_embed + self.type_embed[0]
        t2 = t2 + self.pos_embed + self.type_embed[1]

        tokens = self.encoder(torch.cat([t1, t2], dim=1))
        n = t1.shape[1]

        a = tokens[:, :n].transpose(1, 2).reshape(-1, self.embed_dim, 16, 16)
        b = tokens[:, n:].transpose(1, 2).reshape(-1, self.embed_dim, 16, 16)

        fused = self.fuse(torch.cat([a, b, b - a], dim=1))
        fused = fused + self.low_skip(torch.cat([i5, d5], dim=1))

        s3 = self.skip3(torch.cat([i3, d3], dim=1))
        s2 = self.skip2(torch.cat([i2, d2], dim=1))
        s1 = self.skip1(torch.cat([i1, d1], dim=1))

        x = self.up1(fused, s3)
        x = self.up2(x, s2)
        x = self.up3(x, s1)
        x = self.up4(x)
        return self.head(x)


if __name__ == "__main__":
    model = EffNetEncDec(pretrained=False)
    prediction = model(torch.randn(2, 1, 256, 256), torch.randn(2, 1, 256, 256))
    print("prediction:", prediction.shape)
