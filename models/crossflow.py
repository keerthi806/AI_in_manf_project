import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


class ResNet34Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = ResNet34_Weights.DEFAULT if pretrained else None
        resnet = resnet34(weights=weights)

        # Replace the 3-channel first conv with a 1-channel conv.
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                new_conv.weight.copy_(resnet.conv1.weight.mean(dim=1, keepdim=True))
        resnet.conv1 = new_conv

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2

    def forward(self, image):
        x = self.stem(image)
        high_res = self.layer1(x)   # [B, 64, H/4, W/4]
        low_res = self.layer2(high_res)  # [B, 128, H/8, W/8]
        return high_res, low_res


class SpatialTransformerEncoder(nn.Module):
    def __init__(self, in_channels=128, embed_dim=128, grid_size=32,
                 num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()

        self.grid_size = grid_size
        self.input_proj = (
            nn.Identity() if in_channels == embed_dim
            else nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        )
        self.pos_embed = nn.Parameter(
            torch.randn(1, grid_size * grid_size, embed_dim) * 0.02
        )

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, features):
        features = self.input_proj(features)
        tokens = features.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed
        tokens = self.encoder(tokens)
        return self.norm(tokens)


class CrossAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, dropout=0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query, context):
        attended, _ = self.attention(
            query=query,
            key=context,
            value=context,
            need_weights=False,
        )
        x = self.norm1(query + attended)
        return self.norm2(x + self.ffn(x))


class GlobalMatching(nn.Module):
    def __init__(self, grid_size=32, temperature=0.07):
        super().__init__()
        self.grid_size = grid_size
        self.temperature = temperature

    def forward(self, initial_tokens, final_tokens):
        batch_size = initial_tokens.shape[0]
        size = self.grid_size

        initial_tokens = F.normalize(initial_tokens, dim=-1)
        final_tokens = F.normalize(final_tokens, dim=-1)

        similarity = torch.bmm(initial_tokens, final_tokens.transpose(1, 2))
        probabilities = F.softmax(similarity / self.temperature, dim=-1)

        y, x = torch.meshgrid(
            torch.arange(size, device=initial_tokens.device),
            torch.arange(size, device=initial_tokens.device),
            indexing="ij",
        )
        coords = torch.stack([x.reshape(-1), y.reshape(-1)], dim=-1)
        coords = coords.to(initial_tokens.dtype).unsqueeze(0).expand(batch_size, -1, -1)

        matched = torch.bmm(probabilities, coords)
        flow = matched - coords
        return flow.transpose(1, 2).reshape(batch_size, 2, size, size)


class DisplacementDecoder(nn.Module):
    def __init__(self, feature_channels=64, coarse_scale=8.0, output_size=256):
        super().__init__()

        self.coarse_scale = coarse_scale
        self.output_size = output_size

        self.refine = nn.Sequential(
            nn.Conv2d(feature_channels * 2 + 2, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 2, 3, padding=1),
        )

    def forward(self, initial_features, final_features, coarse_flow):
        flow_64 = F.interpolate(
            coarse_flow,
            size=initial_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ) * self.coarse_scale

        residual = self.refine(torch.cat([initial_features, final_features, flow_64], dim=1))
        residual = F.interpolate(
            residual,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )

        base_flow = F.interpolate(
            coarse_flow,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        ) * self.coarse_scale

        return base_flow + residual


class TransformerDIC(nn.Module):
    def __init__(self, pretrained=True, embed_dim=128, grid_size=32,
                 num_heads=4, num_layers=2, dropout=0.1,
                 temperature=0.07, decoder_channels=64, image_size=256):
        super().__init__()
        self.name = "crossflow"

        self.backbone = ResNet34Backbone(pretrained=pretrained)

        self.transformer = SpatialTransformerEncoder(
            in_channels=128,
            embed_dim=embed_dim,
            grid_size=grid_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.cross_attention = CrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.matcher = GlobalMatching(
            grid_size=grid_size,
            temperature=temperature,
        )
        self.decoder = DisplacementDecoder(
            feature_channels=64,
            coarse_scale=image_size / grid_size,
            output_size=image_size,
        )

    def forward(self, initial, final):
        initial_high, initial_low = self.backbone(initial)
        final_high, final_low = self.backbone(final)

        initial_tokens = self.transformer(initial_low)
        final_tokens = self.transformer(final_low)

        matched_tokens = self.cross_attention(initial_tokens, final_tokens)
        coarse_flow = self.matcher(matched_tokens, final_tokens)

        return self.decoder(initial_high, final_high, coarse_flow)


class CharbonnierEndpointLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        error = pred - target
        endpoint_error_sq = torch.sum(error ** 2, dim=1)
        return torch.sqrt(endpoint_error_sq + self.eps ** 2).mean()


if __name__ == "__main__":
    model = TransformerDIC(pretrained=False)
    initial = torch.randn(2, 1, 256, 256)
    final = torch.randn(2, 1, 256, 256)

    with torch.no_grad():
        prediction = model(initial, final)

    print("prediction:", prediction.shape)
