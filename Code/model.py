import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class Config:
    num_freq: int = 256
    channels_input: int = 2
    num_output: int = 500
    conv_channels: tuple[int, ...] = (16, 32, 64, 64)
    vertical_stride: tuple[int, ...] = (1, 2, 2, 1)
    kernel_size: int = 3
    bottle_neck_out_channels: int = 8
    token_dim: int = 128
    num_transformer_heads: int = 4
    num_transformer_layers: int = 4
    dropout: float = 0.1

class ConvNet(nn.Module):
    def __init__(self, channels_input, conv_channels, vertical_stride, kernel_size, padding):
        super().__init__()
        
        self.layers = nn.ModuleList()
        in_ch = channels_input

        for i in range(len(conv_channels)):
            out_ch = conv_channels[i]
            v_stride = vertical_stride[i]
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=(v_stride, 1), padding=padding),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU()
                )
            )
            in_ch = out_ch

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv1x1(x)))
    
class Tokenizer(nn.Module):
    def __init__(self, freq_bins, in_channels, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.projection = nn.Linear(freq_bins * in_channels, embedding_dim)

    def forward(self, x):
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, -1)
        x = self.projection(x)

        position = torch.arange(0, t, dtype=torch.float, device=x.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2, device=x.device).float()
            * (-math.log(10000.0) / self.embedding_dim)
        )
        pe = torch.zeros(t, self.embedding_dim, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        x = x + pe.unsqueeze(0)
        return x
    
class TransformerNet(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, dropout):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer(x)


class Output(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        return self.fc(x.mean(dim=1))
    

class Talonet(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        total_stride = 1
        for stride in config.vertical_stride:
            total_stride *= stride

        freq_after_conv = config.num_freq // total_stride

        kernel_padding = config.kernel_size // 2

        self.conv_block = ConvNet(
            config.channels_input,
            config.conv_channels,
            config.vertical_stride,
            config.kernel_size,
            kernel_padding
        )
        self.bottleneck = Bottleneck(
            in_channels=config.conv_channels[-1],
            out_channels=config.bottle_neck_out_channels
        )
        self.tokenizer = Tokenizer(
            freq_bins=freq_after_conv,
            in_channels=config.bottle_neck_out_channels,
            embedding_dim=config.token_dim
        )
        self.transformer_block = TransformerNet(
            embedding_dim=config.token_dim,
            num_heads=config.num_transformer_heads,
            num_layers=config.num_transformer_layers,
            dropout=config.dropout
        )
        self.output_layer = Output(
            embedding_dim=config.token_dim,
            num_classes=config.num_output
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.bottleneck(x)
        x = self.tokenizer(x)
        x = self.transformer_block(x)
        x = self.output_layer(x)
        return x

import time

if __name__ == "__main__":
    cfg = Config(
        num_freq=512,
        channels_input=2,
        num_output=10,
        conv_channels=(4, 8, 16, 16),
        vertical_stride=(1, 1, 2, 2),
        kernel_size=5,
        bottle_neck_out_channels=4,
        num_transformer_heads=2,
        num_transformer_layers=2,
        dropout=0.1
    )
    model = Talonet(cfg)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Setup worked!")
    print(f"Total Params: {total_params:,}")

    dummy_input = torch.randn(1, cfg.channels_input, cfg.num_freq, 516)

    _ = model(dummy_input)

    start_time = time.perf_counter()
    output = model(dummy_input)
    end_time = time.perf_counter()

    print(f"Inference Time: {time.perf_counter() - start_time:.4f} seconds")
    print(f"Output Shape: {output.shape}")