import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

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
    def __init__(self, freq_bins, in_channels, embedding_dim, max_time_steps):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.projection = nn.Linear(freq_bins * in_channels, embedding_dim)

        pe = torch.zeros(max_time_steps, embedding_dim)
        position = torch.arange(0, max_time_steps, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() 
            * (-math.log(10000.0) / embedding_dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        b, c, f, t = x.shape
        
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, -1)
        x = self.projection(x)

        x = x + self.pe[:, :t, :]
        
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
    
class RecursiveConvReduction(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.translator = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, stride=2, padding=1)
        self.conv = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.translator(x))
        while x.shape[-1] > 2:
            x = self.relu(self.bn(self.conv(x)))
        x = self.relu(self.conv(x))
        return x.view(x.shape[0], -1)
    
class PoolingReduction(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.fc = nn.Linear(embedding_dim * 2, embedding_dim)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        mean_pool = torch.mean(x, dim=1) 
        max_pool, _ = torch.max(x, dim=1)
        
        combined = torch.cat([mean_pool, max_pool], dim=-1)
        
        out = self.relu(self.fc(combined))
        return out
    
class Output(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
    

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
            embedding_dim=config.token_dim,
            max_time_steps=config.max_time_steps
        )
        self.transformer_block = TransformerNet(
            embedding_dim=config.token_dim,
            num_heads=config.num_transformer_heads,
            num_layers=config.num_transformer_layers,
            dropout=config.dropout
        )
        if config.reduction == 'conv':
            self.reduction = RecursiveConvReduction(
                embedding_dim=config.token_dim
            )
        else:
            self.reduction = PoolingReduction(
                embedding_dim=config.token_dim
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
        x = self.reduction(x)
        x = self.output_layer(x)
        return x