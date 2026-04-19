from dataclasses import dataclass

@dataclass
class Config:
    sample_rate: int = 32000
    f_min: int = 50
    f_max: int = 14000
    hop_length: int = 310
    n_fft_list: tuple[int, ...] = (256, 2048)


    num_freq: int = 256
    max_time_steps: int = 512
    min_time_steps: int = 128
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