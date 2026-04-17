import torch
import torchaudio.transforms as T
import torch.nn.functional as F
from config import Config

class DataProcessor(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.sr = config.sample_rate
        self.n_mels = config.num_freq
        self.hop_length = config.hop_length
        self.max_time_steps = config.max_time_steps

        self.mel_transforms = torch.nn.ModuleList([
            T.MelSpectrogram(
                sample_rate=self.sr,
                n_fft=n,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                f_min=config.f_min,
                f_max=config.f_max,
                center=True,
                pad_mode="reflect",
                power=2.0
            ) for n in sorted(config.n_fft_list)
        ])

        self.db_transform = T.AmplitudeToDB()

    def _prepare_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean()
        std = tensor.std()
        if std < 1e-5:
            return tensor - mean
        return (tensor - mean) / (std + 1e-6)

    def _apply_max_limit(self, spec: torch.Tensor) -> torch.Tensor:
        t = spec.shape[-1]
        if t > self.max_time_steps:
            return spec[:, :, :self.max_time_steps]
        return spec

    def get_spectrogram(self, waveform: torch.Tensor, t_start_sec: float = 0.0) -> torch.Tensor:
        waveform = self._prepare_waveform(waveform)
        start_sample = int(t_start_sec * self.sr)
        sliced_wav = waveform[:, start_sample:]

        channel_list = []
        for mel_layer in self.mel_transforms:
            spec = self.db_transform(mel_layer(sliced_wav))
            
            spec = self._apply_max_limit(spec)
            channel_list.append(self._normalize(spec))

        return torch.cat(channel_list, dim=0)

    def get_relative_slice(self, waveform: torch.Tensor, t_rel: float) -> torch.Tensor:
        assert 0.0 <= t_rel <= 1.0, "t_rel must be between 0 and 1"

        waveform = self._prepare_waveform(waveform)
        total_samples = waveform.shape[1]
        start_sample = int(t_rel * total_samples)
        t_start_sec = start_sample / self.sr

        return self.get_spectrogram(waveform, t_start_sec=t_start_sec)

    def get_instant_frequencies(self, waveform: torch.Tensor, t_sec: float, window_size: int = 2048):
        waveform = self._prepare_waveform(waveform)
        center_sample = int(t_sec * self.sr)
        start_sample = max(0, min(center_sample - window_size // 2, waveform.shape[1] - window_size))
        segment = waveform[:, start_sample:start_sample + window_size]

        if segment.shape[1] < window_size:
            segment = F.pad(segment, (0, window_size - segment.shape[1]))

        hann_window = torch.hann_window(window_size, device=segment.device)
        fft_result = torch.fft.rfft(segment * hann_window)

        freqs = torch.fft.rfftfreq(window_size, 1 / self.sr)
        magnitudes = torch.abs(fft_result).squeeze(0)

        return freqs, magnitudes