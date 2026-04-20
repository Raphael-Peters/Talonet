import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchaudio
import pandas as pd
from pathlib import Path
from audioprocessor import DataProcessor
from config import Config
import shutil
from tqdm import tqdm
import zipfile
import random
import requests

class TalonetDataset(Dataset):
    def __init__(self, 
                csv_file, 
                root_dir, 
                temp_sounds_dir,
                temp_bird_dir,
                temp_noise_dir,
                noise_zip_dir,
                config: Config, 
                epoch_size, 
                samples_per_epoch, 
                label_dim):
        self.df = pd.read_csv(csv_file)
        
        self.root_dir = Path(root_dir)
        self.temp_dir = self.root_dir / temp_sounds_dir
        self.bird_dir = self.temp_dir / temp_bird_dir
        self.noise_dir = self.temp_dir / temp_noise_dir
        self.noise_zip = self.root_dir / noise_zip_dir

        self.bird_dir.mkdir(parents=True, exist_ok=True)
        self.noise_dir.mkdir(parents=True, exist_ok=True)

        self.del_temp_noise_sounds()
        self.del_temp_bird_sounds()

        self.prepare_noise_sounds()

        self.weights = self._calculate_weights()

        self.cfg = config
        self.epoch_size = epoch_size
        self.samples_per_epoch = samples_per_epoch
        self.species_count = label_dim
        self.min_audio_samples = config.min_time_steps * config.hop_length
        self.max_audio_samples = config.max_time_steps * config.hop_length
        
        self.processor = DataProcessor(config)
        self.epoch_data = []

    def _calculate_weights(self):
        print(f"Calculating probability weights for {len(self.df)} entries...")
        
        all_labels = self.df['label_id'].astype(str).str.split(',').explode()
        species_counts = all_labels.str.strip().value_counts().to_dict()
        
        species_counts = {int(k): v for k, v in species_counts.items()}

        def get_avg_inverse(label_str):
            labels = [int(l.strip()) for l in str(label_str).split(',')]
            
            avg_count = sum(species_counts.get(l, 1) for l in labels) / len(labels)
            return 1.0 / avg_count

        weights = self.df['label_id'].apply(get_avg_inverse).values
        
        weights_tensor = torch.from_numpy(weights).double()
        total_sum = weights_tensor.sum()
        
        if total_sum > 0:
            weights_tensor /= total_sum
            
        print(f"Done! Max Probability: {weights_tensor.max():.8f}")
        return weights_tensor
    
    def del_temp_noise_sounds(self):
        for item in self.noise_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    def prepare_noise_sounds(self):
        if not any(self.noise_dir.iterdir()):
            if self.noise_zip.exists():
                print(f"Unzipping {self.noise_zip.name} to {self.noise_dir}...")
                with zipfile.ZipFile(self.noise_zip, 'r') as zip_ref:
                    zip_ref.extractall(self.noise_dir)
                print("Noise data unzipped!")
            else:
                print(f"Warning: No {self.noise_zip.name} found at {self.noise_zip}!")
        else:
            print(f"Warning! {self.noise_dir} is already filled. This should not be the case.")

    def del_temp_bird_sounds(self):
        for item in self.bird_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    def download_sample(self, url):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception as e:
            print(f"Error downloading sample from {url}: {e}")
            return None

    def download_bird_samples(self):
            sampled_indices = torch.multinomial(self.weights, self.samples_per_epoch, replacement=True)
            
            self.epoch_data = []
            
            print(f"Selecting {self.samples_per_epoch} samples for the new epoch...")

            loop = tqdm(sampled_indices, desc='Download')

            for i, idx in enumerate(loop):
                row = self.df.iloc[idx.item()]
                
                sample_info = {
                    'label': str(row['label_id']),
                    'local_path': self.bird_dir / f"sample_{i}.wav"
                }

                self.epoch_data.append(sample_info)

                url = row['url']
                audio_tensor = self.download_sample(sample_info, url)
                
                if audio_tensor is not None:
                    torchaudio.save(str(sample_info['local_path']), audio_tensor, self.cfg.sample_rate)
                
            print(f"Download Complete!")

    def load_new_samples(self):
        print("Deleting old content...")
        self.del_temp_bird_sounds()
        
        print('Loading new content...')
        self.download_bird_samples()

    def __len__(self):
        return self.epoch_size
    
    def __getitem__(self, index):
        target_samples = random.randint(self.min_audio_samples, self.max_audio_samples)
        
        mixed_waveform = torch.zeros((1, target_samples))
        multi_hot_label = torch.zeros(self.species_count)
        
        noise_files = list(self.noise_dir.glob("*.wav"))
        if noise_files:
            noise_path = random.choice(noise_files)
            noise_wav, sr = torchaudio.load(str(noise_path))
            noise_wav = self.processor._prepare_waveform(noise_wav, sr)
            
            if noise_wav.shape[1] >= target_samples:
                start = random.randint(0, noise_wav.shape[1] - target_samples)
                noise_segment = noise_wav[:, start:start + target_samples]
            else:
                noise_segment = F.pad(noise_wav, (0, target_samples - noise_wav.shape[1]), mode='circular')
                
            mixed_waveform += noise_segment * random.uniform(0.0, 0.5)

        num_birds = random.randint(0, 3)
        if num_birds > 0 and len(self.epoch_data) > 0:
            selected_samples = random.sample(self.epoch_data, min(num_birds, len(self.epoch_data)))
            
            for bird_info in selected_samples:
                bird_wav, sr = torchaudio.load(str(bird_info['local_path']))
                bird_wav = self.processor._prepare_waveform(bird_wav, sr)

                bird_len = bird_wav.shape[1]
                
                if bird_len >= target_samples:
                    start = random.randint(0, bird_len - target_samples)
                    bird_segment = bird_wav[:, start:start + target_samples]
                else:
                    bird_segment = torch.zeros((1, target_samples))
                    offset = random.randint(0, target_samples - bird_len)
                    bird_segment[:, offset:offset + bird_len] = bird_wav
                
                mixed_waveform += bird_segment * random.uniform(0.6, 1.2)
                
                labels = [int(l.strip()) for l in bird_info['label'].split(',')]
                for l_id in labels:
                    if l_id < self.species_count:
                        multi_hot_label[l_id] = 1.0

        spectrogram = self.processor.get_spectrogram(mixed_waveform)
        spectrogram = self.processor.spec_augment(spectrogram, 15, 30, 3)

        return spectrogram, multi_hot_label