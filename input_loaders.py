import torch
from torch.utils.data import Dataset
import pandas as pd
import utils
import random


class TrainDataset(Dataset):

    def __init__(self, input_table, interference_table, root_dir, interference_root_dir, n_speakers, mix_coeff, gradual,
                 testing=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.inputs_table = pd.read_csv(input_table,
                                        names=['youtube_id', 'time_start', 'time_end', 'relative_x', 'relative_y'],
                                        dtype=str)
        self.interference_table = pd.read_csv(interference_table,
                                              names=['youtube_id', 'time_start', 'time_end', 'relative_x',
                                                     'relative_y'], dtype=str)

        self.root_dir = root_dir
        self.interference_root_dir = interference_root_dir
        self.testing = testing
        self.n_speakers = n_speakers
        self.mix_coeff = mix_coeff
        self.gradual = gradual
        self.mix_factor = 0

    def update_mix_factor(self):
        self.mix_factor += 0.1
        self.mix_factor = min(1, self.mix_factor)

    def __len__(self):
        return len(self.inputs_table)

    def __getitem__(self, idx):
        while True:
            try:
                while True:
                    interference_idx = random.randint(0, len(self.interference_table) - 1)
                    if interference_idx != idx:
                        break
                original_face_embeds = utils.load_face_embeds(self.inputs_table.iloc[idx], self.root_dir)
                original_spectrogram = utils.load_spectrogram(self.inputs_table.iloc[idx], self.root_dir)
                interference_spectrogram = utils.load_spectrogram(self.interference_table.iloc[interference_idx],
                                                                  self.interference_root_dir)
                # if multiple speakers, load videos for them as well
                if self.n_speakers > 1:
                    interference_face_embeds = utils.load_face_embeds(self.interference_table.iloc[interference_idx],
                                                                      self.interference_root_dir)

                time_start = random.randint(0, original_face_embeds.shape[0] // 25 - 3)
                interference_time_start = random.randint(0, interference_spectrogram.shape[2] // 100 - 3)

                if self.n_speakers > 1:
                    interference_time_start = random.randint(0, interference_face_embeds.shape[0] // 25 - 3)

                frame_start = time_start * 25
                spec_start = time_start * 100
                interference_spec_start = interference_time_start * 100
                if self.n_speakers > 1:
                    interference_frame_start = interference_time_start * 25

                cropped_face_embeds = torch.from_numpy(
                    original_face_embeds[frame_start:frame_start + 3 * 25]).transpose(0, 1).float()
                cropped_original_spec = torch.from_numpy(
                    original_spectrogram[:, :, spec_start:spec_start + 301]).float()
                cropped_interference_spec = torch.from_numpy(
                    interference_spectrogram[:, :, interference_spec_start:interference_spec_start + 301]).float()
                if self.n_speakers > 1:
                    cropped_interference_face_embeds = torch.from_numpy(
                        interference_face_embeds[interference_frame_start:interference_frame_start + 3 * 25]).transpose(
                        0, 1).float()

                if self.gradual:
                    cropped_interference_spec *= (self.mix_coeff + (1 - self.mix_coeff) * self.mix_factor)
                else:
                    cropped_interference_spec *= self.mix_coeff
                mixed_spec = cropped_original_spec + cropped_interference_spec

                mixed_spec = utils.apply_power_compression(mixed_spec)
                cropped_original_spec = utils.apply_power_compression(cropped_original_spec)
                if self.n_speakers > 1:
                    cropped_interference_spec = utils.apply_power_compression(cropped_interference_spec)

                assert cropped_face_embeds.shape == (1792, 75) or cropped_face_embeds.shape == (736, 75) or  cropped_face_embeds.shape == (512, 75)
                if self.n_speakers > 1:
                    assert cropped_interference_face_embeds.shape == (
                        1792, 75) or cropped_interference_face_embeds.shape == (736, 75) or cropped_interference_face_embeds.shape == (512, 75)
                assert mixed_spec.shape == cropped_original_spec.shape == (2, 257, 301)
                if self.n_speakers == 1:
                    sample = {'face_embeds': cropped_face_embeds, 'mixed_spec': mixed_spec,
                              'clean_spec': cropped_original_spec}
                elif self.n_speakers > 1:
                    sample = {'face_embeds': torch.stack((cropped_face_embeds, cropped_interference_face_embeds)),
                              'mixed_spec': mixed_spec,
                              'clean_spec': torch.stack((cropped_original_spec, cropped_interference_spec))}
                else:
                    raise Exception('undef speaker count')

                if self.testing:
                    sample['clean_video_info'] = self.inputs_table.iloc[idx].to_dict()
                    sample['interference_video_info'] = self.inputs_table.iloc[interference_idx].to_dict()
            except Exception as e:
                print('failed to mix audio', str(e), idx, interference_idx)
                idx = random.randint(0, len(self.inputs_table) - 1)
                continue
            break
        return sample


class SingleLoader:

    def __len__(self):
        return 1

    def __getitem__(self, item):
        return utils.load_arbitrary_data()
