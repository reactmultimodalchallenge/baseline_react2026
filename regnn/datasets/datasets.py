import os
import random
import csv
import io
import tarfile
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


DEFAULT_EEG_TARGET_COLS = [
    "TP9", "AF7", "AF8", "TP10",
    "Delta_TP9", "Theta_TP9", "Alpha_TP9", "Beta_TP9", "Gamma_TP9",
    "Delta_TP10", "Theta_TP10", "Alpha_TP10", "Beta_TP10", "Gamma_TP10"
]
EEG_RAW_CHANNELS = {"TP9", "AF7", "AF8", "TP10"}


def custom_collate(batch):
    speaker_param_inputs = [item[0] for item in batch]
    speaker_video_inputs = [item[1] for item in batch]
    speaker_audio_inputs = [item[2] for item in batch]
    listener_param_gts = [item[3] for item in batch]
    cp = [item[4] for item in batch]
    num_frames = [item[5] for item in batch]
    has_eeg = len(batch[0]) > 6

    collated = (
        speaker_param_inputs,
        speaker_video_inputs,
        speaker_audio_inputs,
        listener_param_gts,
        cp,
        num_frames,
    )
    if has_eeg:
        listener_eeg_targets = [item[6] for item in batch]
        listener_eeg_masks = [item[7] for item in batch]
        collated = collated + (listener_eeg_targets, listener_eeg_masks)
    return collated


class ActionData(Dataset):
    def __init__(self, root, data_type, num_frames=50, neighbor_pattern='all', k_select=None,
                 load_eeg_l=False, eeg_dir_name='eeg_processed', eeg_target_cols=None,
                 eeg_channel_scale=1000.0, eeg_use_tar_fallback=True, fps=30,
                 bidirectional=False):
        self.root_dir = Path(root)
        self.split = data_type
        self.data_dir = self.root_dir / self.split
        self.video_dir = self.data_dir / 'video-features'
        self.audio_dir = self.data_dir / 'audio-features'
        self.param_dir = self.data_dir / 'facial-attributes'
        self.eeg_dir = self.data_dir / eeg_dir_name
        self.eeg_tar_path = self.data_dir / f'{eeg_dir_name}.tar.gz'
        self.eeg_tar_members = None
        self.num_frames = num_frames  # 50
        self.neighbor_pattern = neighbor_pattern  # 'all'
        self.k_select = 10 if k_select is None else k_select
        self.load_eeg_l = load_eeg_l
        self.eeg_target_cols = list(eeg_target_cols) if eeg_target_cols is not None else DEFAULT_EEG_TARGET_COLS
        self.eeg_channel_scale = float(eeg_channel_scale)
        self.eeg_use_tar_fallback = eeg_use_tar_fallback
        self.fps = fps
        self.bidirectional = bidirectional

        gt_path_dict = {}
        for root, _, files in os.walk(self.video_dir):
            for path in files:
                path = Path(path)
                file, ext = path.stem, path.suffix
                if ext.lower() != '.pth':
                    continue

                session_id = Path(*Path(root).parts[-2:])  # listener/session0
                file_path = session_id / file
                if session_id not in gt_path_dict:
                    gt_path_dict[session_id] = [file_path]
                else:
                    gt_path_dict[session_id].append(file_path)

        speaker_path_list = []
        listener_path_list = []
        gt_path_list = []

        for root, _, files in os.walk(self.video_dir):
            for path in files:
                path = Path(path)
                file, ext = path.stem, path.suffix
                if ext.lower() != '.pth':
                    continue

                parts = Path(root).parts
                file_path = Path(*parts[-2:]) / file
                role = parts[-2]
                session_id = Path(parts[-1])
                if not self.bidirectional and role != 'speaker':
                    continue
                gt_session_id = 'speaker' / session_id if role == 'listener' else 'listener' / session_id
                listener_file_path = gt_session_id / file

                speaker_path_list.append(file_path)
                listener_path_list.append(listener_file_path)
                listener_gt_paths = gt_path_dict[gt_session_id]
                gt_path_list.append(listener_gt_paths)

        self.speaker_path_list = speaker_path_list.copy()
        self.listener_path_list = listener_path_list.copy()
        self.gt_path_list = gt_path_list.copy()
        self._len = len(self.speaker_path_list)

    def segment_torch2d(self, t: torch.Tensor, seg_len: int) -> torch.Tensor:
        L, dim = t.size()
        pad = (-L) % seg_len
        padded = F.pad(t, (0, 0, 0, pad))
        return padded.view(-1, seg_len, dim)

    @staticmethod
    def _is_archive_metadata(path):
        parts = Path(path).parts
        name = Path(path).name
        return name.startswith('._') or name == '.DS_Store' or 'PaxHeader' in parts

    def _read_eeg_text(self, rel_path):
        eeg_path = self.eeg_dir / rel_path.with_suffix('.csv')
        if eeg_path.exists() and not self._is_archive_metadata(eeg_path):
            return eeg_path.read_text(encoding='utf-8-sig')

        if not self.eeg_use_tar_fallback or not self.eeg_tar_path.exists():
            return None

        if self.eeg_tar_members is None:
            with tarfile.open(self.eeg_tar_path, 'r:gz') as tar:
                self.eeg_tar_members = {
                    name for name in tar.getnames()
                    if not self._is_archive_metadata(name)
                }

        member_candidates = [
            os.fspath(Path(self.eeg_dir.name) / rel_path.with_suffix('.csv')).replace('\\', '/'),
            os.fspath(rel_path.with_suffix('.csv')).replace('\\', '/'),
        ]
        member = next((name for name in member_candidates if name in self.eeg_tar_members), None)
        if member is None:
            return None

        with tarfile.open(self.eeg_tar_path, 'r:gz') as tar:
            extracted = tar.extractfile(member)
            if extracted is None:
                return None
            return extracted.read().decode('utf-8-sig')

    def _load_eeg(self, rel_path, total_length):
        eeg_dim = len(self.eeg_target_cols)
        empty_target = torch.zeros(size=(total_length, eeg_dim), dtype=torch.float32)
        empty_mask = torch.zeros(size=(total_length, eeg_dim), dtype=torch.float32)

        text = self._read_eeg_text(rel_path)
        if text is None:
            return empty_target, empty_mask

        rows = list(csv.DictReader(io.StringIO(text)))
        if len(rows) == 0:
            return empty_target, empty_mask

        values = np.zeros((len(rows), eeg_dim), dtype=np.float32)
        mask = np.zeros((len(rows), eeg_dim), dtype=np.float32)
        for row_idx, row in enumerate(rows):
            for col_idx, col in enumerate(self.eeg_target_cols):
                raw_value = row.get(col, '')
                if raw_value == '':
                    continue
                try:
                    value = float(raw_value)
                except ValueError:
                    continue
                if not np.isfinite(value):
                    continue
                if col in EEG_RAW_CHANNELS:
                    value = value / self.eeg_channel_scale
                values[row_idx, col_idx] = value
                mask[row_idx, col_idx] = 1.0

        frame_to_eeg = np.floor(np.arange(total_length) / self.fps).astype(np.int64)
        frame_to_eeg = np.clip(frame_to_eeg, 0, len(rows) - 1)
        return (
            torch.from_numpy(values[frame_to_eeg]),
            torch.from_numpy(mask[frame_to_eeg]),
        )

    def _chunk_end_indices(self, total_length):
        chunk_count = int(np.ceil(total_length / self.num_frames))
        return torch.tensor(
            [min((chunk_idx + 1) * self.num_frames, total_length) - 1
             for chunk_idx in range(chunk_count)],
            dtype=torch.long,
        )

    def __getitem__(self, index):
        speaker_path = self.speaker_path_list[index]
        # e.g., speaker/session*/Camera-2024-06-21-103121-103102

        # speaker's video features
        s_video_path = Path(self.video_dir) / speaker_path.with_suffix('.pth')
        s_video_inputs = torch.load(s_video_path)

        # speaker's audio features
        s_audio_path = Path(self.audio_dir) / speaker_path.with_suffix('.npy')
        s_audio_inputs = torch.from_numpy(np.load(s_audio_path))

        # speaker's emotion features
        s_param_path = Path(self.param_dir) / speaker_path.with_suffix('.npy')
        s_param_inputs = torch.from_numpy(np.load(s_param_path))

        listener_path = self.listener_path_list[index]
        # e.g., listener/session*/Camera-2024-06-21-103121-103102
        listener_paths = self.gt_path_list[index]
        if len(listener_paths) >= self.k_select:
            listener_paths = ([listener_path] +
                              random.sample([p for p in listener_paths if p != listener_path], self.k_select - 1)) \
                if len(listener_paths) >= self.k_select else random.choices(listener_paths, k=self.k_select)

        total_length = s_video_inputs.size(0)
        num_frames = total_length if self.split == "test" else self.num_frames

        cp = random.randint(0, total_length - num_frames)
        # get speaker's audio and video data inputs
        s_video_inputs = s_video_inputs[cp: cp + num_frames]
        s_audio_inputs = s_audio_inputs[cp: cp + num_frames]
        # 'train' [n_frames, d]

        if self.split == "test":
            s_video_inputs = self.segment_torch2d(s_video_inputs, self.num_frames)
            s_audio_inputs = self.segment_torch2d(s_audio_inputs, self.num_frames)
            # 'test' [N, n_frames, d]

        # listener's emotion features
        l_param_gts = [torch.from_numpy(np.load(Path(self.param_dir) / l_path.with_suffix('.npy')))
                       for l_path in listener_paths]

        sample = (s_param_inputs, s_video_inputs, s_audio_inputs, l_param_gts, cp, total_length)
        if self.load_eeg_l:
            listener_eeg, listener_eeg_mask = self._load_eeg(listener_path, total_length)
            if self.split == "test":
                end_indices = self._chunk_end_indices(total_length)
                listener_eeg = listener_eeg[end_indices]
                listener_eeg_mask = listener_eeg_mask[end_indices]
            else:
                target_idx = min(cp + num_frames - 1, total_length - 1)
                listener_eeg = listener_eeg[target_idx]
                listener_eeg_mask = listener_eeg_mask[target_idx]
            sample = sample + (listener_eeg, listener_eeg_mask)

        return sample

    def __len__(self):
        return self._len


class ActionDataloader:
    def __init__(self,
                 root: str = '../data',
                 num_frames: int = 50,
                 neighbor_pattern: str = 'all',
                 batch_size: int = 16,
                 shuffle: bool = True,
                 num_workers: int = 8,
                 k_select: int = 10,
                 load_eeg_l: bool = False,
                 eeg_dir_name: str = 'eeg_processed',
                 eeg_target_cols=None,
                 eeg_channel_scale: float = 1000.0,
                 eeg_use_tar_fallback: bool = True,
                 fps: int = 30,
                 bidirectional: bool = False,
                 **kwargs):
        self.data_dir = root
        self.num_frames = num_frames
        self.neighbor_pattern = neighbor_pattern
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.k_select = k_select
        self.load_eeg_l = load_eeg_l
        self.eeg_dir_name = eeg_dir_name
        self.eeg_target_cols = eeg_target_cols
        self.eeg_channel_scale = eeg_channel_scale
        self.eeg_use_tar_fallback = eeg_use_tar_fallback
        self.fps = fps
        self.bidirectional = bidirectional
        self.collate_fn_dict = {'none': None,
                                'custom': custom_collate}

    def get_dataloader(self, data_type: str = 'train',
                       collate_fn: str = 'custom', **kwargs):
        dataset = ActionData(
            root=self.data_dir,
            data_type=data_type,
            num_frames=self.num_frames,
            neighbor_pattern=self.neighbor_pattern,
            k_select=self.k_select,
            load_eeg_l=self.load_eeg_l,
            eeg_dir_name=self.eeg_dir_name,
            eeg_target_cols=self.eeg_target_cols,
            eeg_channel_scale=self.eeg_channel_scale,
            eeg_use_tar_fallback=self.eeg_use_tar_fallback,
            fps=self.fps,
            bidirectional=self.bidirectional,
        )

        dataloader = DataLoader(dataset=dataset,
                                collate_fn=self.collate_fn_dict[collate_fn],
                                batch_size=self.batch_size,
                                shuffle=self.shuffle,
                                num_workers=self.num_workers,)
        return dataloader
