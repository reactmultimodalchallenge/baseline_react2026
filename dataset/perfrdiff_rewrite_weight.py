import csv
import io
import os
import random
import tarfile
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dataset.tools.util import extract_audio_features


DEFAULT_EEG_TARGET_COLS = [
    "TP9", "AF7", "AF8", "TP10",
    "Delta_TP9", "Theta_TP9", "Alpha_TP9", "Beta_TP9", "Gamma_TP9",
    "Delta_TP10", "Theta_TP10", "Alpha_TP10", "Beta_TP10", "Gamma_TP10",
]
EEG_RAW_CHANNELS = {"TP9", "AF7", "AF8", "TP10"}


def _empty_tensor():
    return torch.zeros(size=(0,))


def collate_fit(batch):
    columns = list(zip(*batch))
    return tuple(torch.stack(items, dim=0) if items[0].numel() > 0 else torch.zeros(size=(len(items), 0))
                 for items in columns)


def collate_test(batch):
    columns = list(zip(*batch))
    return tuple(list(column) for column in columns)


class PerFRDiffRewriteWeightDataset(Dataset):
    def __init__(
            self,
            root_dir,
            split="train",
            clip_length=750,
            target_size=224,
            crop_size=224,
            fps=30,
            audio_feature_type="wav2vec",
            load_video_s=False,
            load_video_l=False,
            load_audio=True,
            load_emotion_s=True,
            load_emotion_l=True,
            load_3dmm_s=True,
            load_3dmm_l=True,
            load_ref=False,
            load_personality_l=False,
            personal_condition_mode="3dmm_personality",
            personality_dir_name="personality",
            load_eeg_l=False,
            eeg_dir_name="eeg_processed",
            eeg_target_cols=None,
            eeg_channel_scale=1000.0,
            eeg_use_tar_fallback=True,
            normalize_3dmm="standard",
            num_test_gts=10,
            bidirectional=False,
            **kwargs,
    ):
        self.root_dir = root_dir
        self.split = split
        self.clip_length = clip_length
        self.fps = fps
        self.audio_feature_type = audio_feature_type
        self.load_video_s = load_video_s
        self.load_video_l = load_video_l
        self.load_audio = load_audio
        self.load_emotion_s = load_emotion_s
        self.load_emotion_l = load_emotion_l
        self.load_3dmm_s = load_3dmm_s
        self.load_3dmm_l = load_3dmm_l
        self.load_ref = load_ref
        self.personal_condition_mode = personal_condition_mode
        if self.personal_condition_mode not in {"3dmm_personality", "personality_only", "3dmm_only"}:
            raise ValueError(f"Unknown personal_condition_mode: {self.personal_condition_mode}")
        self.load_personality_l = load_personality_l and self.personal_condition_mode != "3dmm_only"
        self.load_eeg_l = load_eeg_l
        self.eeg_target_cols = list(eeg_target_cols) if eeg_target_cols is not None else DEFAULT_EEG_TARGET_COLS
        self.eeg_channel_scale = eeg_channel_scale
        self.eeg_use_tar_fallback = eeg_use_tar_fallback
        self._eeg_tar_members = None
        self.num_test_gts = num_test_gts
        self.bidirectional = bidirectional

        split_dir = Path(root_dir) / split
        self.audio_dir = split_dir / ("audio-features" if audio_feature_type == "wav2vec" else "audio")
        self.video_dir = split_dir / "video-face-crop"
        self.emotion_dir = split_dir / "facial-attributes"
        self.coeff_dir = split_dir / "coefficients"
        self.personality_dir = split_dir / personality_dir_name
        self.eeg_dir = split_dir / eeg_dir_name
        self.eeg_tar_path = split_dir / f"{eeg_dir_name}.tar.gz"
        self.personality_cols = [
            "Extraversion",
            "Agreeableness",
            "Conscientiousness",
            "Neuroticism",
            "Openness",
        ]
        self.listener_personality_by_session = {}
        self.listener_personality_by_stem = {}
        self.personality_by_role_session = {}
        self.personality_by_role_stem = {}
        if self.load_personality_l:
            self._load_personality_index("listener", required=True)
            if self.bidirectional:
                self._load_personality_index("speaker", required=False)

        mean_face_path = os.path.join(hydra.utils.get_original_cwd(), "external/FaceVerse/mean_face.npy")
        std_face_path = os.path.join(hydra.utils.get_original_cwd(), "external/FaceVerse/std_face.npy")
        self.mean_face = torch.FloatTensor(np.load(mean_face_path).astype(np.float32)).view(1, 1, -1)
        self.std_face = torch.FloatTensor(np.load(std_face_path).astype(np.float32)).view(1, 1, -1)
        if normalize_3dmm == "standard":
            self.transform_3dmm = transforms.Lambda(lambda e: (e - self.mean_face) / self.std_face)
        elif normalize_3dmm == "zero_center":
            self.transform_3dmm = transforms.Lambda(lambda e: e - self.mean_face)
        else:
            raise ValueError(f"Unknown normalize_3dmm: {normalize_3dmm}")

        self.samples = []
        self.gt_path_dict = self._build_gt_index()
        self._build_samples()

    @staticmethod
    def _is_real_sample_file(file_name):
        path = Path(file_name)
        if path.suffix.lower() != ".npy":
            return False
        return not path.name.startswith(".") and not path.name.startswith("._")

    def _iter_emotion_files(self):
        for root, dirs, files in os.walk(self.emotion_dir):
            dirs[:] = [
                name for name in dirs
                if not name.startswith(".") and name not in {"__MACOSX", "PaxHeader"}
            ]
            root_parts = Path(root).parts
            if len(root_parts) < 2:
                continue
            role = root_parts[-2]
            session = root_parts[-1]
            if role not in {"speaker", "listener"}:
                continue
            for file_name in files:
                if not self._is_real_sample_file(file_name):
                    continue
                yield role, session, Path(file_name).stem

    def _build_gt_index(self):
        gt_path_dict = {}
        for role, session, stem in self._iter_emotion_files():
            session_id = Path(role) / session
            file_path = session_id / stem
            gt_path_dict.setdefault(session_id, []).append(file_path)
        return gt_path_dict

    def _load_personality_index(self, role, required=True):
        path = self.personality_dir / f"{role}.csv"
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Missing {role} personality file: {path}")
            return

        with path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            for row in reader:
                stem = Path(row.get("video_name", "")).stem
                if not stem:
                    continue
                values = [
                    (float(row[column]) - 1.0) / 4.0
                    for column in self.personality_cols
                ]
                personality = torch.tensor(values, dtype=torch.float32)
                session = row.get("session")
                if session:
                    self.personality_by_role_session[(role, session, stem)] = personality
                    if role == "listener":
                        self.listener_personality_by_session[(session, stem)] = personality
                self.personality_by_role_stem[(role, stem)] = personality
                if role == "listener":
                    self.listener_personality_by_stem[stem] = personality

    def _has_target_personality(self, listener_path):
        if not self.load_personality_l:
            return True

        role = listener_path.parts[0]
        session = listener_path.parts[1] if len(listener_path.parts) > 1 else None
        stem = listener_path.stem
        if session is not None and (role, session, stem) in self.personality_by_role_session:
            return True
        return (role, stem) in self.personality_by_role_stem

    def _has_required(self, speaker_path, listener_path):
        required = [
            self.emotion_dir / speaker_path.with_suffix(".npy"),
            self.emotion_dir / listener_path.with_suffix(".npy"),
            self.coeff_dir / speaker_path.with_suffix(".npy"),
            self.coeff_dir / listener_path.with_suffix(".npy"),
        ]
        if self.audio_feature_type == "wav2vec":
            required.append(self.audio_dir / speaker_path.with_suffix(".npy"))
        else:
            required.append(self.audio_dir / speaker_path.with_suffix(".wav"))
        return all(path.exists() for path in required)

    def _build_samples(self):
        for role, session, stem in self._iter_emotion_files():
            if not self.bidirectional and role != "speaker":
                continue
            speaker_path = Path(role) / session / stem
            listener_role = "listener" if role == "speaker" else "speaker"
            listener_path = Path(listener_role) / session / stem
            if not self._has_required(speaker_path, listener_path):
                continue
            if not self._has_target_personality(listener_path):
                continue
            self.samples.append(
                {
                    "speaker_path": speaker_path,
                    "listener_path": listener_path,
                    "gt_paths": self.gt_path_dict.get(Path(listener_role) / session, [listener_path]),
                }
            )

    def __len__(self):
        return len(self.samples)

    def _pad_clip(self, clip, target_len):
        clip = clip[:target_len]
        if clip.shape[0] >= target_len:
            return clip
        pad_shape = (target_len - clip.shape[0], *clip.shape[1:])
        return torch.cat((clip, clip.new_zeros(pad_shape)), dim=0)

    @staticmethod
    def _load_numpy_array(path, name):
        try:
            return np.load(path, allow_pickle=False)
        except ValueError as exc:
            if "pickled" in str(exc):
                raise ValueError(
                    f"{name} file is not a numeric numpy sample: {path}. "
                    "It is probably an AppleDouble/archive metadata file such as '._*.npy'. "
                    "Remove those metadata files from the dataset or keep them hidden so the loader can skip them."
                ) from exc
            raise

    def _load_emotion(self, rel_path):
        path = self.emotion_dir / rel_path.with_suffix(".npy")
        return torch.from_numpy(self._load_numpy_array(path, "emotion")).float()

    def _load_audio(self, rel_path, total_length):
        if self.audio_feature_type == "wav2vec":
            path = self.audio_dir / rel_path.with_suffix(".npy")
            audio = self._load_numpy_array(path, "audio")
        elif self.audio_feature_type == "mfcc":
            audio = extract_audio_features(os.fspath(self.audio_dir / rel_path.with_suffix(".wav")),
                                           self.fps,
                                           total_length)
        else:
            raise ValueError(f"Unknown audio_feature_type: {self.audio_feature_type}")
        return torch.from_numpy(audio).float()

    def _load_3dmm(self, rel_path):
        path = self.coeff_dir / rel_path.with_suffix(".npy")
        coeff = torch.FloatTensor(self._load_numpy_array(path, "3DMM")).squeeze()
        return self.transform_3dmm(coeff)[0].float()

    def _load_personality(self, listener_path):
        if not self.load_personality_l:
            return _empty_tensor()

        role = listener_path.parts[0]
        session = listener_path.parts[1] if len(listener_path.parts) > 1 else None
        stem = listener_path.stem
        if session is not None and (role, session, stem) in self.personality_by_role_session:
            return self.personality_by_role_session[(role, session, stem)].clone()
        if (role, stem) in self.personality_by_role_stem:
            return self.personality_by_role_stem[(role, stem)].clone()
        raise FileNotFoundError(f"Missing target personality for {listener_path}")

    @staticmethod
    def _is_archive_metadata(path):
        parts = Path(path).parts
        name = Path(path).name
        return name.startswith("._") or name == ".DS_Store" or "PaxHeader" in parts

    def _read_eeg_text(self, rel_path):
        eeg_path = self.eeg_dir / rel_path.with_suffix(".csv")
        if eeg_path.exists() and not self._is_archive_metadata(eeg_path):
            return eeg_path.read_text(encoding="utf-8-sig")

        if not self.eeg_use_tar_fallback or not self.eeg_tar_path.exists():
            return None

        if self._eeg_tar_members is None:
            with tarfile.open(self.eeg_tar_path, "r:gz") as tar:
                self._eeg_tar_members = {
                    name for name in tar.getnames()
                    if not self._is_archive_metadata(name)
                }

        member_candidates = [
            os.fspath(Path(self.eeg_dir.name) / rel_path.with_suffix(".csv")).replace("\\", "/"),
            os.fspath(rel_path.with_suffix(".csv")).replace("\\", "/"),
        ]
        member = next((name for name in member_candidates if name in self._eeg_tar_members), None)
        if member is None:
            return None

        with tarfile.open(self.eeg_tar_path, "r:gz") as tar:
            extracted = tar.extractfile(member)
            if extracted is None:
                return None
            return extracted.read().decode("utf-8-sig")

    def _load_eeg(self, listener_path, total_length):
        eeg_dim = len(self.eeg_target_cols)
        empty_target = torch.zeros(size=(total_length, eeg_dim), dtype=torch.float32)
        empty_mask = torch.zeros(size=(total_length, eeg_dim), dtype=torch.float32)

        text = self._read_eeg_text(listener_path)
        if text is None:
            return empty_target, empty_mask

        rows = list(csv.DictReader(io.StringIO(text)))
        if len(rows) == 0:
            return empty_target, empty_mask

        values = np.zeros((len(rows), eeg_dim), dtype=np.float32)
        mask = np.zeros((len(rows), eeg_dim), dtype=np.float32)
        for row_idx, row in enumerate(rows):
            for col_idx, col in enumerate(self.eeg_target_cols):
                raw_value = row.get(col, "")
                if raw_value == "":
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

    def _choose_personal_path(self, listener_path):
        session_id = Path(*listener_path.parts[:2])
        candidates = [path for path in self.gt_path_dict.get(session_id, []) if path != listener_path]
        if not candidates:
            return listener_path
        return random.choice(candidates)

    def _sample_test_gts(self, listener_path, gt_paths):
        candidates = [path for path in gt_paths if path != listener_path]
        paths = [listener_path]
        if len(candidates) >= self.num_test_gts - 1:
            paths.extend(random.sample(candidates, self.num_test_gts - 1))
        elif candidates:
            paths.extend(random.choices(candidates, k=self.num_test_gts - 1))
        else:
            paths.extend([listener_path] * (self.num_test_gts - 1))
        return paths

    def _load_personal_clip(self, listener_path, deterministic=False):
        personal_3dmm = self._load_3dmm(self._choose_personal_path(listener_path))
        if deterministic or personal_3dmm.shape[0] <= self.clip_length:
            cp = 0
        else:
            cp = random.randint(0, personal_3dmm.shape[0] - self.clip_length)
        return self._pad_clip(personal_3dmm[cp:cp + self.clip_length], self.clip_length)

    def __getitem__(self, index):
        sample = self.samples[index]
        speaker_path = sample["speaker_path"]
        listener_path = sample["listener_path"]

        speaker_emotion = self._load_emotion(speaker_path)
        listener_emotion = self._load_emotion(listener_path)
        speaker_3dmm = self._load_3dmm(speaker_path)
        listener_3dmm = self._load_3dmm(listener_path)
        listener_personality = self._load_personality(listener_path)

        total_length = min(
            speaker_emotion.shape[0],
            listener_emotion.shape[0],
            speaker_3dmm.shape[0],
            listener_3dmm.shape[0],
        )
        speaker_audio = self._load_audio(speaker_path, total_length)
        total_length = min(total_length, speaker_audio.shape[0])
        listener_eeg = listener_eeg_mask = _empty_tensor()
        if self.load_eeg_l:
            listener_eeg, listener_eeg_mask = self._load_eeg(listener_path, total_length)

        if self.split == "test":
            gt_paths = self._sample_test_gts(listener_path, sample["gt_paths"])
            listener_emotion_gts = [self._load_emotion(path) for path in gt_paths]
            listener_3dmm_gts = [self._load_3dmm(path) for path in gt_paths]
            listener_clip_lengths = torch.tensor([emotion.shape[0] for emotion in listener_emotion_gts])
            personal_3dmm = _empty_tensor() if self.personal_condition_mode == "personality_only" \
                else self._load_personal_clip(listener_path, deterministic=True)
            return (
                speaker_audio[:total_length],
                _empty_tensor(),
                speaker_emotion[:total_length],
                speaker_3dmm[:total_length],
                _empty_tensor(),
                listener_emotion_gts,
                listener_3dmm_gts,
                personal_3dmm,
                listener_personality,
                listener_eeg,
                listener_eeg_mask,
                torch.tensor(total_length),
                listener_clip_lengths,
            )

        cp = random.randint(0, total_length - self.clip_length) if total_length > self.clip_length else 0
        personal_3dmm = _empty_tensor() if self.personal_condition_mode == "personality_only" \
            else self._load_personal_clip(listener_path)

        return (
            self._pad_clip(speaker_audio[cp:cp + self.clip_length], self.clip_length),
            _empty_tensor(),
            self._pad_clip(speaker_emotion[cp:cp + self.clip_length], self.clip_length),
            self._pad_clip(speaker_3dmm[cp:cp + self.clip_length], self.clip_length),
            _empty_tensor(),
            self._pad_clip(listener_emotion[cp:cp + self.clip_length], self.clip_length),
            self._pad_clip(listener_3dmm[cp:cp + self.clip_length], self.clip_length),
            personal_3dmm,
            listener_personality,
            self._pad_clip(listener_eeg[cp:cp + self.clip_length], self.clip_length)
            if self.load_eeg_l else listener_eeg,
            self._pad_clip(listener_eeg_mask[cp:cp + self.clip_length], self.clip_length)
            if self.load_eeg_l else listener_eeg_mask,
            _empty_tensor(),
        )


class PerFRDiffRewriteWeightDataModule:
    def __init__(
            self,
            train_dataset: DictConfig = None,
            validation_dataset: DictConfig = None,
            test_dataset: DictConfig = None,
            **kwargs,
    ):
        self.seed = kwargs.pop("seed", 1234)
        self.train_set_cfg = train_dataset
        self.val_set_cfg = validation_dataset
        self.test_set_cfg = test_dataset

    def get_dataloader(self, stage):
        def worker_init_fn(worker_id):
            seed = self.seed + worker_id
            random.seed(seed)
            np.random.seed(seed)

        if stage == "fit":
            train_dataset = instantiate(self.train_set_cfg)
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=self.train_set_cfg.batch_size,
                shuffle=self.train_set_cfg.shuffle,
                num_workers=self.train_set_cfg.num_workers,
                collate_fn=collate_fit,
                worker_init_fn=worker_init_fn,
            )
            val_dataset = instantiate(self.val_set_cfg)
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.val_set_cfg.batch_size,
                shuffle=self.val_set_cfg.shuffle,
                num_workers=self.val_set_cfg.num_workers,
                collate_fn=collate_fit,
                worker_init_fn=worker_init_fn,
            )
            return train_loader, val_loader

        test_dataset = instantiate(self.test_set_cfg)
        return DataLoader(
            dataset=test_dataset,
            batch_size=self.test_set_cfg.batch_size,
            shuffle=self.test_set_cfg.shuffle,
            num_workers=self.test_set_cfg.num_workers,
            collate_fn=collate_test,
            worker_init_fn=worker_init_fn,
        )
