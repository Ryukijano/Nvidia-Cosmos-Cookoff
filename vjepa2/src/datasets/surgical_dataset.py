# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pathlib
import yaml
from logging import getLogger

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .utils.dataloader import ConcatIndices, MonitoredDataset, NondeterministicDataLoader
from .utils.weighted_sampler import DistributedWeightedSampler

_GLOBAL_SEED = 0
logger = getLogger()


def make_surgical_dataset(
    data_path,
    batch_size,
    training=True,
    frames_per_clip=10,
    frame_step=1,
    num_clips=1,
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    deterministic=True,
    log_dir=None,
):
    split = 'train' if training else 'val'
    dataset = SurgicalPhaseDataset(
        data_path=data_path,
        training=training,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_clips=num_clips,
        transform=transform,
        shared_transform=shared_transform,
    )

    log_dir = pathlib.Path(log_dir) if log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        resource_log_filename = log_dir / f"resource_file_{rank}_%w.csv"
        dataset = MonitoredDataset(
            dataset=dataset,
            log_filename=str(resource_log_filename),
            log_interval=10.0,
            monitor_interval=5.0,
        )

    logger.info("SurgicalPhaseDataset dataset created")

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    if deterministic:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collator,
            sampler=dist_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0) and persistent_workers,
        )
    else:
        data_loader = NondeterministicDataLoader(
            dataset,
            collate_fn=collator,
            sampler=dist_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0) and persistent_workers,
        )
    logger.info("SurgicalPhaseDataset data loader created")

    return dataset, data_loader, dist_sampler


class SurgicalPhaseDataset(Dataset):
    """Surgical phase classification dataset from frame images."""

    def __init__(
        self,
        data_path, # Path to data_dict.yaml
        training=True,
        frames_per_clip=10,
        frame_step=1,
        num_clips=1, # For compatibility, we'll treat a sequence as a single clip
        transform=None,
        shared_transform=None,
    ):
        self.data_path = data_path
        self.training = training
        self.split = 'train' if training else 'val'
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.transform = transform
        self.shared_transform = shared_transform
        self.samples = []
        self._load_dataset()

    def _load_dataset(self):
        """Loads dataset from a YAML data dictionary."""
        logger.info(f"Loading data from YAML dictionary: {self.data_path} for split '{self.split}'")
        with open(self.data_path, 'r') as f:
            data_dict = yaml.safe_load(f)

        if self.split not in data_dict:
            logger.warning(f"Split '{self.split}' not found in {self.data_path}. No data will be loaded.")
            return

        videos_data = data_dict[self.split]
        logger.info(f"Loading {self.split} data from {len(videos_data)} videos specified in YAML...")

        base_dir = os.path.dirname(self.data_path)

        # Normalize videos_data to a list of dicts
        if isinstance(videos_data, dict):
            entries = list(videos_data.values())
        elif isinstance(videos_data, list):
            entries = videos_data
        else:
            logger.error(f"Invalid '{self.split}' section type: {type(videos_data)}. Expected list or dict.")
            return

        for video_info in entries:
            if not isinstance(video_info, dict):
                logger.warning("Encountered non-dict entry in videos list; skipping")
                continue

            # Support multiple schema variants: {frames_path, phases} or {frames, label}
            frame_paths = video_info.get('frames_path')
            if frame_paths is None:
                frame_paths = video_info.get('frames')

            labels = video_info.get('phases')
            if labels is None:
                if 'label' in video_info:
                    try:
                        label_value = int(video_info['label'])
                    except Exception:
                        label_value = 0
                    # Replicate a single label across all frames
                    labels = [label_value] * (len(frame_paths) if frame_paths else 0)
                else:
                    # Default to zeros for SSL
                    labels = [0] * (len(frame_paths) if frame_paths else 0)

            if not frame_paths or not isinstance(frame_paths, list):
                logger.warning("Skipping entry due to missing or invalid 'frames_path'/'frames'.")
                continue

            if not labels or len(frame_paths) != len(labels):
                logger.warning("Skipping entry due to missing or mismatched frames/labels.")
                continue

            # Create sequences; respect absolute paths if provided
            max_start = len(frame_paths) - (self.frames_per_clip - 1) * self.frame_step
            for i in range(max_start):
                sequence_frame_paths = []
                for k in range(self.frames_per_clip):
                    fp = frame_paths[i + k * self.frame_step]
                    fp = fp if os.path.isabs(fp) else os.path.join(base_dir, fp)
                    sequence_frame_paths.append(fp)

                # Choose label of the last frame in the sequence
                sequence_label = labels[i + (self.frames_per_clip - 1) * self.frame_step]
                self.samples.append((sequence_frame_paths, sequence_label))
        
        logger.info(f"Loaded {len(self.samples)} samples for split '{self.split}'.")

    def __getitem__(self, index):
        frame_paths, label = self.samples[index]
        
        buffer = []
        for frame_path in frame_paths:
            try:
                frame = Image.open(frame_path).convert('RGB')
                buffer.append(frame)
            except FileNotFoundError:
                logger.error(f"Frame not found at {frame_path}, skipping sample.")
                # Return the next sample to avoid crashing
                return self.__getitem__((index + 1) % len(self))
            except Exception as e:
                logger.error(f"Error loading frame {frame_path}: {e}, skipping sample.")
                return self.__getitem__((index + 1) % len(self))

        if not buffer or len(buffer) != self.frames_per_clip:
            logger.warning("Buffer is empty or incomplete, skipping.")
            return self.__getitem__((index + 1) % len(self))

        # VJEPA expects a list of tensors, one per clip.
        # We will treat our sequence as a single clip.
        if self.shared_transform is not None:
            # This transform is applied to the whole sequence (list of images)
            buffer = self.shared_transform(buffer)

        if self.transform is not None:
            # This transform is applied to the tensor of the whole sequence
            # The transform should expect a (T, H, W, C) tensor
            buffer_tensor = torch.stack([torch.tensor(np.array(img)) for img in buffer])
            transformed_buffer = self.transform(buffer_tensor)
            # The output is a single tensor for the clip
            buffer = [transformed_buffer]
        else:
            # If no further transform, just convert to tensor
            buffer_tensor = torch.stack([torch.tensor(np.array(img)) for img in buffer])
            buffer = [buffer_tensor]

        # Return clip_indices in the format expected by the collator
        # The collator expects clip_indices to be a list of arrays/lists,
        # where each element corresponds to frame indices for each clip
        # For surgical dataset, we have one clip per sample
        clip_indices = [np.arange(self.frames_per_clip)]  # Single clip with frames 0,1,2,...,63
        return buffer, label, clip_indices

    def __len__(self):
        return len(self.samples)
