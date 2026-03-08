import os
import logging
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from PIL import Image

from .extended import ExtendedVisionDataset

logger = logging.getLogger("dinov2")

class EndoscopyDataset(ExtendedVisionDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        logger.info(f"Loading endoscopy dataset from {root}")
        
        # Find all image files
        self.samples = []
        for ext in ['.jpg', '.jpeg', '.png']:
            self.samples.extend(list(Path(root).rglob(f"*{ext}")))
            
        logger.info(f"Found {len(self.samples)} images in endoscopy dataset")
        
    def __getitem__(self, index: int) -> Tuple[any, any]:
        img_path = self.samples[index]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
            
        # For self-supervised learning, return the image with a dummy target
        target = 0  # Dummy target for self-supervised learning
        
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def get_targets(self):
        """Return a list of targets for compatibility with evaluation"""
        return [0] * len(self.samples)