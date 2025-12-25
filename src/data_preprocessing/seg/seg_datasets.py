import random
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

NUM_CLASSES = 19
IGNORE_INDEX = 255


def random_scale(img, mask, scale_range=(0.75, 1.5)):
    s = random.uniform(*scale_range)
    h, w = img.shape[:2]
    nh, nw = int(h * s), int(w * s)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return img, mask


def random_crop(img, mask, size):
    h, w = img.shape[:2]
    if h < size or w < size:
        pad_h = max(size - h, 0)
        pad_w = max(size - w, 0)
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w,
                                 cv2.BORDER_CONSTANT, value=0)
        mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w,
                                  cv2.BORDER_CONSTANT, value=IGNORE_INDEX)
        h, w = img.shape[:2]

    y = random.randint(0, h - size)
    x = random.randint(0, w - size)
    return img[y:y+size, x:x+size], mask[y:y+size, x:x+size]


def normalize(img):
    img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img - mean) / std


class CityscapesDataset(Dataset):
    """
    EXPECTS masks from: gtFine_trainIds (0..18 + 255)
    """
    def __init__(self, img_root, mask_root, img_size=512, is_train=True):
        self.img_root = Path(img_root)
        self.mask_root = Path(mask_root)
        self.img_size = img_size
        self.is_train = is_train

        self.samples = []
        for img in self.img_root.rglob("*_leftImg8bit.png"):
            city = img.parent.name
            base = img.stem.replace("_leftImg8bit", "")
            mask = self.mask_root / city / f"{base}_gtFine_trainIds.png"
            if mask.exists():
                self.samples.append((img, mask))

        if not self.samples:
            raise RuntimeError("No Cityscapes samples found")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))  # trainIds already

        # ✅ sanity clamp: keep only 0..18 and 255
        # (if anything else exists, it becomes ignore)
        mask = mask.astype(np.int32)
        bad = (mask != IGNORE_INDEX) & ((mask < 0) | (mask >= NUM_CLASSES))
        if bad.any():
            mask[bad] = IGNORE_INDEX
        mask = mask.astype(np.uint8)

        if self.is_train:
            img, mask = random_scale(img, mask)
            img, mask = random_crop(img, mask, self.img_size)
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
                mask = np.ascontiguousarray(mask[:, ::-1])

        img = normalize(img)
        mask = torch.from_numpy(mask).long()
        return img, mask


def make_loaders_cs(train_img, train_mask, val_img, val_mask,
                    img_size=512, batch=4, workers=2):
    train_ds = CityscapesDataset(train_img, train_mask, img_size, True)
    val_ds   = CityscapesDataset(val_img, val_mask, img_size, False)

    train_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True,
        num_workers=workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch, shuffle=False,
        num_workers=max(1, workers // 2), pin_memory=True
    )
    return train_loader, val_loader
