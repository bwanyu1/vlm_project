import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
import requests
from io import BytesIO


def get_transforms(image_size: int, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size,
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


class STAIRDataset(Dataset):
    def __init__(self, split: str, config):
        self.config    = config
        self.transform = get_transforms(config.image_size, train=(split == "train"))
        self.split     = split

        # キャッシュの確認
        cache_path = os.path.join("./caption_cache", f"{split}.npy")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"キャッシュが見つかりません: {cache_path}\n"
                f"先に python cache_embeddings.py を実行してください。"
            )

        print(f"Loading STAIR-Captions ({split})...")
        self.data = load_dataset(
            config.dataset_name, "v1.2.0", split=split
        )
        print(f"  {len(self.data)} samples loaded.")

        # キャプション埋め込みキャッシュをロード
        self.caption_embeds = np.load(cache_path)  # (N*5, D)
        print(f"  Caption cache loaded: {self.caption_embeds.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 画像取得
        image = item.get("image")
        if image is None:
            url = item.get("url") or item.get("image_url", "")
            try:
                resp  = requests.get(url, timeout=5)
                image = Image.open(BytesIO(resp.content)).convert("RGB")
            except Exception:
                image = Image.new("RGB", (self.config.image_size,
                                          self.config.image_size))
        else:
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            image = image.convert("RGB")

        image_tensor = self.transform(image)

        # キャッシュから埋め込みをランダムに1つ選ぶ（5つ中）
        cap_offset = random.randint(0, 4)
        embed = self.caption_embeds[idx * 5 + cap_offset]  # (D,)
        embed_tensor = torch.from_numpy(embed.copy()).float()

        return {
            "image":         image_tensor,    # (3, H, W)
            "text_embed":    embed_tensor,    # (D,)  ← Qwen埋め込み済み
        }


def get_dataloaders(config):
    train_dataset = STAIRDataset("train",      config)
    val_dataset   = STAIRDataset("validation", config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader