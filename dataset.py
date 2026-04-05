import os
import platform
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datasets import load_dataset
from torchvision import transforms
from utils import get_captions, get_image


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


def safe_num_workers(requested: int) -> int:
    system = platform.system()
    is_apple_silicon = (system == "Darwin" and platform.machine() == "arm64")
    if system == "Windows" or is_apple_silicon:
        if requested > 0:
            print(f"  [INFO] {system} 環境のため num_workers=0 に変更します")
        return 0
    return requested


# ──────────────────────────────────────────
# キャッシュ済みデータセット（STAIR-Captions）
# ──────────────────────────────────────────

class CachedDataset(Dataset):
    """
    事前キャッシュ済みテキスト埋め込みを使うデータセット（高速）
    STAIR-Captionsのように1枚5キャプションある場合に使用
    """
    def __init__(self, split: str, config):
        self.config    = config
        self.transform = get_transforms(config.image_size, train=(split == "train"))

        cache_path = os.path.join("./caption_cache", f"{split}.npy")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"キャッシュが見つかりません: {cache_path}\n"
                f"先に python cache_embeddings.py を実行してください。"
            )

        print(f"Loading {config.dataset_name} ({split})...")
        self.data = load_dataset(config.dataset_name, "v1.2.0", split=split)
        print(f"  {len(self.data)} samples loaded.")

        self.caption_embeds = np.load(cache_path)
        print(f"  Caption cache loaded: {self.caption_embeds.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = get_image(item, self.config.image_size)
        image_tensor = self.transform(image)

        cap_offset   = random.randint(0, 4)
        embed        = self.caption_embeds[idx * 5 + cap_offset]
        embed_tensor = torch.from_numpy(embed.copy()).float()

        return {"image": image_tensor, "text_embed": embed_tensor}


# ──────────────────────────────────────────
# リアルタイムエンコードデータセット（DEJIMA / COCO等）
# ──────────────────────────────────────────

class OnTheFlyDataset(Dataset):
    """
    テキスト埋め込みをリアルタイムに計算するデータセット
    キャッシュ不要・任意のデータセットに対応
    ただし学習時はtext_encoderを渡す必要がある
    """
    def __init__(self, hf_name: str, split: str, config,
                 hf_config=None, streaming: bool = False,
                 max_samples: int = None):
        self.config    = config
        self.transform = get_transforms(config.image_size, train=(split == "train"))
        self.hf_name   = hf_name

        print(f"Loading {hf_name} ({split})...")
        kwargs = dict(split=split, streaming=streaming)
        if hf_config:
            kwargs["name"] = hf_config

        raw = load_dataset(hf_name, **kwargs)

        if streaming:
            # streamingの場合はリストに変換（max_samplesで制限）
            limit = max_samples or 500_000
            self.data = list(raw.take(limit))
        else:
            self.data = raw
            if max_samples:
                self.data = self.data.select(range(min(max_samples, len(self.data))))

        print(f"  {len(self.data)} samples loaded.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item      = self.data[idx]
        image     = get_image(item, self.config.image_size)
        image_tensor = self.transform(image)
        caps      = get_captions(item, self.hf_name)
        caption   = random.choice(caps) if caps else ""
        return {
            "image":   image_tensor,
            "caption": caption,         # テキストはそのまま返す
        }


# ──────────────────────────────────────────
# キャッシュ作成用データセット（新規追加データ用）
# ──────────────────────────────────────────

class MultiSourceCachedDataset(Dataset):
    """
    複数データセットのキャッシュを結合したデータセット
    cache_embeddings_multi.py で事前にキャッシュを作成しておく
    """
    def __init__(self, split: str, config, sources: list):
        """
        sources: [{"name": "stair", "cache": "caption_cache/stair_train.npy",
                   "hf_name": "shunk031/STAIR-Captions", ...}, ...]
        """
        self.config    = config
        self.transform = get_transforms(config.image_size, train=(split == "train"))
        self.items     = []   # (hf_dataset, idx, cache_array, n_caps)

        for src in sources:
            cache_path = src["cache"].replace("{split}", split)
            if not os.path.exists(cache_path):
                print(f"  スキップ（キャッシュなし）: {cache_path}")
                continue

            # valid_indicesファイル（画像取得成否）
            valid_path = cache_path.replace(".npy", "_valid.npy")
            valid_flags = None
            if os.path.exists(valid_path):
                valid_flags = np.load(valid_path)

            print(f"Loading {src['hf_name']} ({split})...")
            kwargs = dict(split=split)
            if src.get("hf_config"):
                kwargs["name"] = src["hf_config"]
            data   = load_dataset(src["hf_name"], **kwargs)
            embeds = np.load(cache_path)
            n_caps = src.get("n_caps", 1)

            skipped = 0
            for i in range(len(data)):
                # 画像取得失敗したサンプルはスキップ
                if valid_flags is not None and not valid_flags[i]:
                    skipped += 1
                    continue
                self.items.append((data, i, embeds, n_caps))

            msg = f"  {len(data)} samples  cache={embeds.shape}"
            if skipped > 0:
                msg += f"  skipped={skipped}"
            print(msg)

        print(f"Total: {len(self.items)} samples")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        data, i, embeds, n_caps = self.items[idx]
        item  = data[i]
        image = get_image(item, self.config.image_size)
        image_tensor = self.transform(image)

        offset       = random.randint(0, n_caps - 1)
        embed        = embeds[i * n_caps + offset]
        embed_tensor = torch.from_numpy(embed.copy()).float()

        return {"image": image_tensor, "text_embed": embed_tensor}


# ──────────────────────────────────────────
# DataLoader生成
# ──────────────────────────────────────────

def get_dataloaders(config):
    """既存のSTAIR-Captionsキャッシュを使うDataLoader（後方互換）"""
    train_dataset = CachedDataset("train",      config)
    val_dataset   = CachedDataset("validation", config)
    nw = safe_num_workers(config.num_workers)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True,  num_workers=nw, pin_memory=(nw > 0), drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,   batch_size=config.batch_size,
        shuffle=False, num_workers=nw, pin_memory=(nw > 0),
    )
    return train_loader, val_loader


def get_dataloaders_multi(config, sources_train: list, sources_val: list):
    """複数データセットのキャッシュを結合したDataLoader"""
    train_dataset = MultiSourceCachedDataset("train",      config, sources_train)
    val_dataset   = MultiSourceCachedDataset("validation", config, sources_val)
    nw = safe_num_workers(config.num_workers)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True,  num_workers=nw, pin_memory=(nw > 0), drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,   batch_size=config.batch_size,
        shuffle=False, num_workers=nw, pin_memory=(nw > 0),
    )
    return train_loader, val_loader