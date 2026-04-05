"""
複数データセットのキャプションを一括でキャッシュするスクリプト

対応データセット:
  - shunk031/STAIR-Captions  (日本語 82k)
  - MIL-UT/DEJIMA-dataset    (日本語 100万)
  - lmms-lab/COCO-Caption    (英語   33万)

使い方:
    python cache_embeddings_multi.py
    python cache_embeddings_multi.py --datasets stair coco   # 指定して実行
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from utils import get_captions, get_image

# ──────────────────────────────────────────
# データセット定義
# ──────────────────────────────────────────
DATASET_CONFIGS = {
    "stair": {
        "hf_name":   "shunk031/STAIR-Captions",
        "hf_config": "v1.2.0",
        "splits":    {"train": "train", "validation": "validation"},
        "n_caps":    5,
        "cache_dir": "./caption_cache",
        "streaming": False,
    },
    "coco": {
        "hf_name":   "lmms-lab/COCO-Caption",
        "hf_config": None,
        "splits":    {"train": "val"},   # COCOはvalのみ
        "n_caps":    5,
        "cache_dir": "./caption_cache",
        "streaming": False,
    },
    "dejima": {
        "hf_name":   "MIL-UT/DEJIMA-dataset",
        "hf_config": None,
        "splits":    {"train": "train"},
        "n_caps":    1,
        "cache_dir": "./caption_cache",
        "streaming": True,
        "max_samples": 200_000,
    },
    "gaia": {
        "hf_name":   "azavras/GAIA",
        "hf_config": None,
        "splits":    {"train": "train", "validation": "validation"},
        "n_caps":    5,
        "cache_dir": "./caption_cache",
        "streaming": False,
    },
}


def get_model_params_b(model_name):
    try:
        cfg    = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        hidden = getattr(cfg, "hidden_size", 2048)
        layers = getattr(cfg, "num_hidden_layers", 24)
        vocab  = getattr(cfg, "vocab_size", 32000)
        inter  = getattr(cfg, "intermediate_size", hidden * 4)
        return (layers * (4 * hidden**2 + 2 * hidden * inter) + vocab * hidden) / 1e9
    except Exception:
        return 0.0


def load_text_encoder(model_name, quantize_threshold_b=4.0):
    params_b = get_model_params_b(model_name)
    use_4bit = params_b > quantize_threshold_b

    if use_4bit:
        print(f"Loading {model_name} (~{params_b:.1f}B) → 4bit量子化")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModel.from_pretrained(
            model_name, quantization_config=bnb,
            device_map="auto",
        )
    else:
        print(f"Loading {model_name} (~{params_b:.1f}B) → BF16")
        model = AutoModel.from_pretrained(
            model_name, dtype=torch.bfloat16,
        ).to("cuda")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print(f"  hidden_size={model.config.hidden_size}")
    return model


def encode_and_save(model, tokenizer, captions, out_path,
                    max_length=64, batch_size=64):
    hidden_size = model.config.hidden_size
    total       = len(captions)
    out = np.lib.format.open_memmap(
        out_path, mode='w+', dtype=np.float32, shape=(total, hidden_size)
    )

    idx = 0
    for i in tqdm(range(0, total, batch_size), desc="  Encoding"):
        batch = captions[i:i + batch_size]
        enc   = tokenizer(
            batch, max_length=max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            out_model  = model(
                input_ids=enc["input_ids"].to("cuda"),
                attention_mask=enc["attention_mask"].to("cuda"),
            )
            mask   = enc["attention_mask"].unsqueeze(-1).float().to("cuda")
            pooled = (out_model.last_hidden_state.float() * mask).sum(1) \
                     / mask.sum(1).clamp(min=1e-9)

        arr = pooled.cpu().float().numpy()
        out[idx:idx + len(arr)] = arr
        idx += len(arr)

        if i == 0:
            print(f"  [確認] mean={arr.mean():.4f}  std={arr.std():.4f}"
                  f"  sample={arr[0][:3]}")

    out.flush()
    saved     = np.load(out_path)
    zero_rows = (saved.sum(axis=1) == 0).sum()
    if zero_rows > 0:
        print(f"  警告: {zero_rows}/{total} 行がゼロです")
    else:
        print(f"  OK: mean={saved.mean():.4f}  std={saved.std():.4f}")


def cache_dataset(ds_key, model, tokenizer, max_length=64):
    cfg        = DATASET_CONFIGS[ds_key]
    cache_dir  = cfg["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)

    for local_split, hf_split in cfg["splits"].items():
        out_path     = os.path.join(cache_dir, f"{ds_key}_{local_split}.npy")
        valid_path   = os.path.join(cache_dir, f"{ds_key}_{local_split}_valid.npy")

        if os.path.exists(out_path) and os.path.exists(valid_path):
            existing = np.load(out_path)
            if existing.sum() != 0:
                valid = np.load(valid_path)
                print(f"  スキップ（既存）: {out_path}  valid={valid.sum()}/{len(valid)}")
                continue
            print(f"  再作成（ゼロ検出）: {out_path}")
            os.remove(out_path)

        print(f"\nLoading {cfg['hf_name']} ({hf_split})...")
        kwargs = dict(split=hf_split, streaming=cfg.get("streaming", False))
        if cfg.get("hf_config"):
            kwargs["name"] = cfg["hf_config"]

        raw = load_dataset(cfg["hf_name"], **kwargs)

        if cfg.get("streaming"):
            limit = cfg.get("max_samples", 100_000)
            data  = list(raw.take(limit))
        else:
            data = raw

        print(f"  {len(data)} samples")

        # 画像取得可否チェック + キャプション収集
        from utils import get_image
        all_captions = []
        valid_flags  = []
        n_caps = cfg["n_caps"]
        for item in tqdm(data, desc="  Collecting"):
            _, ok = get_image(item, 224)
            valid_flags.append(ok)
            caps = get_captions(item, cfg["hf_name"])
            while len(caps) < n_caps:
                caps.append(caps[-1])
            all_captions.extend(caps[:n_caps])

        valid_arr = np.array(valid_flags, dtype=bool)
        failed    = (~valid_arr).sum()
        print(f"  画像取得: 成功 {valid_arr.sum()}/{len(valid_arr)}  失敗 {failed}")

        np.save(valid_path, valid_arr)
        print(f"  Saved valid_indices: {valid_path}")

        print(f"  Total captions: {len(all_captions)}")
        print(f"  Sample: {all_captions[:2]}")
        encode_and_save(model, tokenizer, all_captions, out_path,
                        max_length=max_length)
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+",
        default=["stair", "coco", "dejima"],
        choices=list(DATASET_CONFIGS.keys()),
        help="キャッシュするデータセット"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_text_encoder(args.model)

    for ds_key in args.datasets:
        print(f"\n{'='*50}")
        print(f"Processing: {ds_key}")
        print('='*50)
        cache_dataset(ds_key, model, tokenizer)

    print("\n全キャッシュ完了！")
    print("次は train_multi.py を実行してください。")