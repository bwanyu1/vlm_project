"""
キャプションをQwenで事前にベクトル化してディスクに保存するスクリプト
1回だけ実行すれば OK。学習時はキャッシュから読むだけになる。

使い方:
    python cache_embeddings.py
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, AutoConfig

from config import Config


def get_model_params_b(model_name: str) -> float:
    try:
        cfg    = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        hidden = getattr(cfg, "hidden_size", 2048)
        layers = getattr(cfg, "num_hidden_layers", 24)
        vocab  = getattr(cfg, "vocab_size", 32000)
        inter  = getattr(cfg, "intermediate_size", hidden * 4)
        return (layers * (4 * hidden**2 + 2 * hidden * inter) + vocab * hidden) / 1e9
    except Exception:
        return 0.0


def load_text_encoder(config):
    params_b = get_model_params_b(config.text_model_name)
    use_4bit = params_b > config.quantize_threshold_b

    if use_4bit:
        print(f"Loading {config.text_model_name} (~{params_b:.1f}B) → 4bit量子化")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModel.from_pretrained(
            config.text_model_name,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        print(f"Loading {config.text_model_name} (~{params_b:.1f}B) → BF16")
        model = AutoModel.from_pretrained(
            config.text_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to("cuda")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model


@torch.no_grad()
def encode_captions(model, tokenizer, captions, config, batch_size=64):
    """キャプションリストをバッチ処理してベクトル化"""
    all_embeds = []

    for i in tqdm(range(0, len(captions), batch_size), desc="Encoding"):
        batch = captions[i:i + batch_size]
        enc = tokenizer(
            batch,
            max_length=config.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to("cuda")
        attention_mask = enc["attention_mask"].to("cuda")

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state          # (B, L, D)
        mask   = attention_mask.unsqueeze(-1).float()
        pooled = (last_hidden.float() * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        all_embeds.append(pooled.cpu().float().numpy())

    return np.concatenate(all_embeds, axis=0)   # (N, D)


def cache_split(split, model, tokenizer, config, cache_dir):
    out_path = os.path.join(cache_dir, f"{split}.npy")
    if os.path.exists(out_path):
        print(f"  既存キャッシュをスキップ: {out_path}")
        return

    print(f"\nLoading STAIR-Captions ({split})...")
    data = load_dataset(config.dataset_name, "v1.2.0", split=split)
    print(f"  {len(data)} samples")

    # 各サンプルのキャプションを全て結合（5文 × N枚）
    # インデックスiのキャプションはembeds[i*5 : i*5+5]
    # 学習時はランダムに1つ選ぶ
    all_captions = []
    for item in tqdm(data, desc="Collecting captions"):
        caps = item.get("captions") or item.get("caption") or [""]
        if not isinstance(caps, list):
            caps = [str(caps)]
        # 5つに満たない場合は最後のもので埋める
        while len(caps) < 5:
            caps.append(caps[-1])
        all_captions.extend(caps[:5])

    print(f"  Total captions to encode: {len(all_captions)}")
    embeds = encode_captions(model, tokenizer, all_captions, config)

    np.save(out_path, embeds)
    print(f"  Saved: {out_path}  shape={embeds.shape}")


def main():
    config    = Config()
    cache_dir = "./caption_cache"
    os.makedirs(cache_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        config.text_model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_text_encoder(config)

    cache_split("train",      model, tokenizer, config, cache_dir)
    cache_split("validation", model, tokenizer, config, cache_dir)

    print("\nキャッシュ完了！次は python train.py を実行してください。")


if __name__ == "__main__":
    main()