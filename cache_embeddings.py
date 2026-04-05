import os
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from utils import get_captions


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
            device_map="auto", trust_remote_code=True,
        )
    else:
        print(f"Loading {model_name} (~{params_b:.1f}B) → BF16")
        model = AutoModel.from_pretrained(
            model_name, dtype=torch.bfloat16, trust_remote_code=True,
        ).to("cuda")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print(f"  hidden_size={model.config.hidden_size}")
    return model


def encode_and_save(model, tokenizer, captions, out_path, max_length=64, batch_size=64):
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
        input_ids      = enc["input_ids"].to("cuda")
        attention_mask = enc["attention_mask"].to("cuda")

        with torch.no_grad():
            outputs     = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state
            mask        = attention_mask.unsqueeze(-1).float()
            pooled      = (last_hidden.float() * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        arr = pooled.cpu().float().numpy()
        out[idx:idx + len(arr)] = arr
        idx += len(arr)

        if i == 0:
            print(f"  [確認] batch[0] mean={arr.mean():.4f}  std={arr.std():.4f}  "
                  f"sample={arr[0][:3]}")

    out.flush()

    saved     = np.load(out_path)
    zero_rows = (saved.sum(axis=1) == 0).sum()
    if zero_rows > 0:
        print(f"  警告: {zero_rows}/{total} 行がゼロです")
    else:
        print(f"  OK: mean={saved.mean():.4f}  std={saved.std():.4f}")


def cache_split(split, model, tokenizer, dataset_name, cache_dir, max_length=64):
    out_path = os.path.join(cache_dir, f"{split}.npy")

    if os.path.exists(out_path):
        existing  = np.load(out_path)
        if existing.sum() != 0:
            print(f"  既存キャッシュをスキップ: {out_path}")
            return
        print(f"  既存キャッシュがゼロのため再作成")
        os.remove(out_path)

    print(f"\nLoading {dataset_name} ({split})...")
    data = load_dataset(dataset_name, "v1.2.0", split=split)
    print(f"  {len(data)} samples")

    all_captions = []
    for item in tqdm(data, desc="  Collecting captions"):
        caps = get_captions(item)
        while len(caps) < 5:
            caps.append(caps[-1])
        all_captions.extend(caps[:5])

    print(f"  Total captions: {len(all_captions)}")
    print(f"  Sample: {all_captions[:2]}")  # 確認用
    encode_and_save(model, tokenizer, all_captions, out_path, max_length=max_length)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    model_name   = "Qwen/Qwen2.5-1.5B"
    dataset_name = "shunk031/STAIR-Captions"
    cache_dir    = "./caption_cache"
    max_length   = 64

    os.makedirs(cache_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_text_encoder(model_name)

    cache_split("train",      model, tokenizer, dataset_name, cache_dir, max_length)
    cache_split("validation", model, tokenizer, dataset_name, cache_dir, max_length)

    print("\nキャッシュ完了！次は python train.py を実行してください。")