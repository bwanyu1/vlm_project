"""
パイプライン全体を少量データで高速テストする
所要時間: 約1分

確認項目:
  1. Qwenが正常に埋め込みを返す
  2. キャプションフィールドが正しく取得できる
  3. キャッシュの保存・読み込みが正常（ゼロでない）
  4. PatchProjectionのforward/backwardが正常
  5. lossが変化する（勾配が流れている）
"""

import os
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from torchvision import transforms
from utils import get_captions

MODEL_NAME   = "Qwen/Qwen2.5-1.5B"
DATASET_NAME = "shunk031/STAIR-Captions"
N_SAMPLES    = 32
BATCH_SIZE   = 4
IMAGE_SIZE   = 224
PATCH_SIZE   = 16
N_STEPS      = 5
DEVICE       = "cuda"

if __name__ == "__main__":
    print("=" * 50)
    print("Pipeline Quick Test")
    print("=" * 50)

    # ──────────────────────────────
    # Step 1: Qwen 埋め込みテスト
    # ──────────────────────────────
    print("\n[1/5] Qwen 埋め込みテスト...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, trust_remote_code=True
    ).to(DEVICE).eval()

    test_texts = ["猫が座っています。", "公園で犬が走っている。"]
    enc = tokenizer(test_texts, padding=True, truncation=True,
                    max_length=64, return_tensors="pt")
    with torch.no_grad():
        out    = model(input_ids=enc["input_ids"].to(DEVICE),
                       attention_mask=enc["attention_mask"].to(DEVICE))
        mask   = enc["attention_mask"].unsqueeze(-1).float().to(DEVICE)
        pooled = (out.last_hidden_state.float() * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    assert pooled.abs().mean() > 0.01, "NG: Qwen出力がゼロです"
    print(f"  OK  mean={pooled.mean():.4f}  std={pooled.std():.4f}")
    hidden_size = model.config.hidden_size

    # ──────────────────────────────
    # Step 2: キャプションフィールド確認
    # ──────────────────────────────
    print(f"\n[2/5] キャプションフィールド確認...")
    data = load_dataset(DATASET_NAME, "v1.2.0", split="train")
    item = data[0]
    caps = get_captions(item)
    assert len(caps) > 0 and caps[0] != "", f"NG: キャプションが空です: {caps}"
    print(f"  OK  取得できたキャプション数: {len(caps)}")
    print(f"  sample: {caps[0][:30]}...")

    # ──────────────────────────────
    # Step 3: ミニキャッシュ作成・検証
    # ──────────────────────────────
    print(f"\n[3/5] ミニキャッシュ作成 ({N_SAMPLES}サンプル)...")
    all_captions = []
    for i in range(N_SAMPLES):
        caps = get_captions(data[i])
        while len(caps) < 5:
            caps.append(caps[-1])
        all_captions.extend(caps[:5])

    assert all(c != "" for c in all_captions), "NG: 空キャプションが含まれています"
    print(f"  キャプション収集 OK: {len(all_captions)}件")

    enc = tokenizer(all_captions, padding="max_length", truncation=True,
                    max_length=64, return_tensors="pt")
    with torch.no_grad():
        out    = model(input_ids=enc["input_ids"].to(DEVICE),
                       attention_mask=enc["attention_mask"].to(DEVICE))
        mask   = enc["attention_mask"].unsqueeze(-1).float().to(DEVICE)
        embeds = (out.last_hidden_state.float() * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    embeds_np = embeds.cpu().numpy()
    assert embeds_np.sum() != 0, "NG: 埋め込みがゼロです"

    os.makedirs("./caption_cache_test", exist_ok=True)
    np.save("./caption_cache_test/mini.npy", embeds_np)
    loaded = np.load("./caption_cache_test/mini.npy")
    assert loaded.sum() != 0, "NG: ロード後にゼロになりました"
    print(f"  OK  shape={loaded.shape}  mean={loaded.mean():.4f}  std={loaded.std():.4f}")

    del model
    torch.cuda.empty_cache()

    # ──────────────────────────────
    # Step 4: PatchProjection forward
    # ──────────────────────────────
    print(f"\n[4/5] PatchProjection forward テスト...")
    patch_dim = 3 * PATCH_SIZE * PATCH_SIZE

    proj = torch.nn.Sequential(
        torch.nn.Linear(patch_dim, hidden_size),
        torch.nn.GELU(),
        torch.nn.Linear(hidden_size, hidden_size),
    ).to(DEVICE)

    dummy = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    p = PATCH_SIZE
    x = dummy.unfold(2, p, p).unfold(3, p, p)
    x = x.contiguous().view(BATCH_SIZE, 3, -1, p, p)
    x = x.permute(0, 2, 1, 3, 4).contiguous().view(BATCH_SIZE, -1, 3 * p * p)
    out = proj(x).mean(dim=1)

    assert out.shape == (BATCH_SIZE, hidden_size), f"NG: shape={out.shape}"
    assert out.abs().mean() > 0, "NG: PatchProjection出力がゼロです"
    print(f"  OK  output shape={out.shape}  mean={out.mean():.4f}")

    # ──────────────────────────────
    # Step 5: 学習ループテスト
    # ──────────────────────────────
    print(f"\n[5/5] 学習ループテスト ({N_STEPS}ステップ)...")
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    logit_scale = torch.nn.Parameter(torch.ones([]) * 2.659)
    optimizer   = torch.optim.AdamW(
        list(proj.parameters()) + [logit_scale], lr=1e-4
    )

    losses = []
    for step in range(N_STEPS):
        idx       = torch.randint(0, N_SAMPLES, (BATCH_SIZE,))
        img_batch = torch.stack([
            transform(data[int(i)]["image"].convert("RGB")) for i in idx
        ]).to(DEVICE)
        txt_batch = torch.from_numpy(
            loaded[idx.numpy() * 5]   # 各サンプルの0番目キャプション
        ).float().to(DEVICE)

        x = img_batch.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(BATCH_SIZE, 3, -1, p, p)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(BATCH_SIZE, -1, 3*p*p)
        img_emb = F.normalize(proj(x).mean(dim=1), dim=-1)
        txt_emb = F.normalize(txt_batch, dim=-1)

        scale  = logit_scale.exp().clamp(max=100)
        logits = scale * img_emb @ txt_emb.T
        labels = torch.arange(BATCH_SIZE, device=DEVICE)
        loss   = (F.cross_entropy(logits, labels) +
                  F.cross_entropy(logits.T, labels)) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"  step {step+1}/{N_STEPS}  loss={loss.item():.4f}  scale={scale.item():.2f}")

    loss_changed = abs(losses[-1] - losses[0]) > 1e-6
    status = "OK ✓" if loss_changed else "NG 変化なし"
    print(f"\n  loss変化: {losses[0]:.6f} → {losses[-1]:.6f}  {status}")
    assert loss_changed, "NG: lossが全く変化していません"

    import shutil
    shutil.rmtree("./caption_cache_test", ignore_errors=True)

    print("\n" + "=" * 50)
    print("全テスト通過！python cache_embeddings.py を実行してください。")
    print("=" * 50)