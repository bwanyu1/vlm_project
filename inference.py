"""
学習済みモデルで画像からキャプションを生成する推論スクリプト
"""

import torch
from PIL import Image
from torchvision import transforms

from config import Config
from model import LLaVAModel, get_tokenizer

# ──────────────────────────────
# ここを編集してください
# ──────────────────────────────
IMAGE_PATH  = "./images.jpg"                  # 画像パス
PROMPT      = "この画像を説明してください。"   # プロンプト
CHECKPOINT  = "./checkpoints/epoch_2.pt"         # チェックポイント
MAX_TOKENS  = 64                              # 生成する最大トークン数
# ──────────────────────────────


def load_image(path: str, image_size: int) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize(image_size,
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(Image.open(path).convert("RGB"))


if __name__ == "__main__":
    config    = Config()
    tokenizer = get_tokenizer(config)

    print(f"Loading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    print(f"  epoch={ckpt['epoch']+1}  val_loss={ckpt['val_loss']:.6f}")

    print(f"Loading model...")
    model = LLaVAModel(config, patch_proj_state_dict=ckpt["patch_proj"])

    llm_dtype        = next(model.llm.parameters()).dtype
    model.llm        = model.llm.to(config.device)
    model.patch_proj = model.patch_proj.to(device=config.device, dtype=llm_dtype)
    model.eval()

    image = load_image(IMAGE_PATH, config.image_size).to(
        device=config.device, dtype=llm_dtype
    )

    print(f"Image:      {IMAGE_PATH}")
    print(f"Prompt:     {PROMPT}")
    print(f"Checkpoint: epoch={ckpt['epoch']+1}  val_loss={ckpt['val_loss']:.4f}")
    print()

    with torch.no_grad():
        # patch_projで画像埋め込み生成
        patch_embed   = model.patch_proj(image.unsqueeze(0))       # (1, N, D)

        # プロンプトのembedding
        prompt_ids    = tokenizer(PROMPT, return_tensors="pt").input_ids.to(config.device)
        prompt_embed  = model.llm.get_input_embeddings()(prompt_ids).to(llm_dtype)

        # 結合
        inputs_embeds = torch.cat([patch_embed, prompt_embed], dim=1)

        # attention_maskを明示的に設定
        attention_mask = torch.ones(
            inputs_embeds.shape[:2],
            dtype=torch.long,
            device=config.device
        )

        out = model.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=MAX_TOKENS,
            do_sample=True,          # サンプリング有効
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.3,  # 数字の繰り返しを抑制
            pad_token_id=tokenizer.eos_token_id,
        )

    caption = tokenizer.decode(out[0], skip_special_tokens=True)
    print("=== 生成結果 ===")
    print(caption)