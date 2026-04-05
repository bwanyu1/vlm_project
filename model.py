import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    AutoConfig, BitsAndBytesConfig
)


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


# ──────────────────────────────────────────
# Patch Projection（学習対象）
# ──────────────────────────────────────────

class PatchProjection(nn.Module):
    """
    画像パッチ → Qwen埋め込み空間への写像

    入力: (B, 3, H, W)
    出力: (B, num_patches, qwen_hidden)
    """
    def __init__(self, patch_size: int, qwen_hidden: int):
        super().__init__()
        patch_dim = 3 * patch_size * patch_size  # 3*16*16 = 768

        # LLaVA-1.5式 2層MLP
        self.proj = nn.Sequential(
            nn.Linear(patch_dim, qwen_hidden),
            nn.GELU(),
            nn.Linear(qwen_hidden, qwen_hidden),
        )
        self.patch_size  = patch_size
        self.qwen_hidden = qwen_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)    # (B, C, H/p, W/p, p, p)
        x = x.contiguous().view(B, C, -1, p, p)   # (B, C, N, p, p)
        x = x.permute(0, 2, 1, 3, 4)              # (B, N, C, p, p)
        x = x.contiguous().view(B, -1, C * p * p) # (B, N, patch_dim)
        return self.proj(x)                        # (B, N, qwen_hidden)


# ──────────────────────────────────────────
# 学習用モデル（キャッシュ対応・Qwen不要）
# ──────────────────────────────────────────

class PatchProjectionTrainer(nn.Module):
    """
    学習時はQwenを使わない。
    キャッシュ済みテキスト埋め込みとPatchProjectionの出力を
    コサイン類似度で近づける。

    学習対象: PatchProjectionのみ（約6M params）
    """
    def __init__(self, config):
        super().__init__()

        # Qwenのhidden_sizeだけ取得（モデル本体は不要）
        cfg = AutoConfig.from_pretrained(
            config.text_model_name, trust_remote_code=True
        )
        qwen_hidden = cfg.hidden_size
        print(f"Qwen hidden_size: {qwen_hidden} (config only, no model load)")

        self.patch_proj = PatchProjection(config.patch_size, qwen_hidden)

        # 温度パラメータ（学習対象）
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.tensor(1.0 / config.temperature).log()
        )

        print(f"PatchProjection params: "
              f"{sum(p.numel() for p in self.patch_proj.parameters())/1e6:.1f}M")

    def forward(self, image: torch.Tensor, text_embed: torch.Tensor):
        """
        image:      (B, 3, H, W)
        text_embed: (B, D)  ← キャッシュ済みQwen埋め込み

        Returns: InfoNCE loss
        """
        # 画像パッチ埋め込み → 平均プーリングで1ベクトルに
        patch_embed = self.patch_proj(
            image.to(self.patch_proj.proj[0].weight.dtype)
        )                                          # (B, N, D)
        image_embed = patch_embed.mean(dim=1)      # (B, D)

        # L2正規化
        image_embed = F.normalize(image_embed.float(), dim=-1)
        text_embed  = F.normalize(text_embed.float(),  dim=-1)

        # InfoNCE Loss
        scale  = self.logit_scale.exp().clamp(max=100)
        logits = scale * image_embed @ text_embed.T   # (B, B)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_i = F.cross_entropy(logits,   labels)
        loss_t = F.cross_entropy(logits.T, labels)
        return (loss_i + loss_t) / 2, scale


# ──────────────────────────────────────────
# 推論用モデル（Qwen込みのフルモデル）
# ──────────────────────────────────────────

class LLaVAModel(nn.Module):
    """
    推論時: 学習済みPatchProjectionをQwenに繋いで画像キャプション生成
    """
    def __init__(self, config, patch_proj_state_dict=None):
        super().__init__()

        params_b = get_model_params_b(config.text_model_name)
        use_4bit = params_b > config.quantize_threshold_b

        if use_4bit:
            print(f"Loading {config.text_model_name} → 4bit量子化")
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                config.text_model_name,
                quantization_config=bnb,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                config.text_model_name,
                dtype=torch.bfloat16,
                trust_remote_code=True,
            )

        for param in self.llm.parameters():
            param.requires_grad = False

        qwen_hidden = self.llm.config.hidden_size
        self.patch_proj = PatchProjection(config.patch_size, qwen_hidden)

        if patch_proj_state_dict is not None:
            self.patch_proj.load_state_dict(patch_proj_state_dict)
            print("PatchProjection weights loaded.")

        # PatchProjectionをQwenと同じdtype(bfloat16)に統一
        llm_dtype = next(self.llm.parameters()).dtype
        self.patch_proj = self.patch_proj.to(dtype=llm_dtype)

    @torch.no_grad()
    def generate(self, image: torch.Tensor, tokenizer,
                 prompt: str = "この画像を説明してください。",
                 max_new_tokens: int = 64):
        self.eval()

        # Qwenのdtypeに合わせる（bfloat16）
        llm_dtype = next(self.llm.parameters()).dtype
        device    = next(self.llm.parameters()).device

        image_embeds = self.patch_proj(
            image.unsqueeze(0).to(device=device, dtype=llm_dtype)
        )                                          # (1, N, D)

        prompt_ids = tokenizer(
            prompt, return_tensors="pt"
        ).input_ids.to(device)
        prompt_embeds = self.llm.get_input_embeddings()(
            prompt_ids
        ).to(dtype=llm_dtype)

        inputs_embeds = torch.cat([image_embeds, prompt_embeds], dim=1)

        # attention_maskを明示的に渡す
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=device
        )

        out = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(out[0], skip_special_tokens=True)


def get_tokenizer(config):
    tok = AutoTokenizer.from_pretrained(
        config.text_model_name, trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok