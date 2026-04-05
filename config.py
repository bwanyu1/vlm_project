from dataclasses import dataclass

@dataclass
class Config:
    # テキスト/ベースLLM
    # "Qwen/Qwen2.5-1.5B"  軽量・推奨
    # "Qwen/Qwen3-9B"       高精度・4bit量子化自動適用
    text_model_name: str = "Qwen/Qwen2.5-1.5B"

    # 画像設定
    image_size: int = 224
    patch_size: int = 16   # 224/16 = 196パッチ

    # 学習
    batch_size: int = 4
    grad_accum_steps: int = 8   # 実効バッチサイズ = 32
    epochs: int = 10
    lr: float = 1e-4            # Projectionのみ学習なので大きめでOK
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_new_tokens: int = 64    # 生成するキャプションの最大長
    temperature: float = 0.07   # InfoNCE温度パラメータ

    # データ
    dataset_name: str = "shunk031/STAIR-Captions"
    max_text_length: int = 64
    num_workers: int = 2

    # 保存
    save_dir: str = "./checkpoints"
    save_every: int = 1
    log_every: int = 50

    # デバイス
    device: str = "cuda"
    mixed_precision: bool = True

    # 量子化しきい値（パラメータ数B単位）
    quantize_threshold_b: float = 4.0