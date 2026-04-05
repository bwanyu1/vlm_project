import os
import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from config import Config
from model import PatchProjectionTrainer
from dataset import get_dataloaders


def train_one_epoch(model, loader, optimizer, scheduler, scaler, config, epoch):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [train]")
    for step, batch in enumerate(pbar):
        image      = batch["image"].to(config.device)
        text_embed = batch["text_embed"].to(config.device)

        with autocast("cuda", dtype=torch.bfloat16,
                      enabled=config.mixed_precision):
            loss, scale = model(image, text_embed)
            loss = loss / config.grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % config.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * config.grad_accum_steps

        if step % config.log_every == 0:
            pbar.set_postfix({
                "loss":  f"{loss.item() * config.grad_accum_steps:.4f}",
                "scale": f"{scale.item():.1f}",
                "lr":    f"{scheduler.get_last_lr()[0]:.1e}",
            })

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, config):
    model.eval()
    total_loss = 0.0

    for batch in tqdm(loader, desc="[val]"):
        image      = batch["image"].to(config.device)
        text_embed = batch["text_embed"].to(config.device)

        loss, _ = model(image, text_embed)
        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    config = Config()
    os.makedirs(config.save_dir, exist_ok=True)

    print("=" * 50)
    print("LLaVA-style VLM Training")
    print(f"  Text encoder : {config.text_model_name} [cache only]")
    print(f"  Patch size   : {config.patch_size}x{config.patch_size} "
          f"→ {(config.image_size // config.patch_size)**2} patches")
    print(f"  Batch size   : {config.batch_size} x "
          f"{config.grad_accum_steps} accum = "
          f"{config.batch_size * config.grad_accum_steps} 実効")
    print("=" * 50)

    model        = PatchProjectionTrainer(config).to(config.device)
    train_loader, val_loader = get_dataloaders(config)

    trainable = list(model.parameters())
    print(f"\nTrainable: {sum(p.numel() for p in trainable)/1e6:.1f}M\n")

    optimizer = AdamW(trainable, lr=config.lr, weight_decay=config.weight_decay)

    total_steps = (len(train_loader) // config.grad_accum_steps * config.epochs)
    scheduler   = OneCycleLR(
        optimizer,
        max_lr=config.lr,
        total_steps=total_steps,
        pct_start=config.warmup_steps / total_steps,
        anneal_strategy="cos",
    )

    scaler        = GradScaler("cuda", enabled=config.mixed_precision)
    best_val_loss = float("inf")

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, config, epoch
        )
        val_loss = evaluate(model, val_loader, config)

        print(f"\nEpoch {epoch+1}/{config.epochs}  "
              f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        ckpt = {
            "epoch":      epoch,
            "patch_proj": model.patch_proj.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "val_loss":   val_loss,
            "config":     config,
        }

        if (epoch + 1) % config.save_every == 0:
            path = os.path.join(config.save_dir, f"epoch_{epoch+1}.pt")
            torch.save(ckpt, path)
            print(f"Saved: {path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, os.path.join(config.save_dir, "best.pt"))
            print(f"  → Best model updated (val_loss={val_loss:.6f})")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()