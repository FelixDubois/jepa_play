"""Boucle d'entraînement JEPA : supervisée, stable, reprenable.

Conçue pour le Colab gratuit : AMP sur GPU, checkpoint à chaque epoch sur
Drive, reprise automatique après déconnexion.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import WindowDataset
from .model import JEPA


def _device(device: str | None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def train_jepa(episodes, out_dir, epochs: int = 10, k: int = 8,
               batch_size: int = 256, lr: float = 3e-4, tau: float = 0.996,
               device: str | None = None, resume: bool = True,
               num_workers: int = 2):
    dev = _device(device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "jepa.pt"

    model = JEPA().to(dev)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    history: list[dict] = []
    start_epoch = 0
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=True)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        history = ckpt["history"]
        start_epoch = ckpt["epoch"]
        print(f"reprise du checkpoint : epoch {start_epoch}")

    ds = WindowDataset(episodes, k=k)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, drop_last=True)
    if len(dl) == 0:
        raise ValueError(
            f"batch_size={batch_size} > {len(ds)} fenêtres disponibles : "
            "réduire batch_size ou collecter plus de données")
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    model.train()
    for epoch in range(start_epoch, epochs):
        agg = {"loss": 0.0, "pred_mse": 0.0, "copy_mse": 0.0, "latent_std": 0.0}
        n_batches = 0
        for batch in dl:
            frames = batch["frames"].to(dev, non_blocking=True)
            actions = batch["actions"].to(dev, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss, metrics = model.loss(frames, actions)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            model.update_target(tau)
            agg["loss"] += loss.item()
            for key in ("pred_mse", "copy_mse", "latent_std"):
                agg[key] += metrics[key]
            n_batches += 1
        row = {"epoch": epoch + 1,
               **{key: val / max(n_batches, 1) for key, val in agg.items()}}
        history.append(row)
        if row["latent_std"] < 0.05:
            print("⚠ variance latente faible — collapse possible "
                  f"(latent_std={row['latent_std']:.4f})")
        print(f"epoch {row['epoch']}/{epochs}  loss={row['loss']:.4f}  "
              f"pred={row['pred_mse']:.4f}  copy={row['copy_mse']:.4f}  "
              f"std={row['latent_std']:.3f}")
        torch.save({"model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "epoch": epoch + 1,
                    "history": history}, ckpt_path)
    return model, history


def load_jepa(ckpt_path, device: str | None = None) -> JEPA:
    dev = _device(device)
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=True)
    model = JEPA().to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model
