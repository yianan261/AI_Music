"""Training loop for contrastive fine-tuning (Phase B).

Usage:
    python scripts/train_contrastive.py --gpu 0 --epochs 50

Trains a projection head on top of frozen MERT using triplet loss.
"""

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from ai_music import config
from ai_music.src_data.dataset import MusicTripletDataset
from ai_music.training.losses import TripletLoss
from ai_music.training.model import MERTEmbedder, ProjectionHead


def split_piece_paths(
    piece_paths: list[Path],
    val_fraction: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    """Piece-level train/val split (no snippet leakage).

    Both splits need >=2 pieces so triplet negatives stay in-domain. If that is
    impossible (e.g. fewer than 4 pieces total), returns (all, []) and training
    uses train loss only for best checkpoint.
    """
    if len(piece_paths) < 2:
        raise ValueError(f"Need >=2 WAV files for training, found {len(piece_paths)}")
    if val_fraction <= 0:
        return piece_paths, []
    n = len(piece_paths)
    if n < 4:
        return piece_paths, []
    n_val = max(2, int(round(n * val_fraction)))
    n_val = min(n_val, n - 2)
    if n_val < 2 or n - n_val < 2:
        return piece_paths, []
    rng = random.Random(seed)
    shuffled = piece_paths.copy()
    rng.shuffle(shuffled)
    val_paths = sorted(shuffled[:n_val])
    train_paths = sorted(shuffled[n_val:])
    return train_paths, val_paths


def collate_triplets(batch):
    """Collate (anchor, positive, negative, label) tuples."""
    anchors, positives, negatives, labels = zip(*batch)
    return (
        torch.stack(anchors),
        torch.stack(positives),
        torch.stack(negatives),
        torch.tensor(labels, dtype=torch.long),
    )


def train_one_epoch(model, processor, loader, optimizer, criterion, device):
    model.train()
    model.backbone.eval()  # keep BN/dropout in eval for frozen backbone
    total_loss = 0.0
    n_batches = 0

    for anchors, positives, negatives, labels in loader:
        anchors = anchors.to(device)
        positives = positives.to(device)
        negatives = negatives.to(device)

        z_a = model(anchors)
        z_p = model(positives)
        z_n = model(negatives)

        loss = criterion(z_a, z_p, z_n)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for anchors, positives, negatives, labels in loader:
            anchors = anchors.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)
            z_a = model(anchors)
            z_p = model(positives)
            z_n = model(negatives)
            loss = criterion(z_a, z_p, z_n)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def run_training(
    processed_dir: Path | None = None,
    device: str = "cpu",
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    margin: float = 0.3,
    snippet_duration: float = 20.0,
    output_dim: int = 256,
    save_dir: Path | None = None,
    seed: int = 42,
    val_fraction: float = 0.15,
):
    processed_dir = processed_dir or config.PROCESSED_24K_DIR
    save_dir = save_dir or config.PROJECT_ROOT / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading MERT backbone on {device}...")
    from ai_music.retrieval.mert import load_mert
    mert_model, processor, device = load_mert(device=device)

    hidden_dim = mert_model.config.hidden_size  # typically 768
    head = ProjectionHead(input_dim=hidden_dim, output_dim=output_dim)
    model = MERTEmbedder(mert_model, processor, head, freeze_backbone=True).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    all_piece_paths = sorted(processed_dir.glob("*.wav"))
    train_paths, val_paths = split_piece_paths(all_piece_paths, val_fraction, seed)

    train_dataset = MusicTripletDataset(
        processed_dir=processed_dir,
        sr=config.MERT_SR,
        snippet_duration=snippet_duration,
        augment_positive=True,
        seed=seed,
        piece_paths=train_paths,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_triplets,
        drop_last=True,
    )

    use_val = len(val_paths) >= 2
    val_loader = None
    if use_val:
        val_dataset = MusicTripletDataset(
            processed_dir=processed_dir,
            sr=config.MERT_SR,
            snippet_duration=snippet_duration,
            augment_positive=False,
            seed=seed + 1,
            piece_paths=val_paths,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_triplets,
            drop_last=False,
        )

    criterion = TripletLoss(margin=margin)
    optimizer = torch.optim.Adam(model.projection.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Resume from checkpoint if one exists
    start_epoch = 1
    best_metric = float("inf")
    resume_path = save_dir / "checkpoint_latest.pt"
    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.projection.load_state_dict(ckpt["projection"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        # Prefer best_val_loss; fall back to legacy best_loss
        best_metric = ckpt.get("best_val_loss", ckpt.get("best_loss", float("inf")))
        print(f"Resumed from epoch {ckpt['epoch']}, best_metric={best_metric:.4f}")

    if use_val:
        print(
            f"Train/val: {len(train_paths)} / {len(val_paths)} pieces "
            f"(val_fraction={val_fraction})"
        )
    else:
        print(
            f"Training on all {len(train_paths)} pieces "
            "(val disabled: use --val-fraction > 0 and >=4 pieces for train/val split)"
        )

    print(f"Training for {epochs} epochs (starting at {start_epoch}), batch_size={batch_size}")
    print(f"Snippet duration: {snippet_duration:.1f}s (recommended vanilla runs: 20s and 30s)")
    print(f"Augmentation pipeline: {train_dataset.augment}")

    for epoch in range(start_epoch, epochs + 1):
        avg_train = train_one_epoch(model, processor, train_loader, optimizer, criterion, device)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        if use_val and val_loader is not None:
            avg_val = evaluate_one_epoch(model, val_loader, criterion, device)
            metric = avg_val
            status = f"  train={avg_train:.4f}  val={avg_val:.4f}"
        else:
            avg_val = None
            metric = avg_train
            status = f"  train={avg_train:.4f}"

        best_str = ""
        if metric < best_metric:
            best_metric = metric
            torch.save(model.projection.state_dict(), save_dir / "projection_best.pt")
            best_str = " [best]"

        # Save full checkpoint every epoch for resume
        ckpt_dict = {
            "epoch": epoch,
            "projection": model.projection.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_loss": best_metric,
            "best_val_loss": best_metric if use_val else None,
            "train_loss": avg_train,
            "loss": avg_train,
            "val_loss": avg_val,
        }
        torch.save(ckpt_dict, resume_path)

        print(f"  Epoch {epoch:3d}/{epochs}{status}  lr={lr_now:.2e}{best_str}")

    torch.save(model.projection.state_dict(), save_dir / "projection_final.pt")
    metric_name = "val" if use_val else "train"
    print(f"Training complete. Best {metric_name} loss: {best_metric:.4f}")
    print(f"Checkpoints: {save_dir}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Contrastive fine-tuning (Phase B)")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument(
        "--snippet-duration",
        type=float,
        default=30.0,
        help="Seconds per training snippet (default 30; recommended vanilla sweeps: 20s and 30s)",
    )
    parser.add_argument("--output-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=Path, default=None, help="Checkpoint directory")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of pieces for validation (0 disables val; best checkpoint uses val loss when val is used)",
    )
    args = parser.parse_args()

    from ai_music.utils.device import select_device
    device = select_device(args.gpu)

    run_training(
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        margin=args.margin,
        snippet_duration=args.snippet_duration,
        output_dim=args.output_dim,
        save_dir=args.save_dir,
        seed=args.seed,
        val_fraction=args.val_fraction,
    )
