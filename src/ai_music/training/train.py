"""Training loop for contrastive fine-tuning (Phase B).

Usage:
    python scripts/train_contrastive.py --gpu 0 --epochs 50

Trains a projection head on top of frozen MERT using triplet loss.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from ai_music import config
from ai_music.data.dataset import MusicTripletDataset
from ai_music.training.losses import TripletLoss
from ai_music.training.model import MERTEmbedder, ProjectionHead


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


def run_training(
    processed_dir: Path | None = None,
    device: str = "cpu",
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    margin: float = 0.3,
    snippet_duration: float = 10.0,
    output_dim: int = 256,
    save_dir: Path | None = None,
    seed: int = 42,
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

    dataset = MusicTripletDataset(
        processed_dir=processed_dir,
        sr=config.MERT_SR,
        snippet_duration=snippet_duration,
        augment_positive=True,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_triplets,
        drop_last=True,
    )

    criterion = TripletLoss(margin=margin)
    optimizer = torch.optim.Adam(model.projection.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Resume from checkpoint if one exists
    start_epoch = 1
    best_loss = float("inf")
    resume_path = save_dir / "checkpoint_latest.pt"
    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.projection.load_state_dict(ckpt["projection"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt["best_loss"]
        print(f"Resumed from epoch {ckpt['epoch']}, best_loss={best_loss:.4f}")

    print(f"Training for {epochs} epochs (starting at {start_epoch}), {len(dataset)} pieces, batch_size={batch_size}")
    print(f"Augmentation pipeline: {dataset.augment}")

    for epoch in range(start_epoch, epochs + 1):
        avg_loss = train_one_epoch(model, processor, loader, optimizer, criterion, device)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        status = ""
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.projection.state_dict(), save_dir / "projection_best.pt")
            status = " [best]"

        # Save full checkpoint every epoch for resume
        torch.save({
            "epoch": epoch,
            "projection": model.projection.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_loss": best_loss,
            "loss": avg_loss,
        }, resume_path)

        print(f"  Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  lr={lr_now:.2e}{status}")

    torch.save(model.projection.state_dict(), save_dir / "projection_final.pt")
    print(f"Training complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints: {save_dir}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Contrastive fine-tuning (Phase B)")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--snippet-duration", type=float, default=10.0)
    parser.add_argument("--output-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=Path, default=None, help="Checkpoint directory")
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
    )
