"""
train.py
--------
Training loop for AVITM with KL annealing.

Key features:
    - KL annealing: beta increases linearly from 0 to 1 over warmup epochs
    - Logs train loss, recon loss, KL loss per epoch
    - Saves best model checkpoint based on test loss
    - Saves loss history for plotting in notebook
"""

import torch
import torch.optim as optim
import numpy as np
import json
from pathlib import Path

from data import get_dataloaders, load_vocab
from model import AVITM, compute_loss


# ── Hyperparameters ────────────────────────────────────────────────────────
N_TOPICS    = 50
HIDDEN_SIZE = 100
DROPOUT     = 0.2
LR          = 2e-3
N_EPOCHS    = 100
WARMUP      = 20       # epochs to anneal KL from 0 to 1
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── KL annealing schedule ──────────────────────────────────────────────────
def get_kl_weight(epoch, warmup=WARMUP):
    """
    Linear annealing: beta goes from 0 to 1 over `warmup` epochs.
    After warmup, beta = 1 (standard ELBO).

    Why: At the start of training, the KL term can dominate and push
    the posterior toward the prior before the decoder has learned anything
    useful. Annealing gives the reconstruction term time to stabilize first.
    """
    return min(1.0, epoch / warmup)


# ── Training loop ──────────────────────────────────────────────────────────
def train(
    n_topics=N_TOPICS,
    hidden_size=HIDDEN_SIZE,
    dropout=DROPOUT,
    lr=LR,
    n_epochs=N_EPOCHS,
    warmup=WARMUP,
    device=DEVICE,
    save_dir="checkpoints",
):
    """
    Full training loop for AVITM.

    Parameters
    ----------
    n_topics    : int   — number of latent topics
    hidden_size : int   — encoder hidden layer size
    dropout     : float — encoder dropout rate
    lr          : float — Adam learning rate
    n_epochs    : int   — total training epochs
    warmup      : int   — KL annealing warmup epochs
    device      : torch.device
    save_dir    : str   — directory to save checkpoints and loss history

    Returns
    -------
    model   : trained AVITM
    history : dict with keys 'train_loss', 'recon_loss', 'kl_loss',
                             'test_loss', 'kl_weight'
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────
    print(f"Device: {device}")
    train_loader, test_loader, vocab = get_dataloaders()
    vocab_size = len(vocab)

    # ── Model ─────────────────────────────────────────────────────────────
    model = AVITM(
        vocab_size=vocab_size,
        n_topics=n_topics,
        hidden_size=hidden_size,
        dropout=dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.99, 0.999))

    # ── History ───────────────────────────────────────────────────────────
    history = {
        "train_loss"  : [],
        "recon_loss"  : [],
        "kl_loss"     : [],
        "test_loss"   : [],
        "kl_weight"   : [],
    }
    best_test_loss = float("inf")

    # ── Training ──────────────────────────────────────────────────────────
    print(f"\nTraining AVITM for {n_epochs} epochs...")
    print(f"  Topics      : {n_topics}")
    print(f"  Vocab size  : {vocab_size}")
    print(f"  KL warmup   : {warmup} epochs\n")

    for epoch in range(1, n_epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        kl_weight = get_kl_weight(epoch, warmup)

        epoch_loss   = 0.0
        epoch_recon  = 0.0
        epoch_kl     = 0.0

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            recon, mu, log_var = model(batch)
            loss, recon_loss, kl_loss = compute_loss(recon, batch, mu, log_var)

            # Apply KL annealing weight
            annealed_loss = recon_loss + kl_weight * kl_loss
            annealed_loss.backward()

            # Gradient clipping — prevents exploding gradients early in training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss  += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl    += kl_loss.item()

        # Average over batches
        n_batches    = len(train_loader)
        epoch_loss  /= n_batches
        epoch_recon /= n_batches
        epoch_kl    /= n_batches

        # ── Evaluate on test ──────────────────────────────────────────────
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                recon, mu, log_var = model(batch)
                loss, _, _ = compute_loss(recon, batch, mu, log_var)
                test_loss += loss.item()
        test_loss /= len(test_loader)

        # ── Save history ──────────────────────────────────────────────────
        history["train_loss"].append(epoch_loss)
        history["recon_loss"].append(epoch_recon)
        history["kl_loss"].append(epoch_kl)
        history["test_loss"].append(test_loss)
        history["kl_weight"].append(kl_weight)

        # ── Save best model ───────────────────────────────────────────────
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), save_dir / "best_model.pt")

        # ── Logging ───────────────────────────────────────────────────────
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{n_epochs} | "
                f"KL weight: {kl_weight:.2f} | "
                f"Train: {epoch_loss:.1f} | "
                f"Recon: {epoch_recon:.1f} | "
                f"KL: {epoch_kl:.1f} | "
                f"Test: {test_loss:.1f}"
            )

    # ── Save loss history ─────────────────────────────────────────────────
    history_path = save_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f)
    print(f"\nTraining complete.")
    print(f"  Best test loss : {best_test_loss:.1f}")
    print(f"  Model saved to : {save_dir / 'best_model.pt'}")
    print(f"  History saved  : {history_path}")

    return model, history, vocab


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, history, vocab = train()

    # Print sample topics from trained model
    print("\nSample topics from trained model:")
    topics = model.get_topics(vocab, top_n=10)
    for i, words in enumerate(topics[:5]):   # show first 5 topics
        print(f"  Topic {i+1:2d}: {', '.join(words)}")