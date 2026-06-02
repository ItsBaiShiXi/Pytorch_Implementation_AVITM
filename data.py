"""
data.py
-------
Data preprocessing pipeline for AVITM.

Outputs:
    train_loader : DataLoader of shape (batch_size, vocab_size), raw word counts
    test_loader  : DataLoader of shape (batch_size, vocab_size), raw word counts
    vocab        : np.ndarray of shape (vocab_size,), word strings
"""

import numpy as np
import torch
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────
VOCAB_SIZE   = 2000
BATCH_SIZE   = 200
TEST_SPLIT   = 0.2
RANDOM_STATE = 2333


# ── Dataset ────────────────────────────────────────────────────────────────
class BOWDataset(Dataset):
    """
    Wraps a bag-of-words sparse matrix as a PyTorch Dataset.
    Each item is a float32 tensor of raw word counts, shape (vocab_size,).
    """
    def __init__(self, bow_matrix):
        # Convert sparse matrix → dense numpy → float32 tensor
        # bow_matrix: scipy sparse (n_docs, vocab_size)
        self.data = torch.tensor(
            bow_matrix.toarray(), dtype=torch.float32
        )

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


# ── Main function ──────────────────────────────────────────────────────────
def get_dataloaders(
    vocab_size=VOCAB_SIZE,
    batch_size=BATCH_SIZE,
    test_split=TEST_SPLIT,
    random_state=RANDOM_STATE,
):
    """
    Load 20 Newsgroups, preprocess into BOW, return DataLoaders and vocab.

    Parameters
    ----------
    vocab_size    : int   — number of words to keep (by corpus frequency)
    batch_size    : int   — batch size for both loaders
    test_split    : float — fraction of data reserved for test
    random_state  : int   — random seed for reproducibility

    Returns
    -------
    train_loader : DataLoader
    test_loader  : DataLoader
    vocab        : np.ndarray of shape (vocab_size,)
    """

    # ── 1. Load data ───────────────────────────────────────────────────────
    print("Loading 20 Newsgroups...")
    train_raw = fetch_20newsgroups(
        subset='train',
        remove=('headers', 'footers', 'quotes'),
        random_state=random_state,
    )
    test_raw = fetch_20newsgroups(
        subset='test',
        remove=('headers', 'footers', 'quotes'),
        random_state=random_state,
    )
    print(f"  Train docs : {len(train_raw.data)}")
    print(f"  Test docs  : {len(test_raw.data)}")

    # ── 2. Fit vectorizer on train, transform both splits ──────────────────
    # Key decisions (matching AVITM paper):
    #   - max_features=vocab_size : keep top-N words by corpus frequency
    #   - min_df=5               : ignore very rare words (noise)
    #   - max_df=0.7             : ignore words appearing in >70% of docs
    #   - stop_words='english'   : remove common English stopwords
    print("Fitting CountVectorizer...")
    vectorizer = CountVectorizer(
        max_features=vocab_size,
        min_df=5,
        max_df=0.7,
        stop_words='english',
        token_pattern=r'(?u)\b[a-zA-Z]{3,}\b',
    )
    train_bow = vectorizer.fit_transform(train_raw.data)  # fit only on train
    test_bow  = vectorizer.transform(test_raw.data)       # apply to test

    vocab = np.array(vectorizer.get_feature_names_out())
    print(f"  Vocabulary size : {len(vocab)}")
    print(f"  Train matrix    : {train_bow.shape}")
    print(f"  Test matrix     : {test_bow.shape}")
    vocab_path = Path("vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(vocab.tolist(), f)
    print(f"  Vocab saved to {vocab_path}")

    # ── 3. Filter out empty documents ─────────────────────────────────────
    # Some docs become empty after stopword removal — these cause NaN in loss
    train_mask = np.array(train_bow.sum(axis=1)).flatten() > 0
    test_mask  = np.array(test_bow.sum(axis=1)).flatten() > 0
    train_bow  = train_bow[train_mask]
    test_bow   = test_bow[test_mask]
    print(f"  After filtering empty docs:")
    print(f"    Train : {train_bow.shape[0]} docs")
    print(f"    Test  : {test_bow.shape[0]} docs")

    # ── 4. Wrap in Dataset and DataLoader ─────────────────────────────────
    train_dataset = BOWDataset(train_bow)
    test_dataset  = BOWDataset(test_bow)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,                # shuffle train for stochastic training
        drop_last=True,              # drop incomplete last batch
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,               # no shuffle for eval
        drop_last=False,
    )

    print(f"\nDataLoaders ready.")
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Test batches  : {len(test_loader)}")

    return train_loader, test_loader, vocab

def load_vocab(path="vocab.json"):
    """Load vocabulary from disk."""
    with open(path, "r") as f:
        vocab = json.load(f)
    return np.array(vocab)

# ── Quick sanity check ─────────────────────────────────────────────────────
if __name__ == "__main__":
    train_loader, test_loader, vocab = get_dataloaders()

    # Check one batch
    batch = next(iter(train_loader))
    print(f"\nSanity check:")
    print(f"  Batch shape : {batch.shape}")       # (batch_size, vocab_size)
    print(f"  dtype       : {batch.dtype}")        # float32
    print(f"  Min / Max   : {batch.min():.0f} / {batch.max():.0f}")  # counts
    print(f"  Vocab sample: {vocab[:10]}")         # first 10 words