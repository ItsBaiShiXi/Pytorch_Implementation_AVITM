"""
evaluate.py
-----------
Evaluation metrics for AVITM:
    1. Perplexity   : how surprised is the model by test documents
    2. NPMI         : how coherent are the learned topics

Usage in notebook:
    from evaluate import compute_perplexity, compute_npmi, print_topics
"""

import numpy as np
import torch
from itertools import combinations

from gibbs import CollapsedGibbsLDA


# ── 1. Perplexity ──────────────────────────────────────────────────────────
def compute_perplexity(model, data_loader, device):
    """
    Compute perplexity on a dataset.

    Perplexity = exp( -total_log_likelihood / total_word_count )

    Where total_log_likelihood = sum over all documents and words of:
        word_count * log p(word | document)

    Lower perplexity = better model.

    Parameters
    ----------
    model       : AVITM — trained model in eval mode
    data_loader : DataLoader — yields BOW tensors (batch_size, vocab_size)
    device      : torch.device

    Returns
    -------
    perplexity : float
    """
    model.eval()

    total_log_likelihood = 0.0  # sum of (x * log_prob) across all docs
    total_word_count     = 0.0  # sum of all word counts across all docs

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)

            # Forward pass — get log probabilities over vocabulary
            recon, mu, log_var = model(batch)
            # recon: (batch_size, vocab_size) — log p(w | theta)

            # Weighted log-likelihood: words that appear more contribute more
            # (x * recon).sum(dim=1): log-likelihood per document
            log_likelihood = (batch * recon).sum(dim=1).sum()  # scalar

            # Total words in this batch
            word_count = batch.sum()

            total_log_likelihood += log_likelihood.item()
            total_word_count     += word_count.item()

    # Average log-likelihood per word, then exponentiate
    perplexity = np.exp(-total_log_likelihood / total_word_count)
    return perplexity


# ── 2. Topic Coherence (NPMI) ──────────────────────────────────────────────
def compute_npmi(model, train_loader, vocab, top_n=10, device="cpu"):
    """
    Compute average NPMI topic coherence across all topics.

    For each topic:
        1. Get the top_n words
        2. For every pair of top words (w_i, w_j), compute NPMI:
               NPMI(w_i, w_j) = log[ p(w_i, w_j) / (p(w_i) * p(w_j)) ]
                                 / -log p(w_i, w_j)
        3. Average NPMI over all pairs in the topic
    Then average over all topics.

    p(w_i)       = fraction of documents containing word w_i
    p(w_i, w_j)  = fraction of documents containing BOTH w_i and w_j

    Higher NPMI = more coherent topics.

    Parameters
    ----------
    model        : AVITM — trained model
    train_loader : DataLoader — used to compute word co-occurrence stats
    vocab        : np.ndarray — vocabulary words, shape (vocab_size,)
    top_n        : int — number of top words per topic to evaluate
    device       : torch.device

    Returns
    -------
    avg_npmi     : float — average NPMI across all topics
    topic_npmis  : list  — NPMI score per topic
    """

    # ── Step 1: Compute word co-occurrence statistics from training data ───
    # We need document-level co-occurrence (not sentence-level)
    # So we binarize the BOW: 1 if word appears in doc, 0 otherwise
    print("Computing word co-occurrence statistics...")

    vocab_size  = len(vocab)
    n_docs      = 0
    word_counts = np.zeros(vocab_size)          # p(w_i): doc frequency
    # co_counts will be computed on-the-fly per topic to save memory

    # Collect all binarized documents
    all_docs = []
    for batch in train_loader:
        # Binarize: 1 if word appears, 0 otherwise
        binary = (batch > 0).float().numpy()    # (batch_size, vocab_size)
        all_docs.append(binary)
        word_counts += binary.sum(axis=0)
        n_docs      += binary.shape[0]

    all_docs    = np.vstack(all_docs)           # (n_docs, vocab_size)
    word_counts = word_counts / n_docs          # normalize to probabilities
    print(f"  Documents used : {n_docs}")

    # ── Step 2: Get top words for each topic ──────────────────────────────
    topics     = model.get_topics(vocab, top_n=top_n)  # list of word lists
    n_topics   = len(topics)

    # ── Step 3: Compute NPMI per topic ────────────────────────────────────
    topic_npmis = []

    for k, top_words in enumerate(topics):
        # Get vocabulary indices for this topic's top words
        word_to_idx = {w: i for i, w in enumerate(vocab)}
        top_indices = [word_to_idx[w] for w in top_words if w in word_to_idx]

        if len(top_indices) < 2:
            topic_npmis.append(0.0)
            continue

        # Compute NPMI for every pair of top words
        pair_npmis = []
        for i, j in combinations(top_indices, 2):
            p_i  = word_counts[i]           # p(w_i)
            p_j  = word_counts[j]           # p(w_j)

            # p(w_i, w_j): fraction of docs containing both words
            co_occur = (all_docs[:, i] * all_docs[:, j]).mean()

            # Avoid log(0) — skip pairs that never co-occur
            if co_occur < 1e-12 or p_i < 1e-12 or p_j < 1e-12:
                continue

            # PMI = log[ p(w_i, w_j) / (p(w_i) * p(w_j)) ]
            pmi  = np.log(co_occur / (p_i * p_j))

            # Normalize: NPMI = PMI / -log(p(w_i, w_j))
            npmi = pmi / (-np.log(co_occur))

            pair_npmis.append(npmi)

        topic_npmi = np.mean(pair_npmis) if pair_npmis else 0.0
        topic_npmis.append(topic_npmi)

    avg_npmi = np.mean(topic_npmis)
    return avg_npmi, topic_npmis


# ── 3. Print topics ────────────────────────────────────────────────────────
def print_topics(model, vocab, top_n=10, topic_npmis=None):
    """
    Print all topics with their top words and optional NPMI scores.

    Parameters
    ----------
    model       : AVITM
    vocab       : np.ndarray
    top_n       : int
    topic_npmis : list or None — if provided, print NPMI next to each topic
    """
    topics = model.get_topics(vocab, top_n=top_n)

    print(f"\n{'='*60}")
    print(f"  Learned Topics (top {top_n} words each)")
    print(f"{'='*60}")

    for k, words in enumerate(topics):
        npmi_str = ""
        if topic_npmis is not None:
            npmi_str = f"  [NPMI: {topic_npmis[k]:.3f}]"
        print(f"  Topic {k+1:2d}{npmi_str}: {', '.join(words)}")

    print(f"{'='*60}\n")


def evaluate_gibbs_lda(train_loader, test_loader, vocab, n_topics=50):
    """Train and evaluate the collapsed Gibbs LDA baseline."""
    print("\nTraining collapsed Gibbs LDA baseline...")

    train_matrix = train_loader.dataset.data
    test_matrix = test_loader.dataset.data

    lda = CollapsedGibbsLDA(n_topics=n_topics, verbose=True)
    lda.fit(train_matrix)

    print("\nEvaluating collapsed Gibbs LDA baseline...")
    train_ppl = lda.perplexity(train_matrix)
    test_ppl = lda.perplexity(test_matrix)
    avg_npmi, topic_npmis = compute_npmi(lda, train_loader, vocab, top_n=10, device="cpu")

    print(f"  Train perplexity : {train_ppl:.1f}")
    print(f"  Test  perplexity : {test_ppl:.1f}")
    print(f"  Average NPMI     : {avg_npmi:.4f}")
    print_topics(lda, vocab, top_n=10, topic_npmis=topic_npmis)

    return {
        "model": lda,
        "train_perplexity": train_ppl,
        "test_perplexity": test_ppl,
        "avg_npmi": avg_npmi,
        "topic_npmis": topic_npmis,
    }


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data import get_dataloaders
    from model import AVITM

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, test_loader, vocab = get_dataloaders()
    vocab_size = len(vocab)

    # Load trained model
    model = AVITM(vocab_size=vocab_size, n_topics=50).to(DEVICE)
    model.load_state_dict(
        torch.load("checkpoints/best_model.pt", map_location=DEVICE)
    )
    model.eval()

    # ── Perplexity ────────────────────────────────────────────────────────
    print("Computing perplexity...")
    train_ppl = compute_perplexity(model, train_loader, DEVICE)
    test_ppl  = compute_perplexity(model, test_loader,  DEVICE)
    print(f"  Train perplexity : {train_ppl:.1f}")
    print(f"  Test  perplexity : {test_ppl:.1f}")

    # ── NPMI ──────────────────────────────────────────────────────────────
    print("\nComputing topic coherence (NPMI)...")
    avg_npmi, topic_npmis = compute_npmi(
        model, train_loader, vocab, top_n=10, device=DEVICE
    )
    print(f"  Average NPMI : {avg_npmi:.4f}")

    # ── Print topics ──────────────────────────────────────────────────────
    print_topics(model, vocab, top_n=10, topic_npmis=topic_npmis)

    # ── Collapsed Gibbs LDA baseline ─────────────────────────────────────
    gibbs_results = evaluate_gibbs_lda(train_loader, test_loader, vocab, n_topics=50)

    print("\n=== Comparison summary ===")
    print(
        f"AVITM: train ppl={train_ppl:.1f}, test ppl={test_ppl:.1f}, "
        f"NPMI={avg_npmi:.4f}"
    )
    print(
        f"Gibbs: train ppl={gibbs_results['train_perplexity']:.1f}, "
        f"test ppl={gibbs_results['test_perplexity']:.1f}, "
        f"NPMI={gibbs_results['avg_npmi']:.4f}"
    )