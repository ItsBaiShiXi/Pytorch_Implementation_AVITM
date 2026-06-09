"""
run_all.py
----------
Trains and evaluates all four models and outputs a comparison table.

Models:
    1. LDA-DMFVI    : sklearn online variational inference (Hoffman et al. 2010)
    2. Gibbs LDA    : collapsed Gibbs sampling (Griffiths & Steyvers 2004)
    3. AVITM LDA    : autoencoded VI with LDA decoder (mixture model)
    4. AVITM ProdLDA: autoencoded VI with product-of-experts decoder

Output:
    checkpoints/all_results.json  — full results for notebook/slides
"""

import torch
import torch.optim as optim
import numpy as np
import json
import time
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation
from itertools import combinations

from data import get_dataloaders
from model import AVITM, compute_loss
from gibbs import CollapsedGibbsLDA


# ── Hyperparameters ────────────────────────────────────────────────────────
N_TOPICS     = 50
HIDDEN_SIZE  = 100
DROPOUT      = 0.2
LR           = 2e-3
N_EPOCHS     = 100
WARMUP       = 20
RANDOM_STATE = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Shared NPMI utility ────────────────────────────────────────────────────
def build_cooccurrence(train_loader):
    """Build binarized document matrix and word frequencies from DataLoader."""
    all_docs = []
    for batch in train_loader:
        binary = (batch > 0).float().numpy()
        all_docs.append(binary)
    all_docs    = np.vstack(all_docs)
    word_counts = all_docs.mean(axis=0)
    return all_docs, word_counts


def npmi_from_topics(topic_words_list, all_docs, word_counts, vocab):
    """
    Compute per-topic and average NPMI given a list of top-word lists.
    Works for any model — just pass in the top words per topic.
    """
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    topic_npmis = []

    for top_words in topic_words_list:
        top_indices = [word_to_idx[w] for w in top_words if w in word_to_idx]
        if len(top_indices) < 2:
            topic_npmis.append(0.0)
            continue

        pair_npmis = []
        for i, j in combinations(top_indices, 2):
            p_i      = word_counts[i]
            p_j      = word_counts[j]
            co_occur = (all_docs[:, i] * all_docs[:, j]).mean()

            if co_occur < 1e-12 or p_i < 1e-12 or p_j < 1e-12:
                continue

            pmi  = np.log(co_occur / (p_i * p_j))
            npmi = pmi / (-np.log(co_occur))
            pair_npmis.append(npmi)

        topic_npmis.append(np.mean(pair_npmis) if pair_npmis else 0.0)

    return float(np.mean(topic_npmis)), [float(x) for x in topic_npmis]


# ── KL annealing ───────────────────────────────────────────────────────────
def get_kl_weight(epoch, warmup=WARMUP):
    return min(1.0, epoch / warmup)


# ── AVITM training ─────────────────────────────────────────────────────────
def train_avitm(train_loader, test_loader, vocab_size, model_type,
                save_path, n_topics=N_TOPICS):
    """Train one AVITM variant (LDA or prodLDA) and return trained model."""
    model = AVITM(
        vocab_size=vocab_size,
        n_topics=n_topics,
        hidden_size=HIDDEN_SIZE,
        dropout_rate=DROPOUT,
        model_type=model_type,
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.99, 0.999))
    best_test_loss = float("inf")

    print(f"\n  Training AVITM ({model_type})...")
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        kl_weight = get_kl_weight(epoch)

        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon, mu, log_var = model(batch)
            loss, recon_loss, kl_loss = compute_loss(recon, batch, mu, log_var, model.prior_mean, model.prior_log_var)
            annealed_loss = recon_loss + kl_weight * kl_loss
            annealed_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Eval
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(DEVICE)
                recon, mu, log_var = model(batch)
                loss, _, _ = compute_loss(recon, batch, mu, log_var, model.prior_mean, model.prior_log_var)
                test_loss += loss.item()
        test_loss /= len(test_loader)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), save_path)

        if epoch % 20 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{N_EPOCHS} | "
                  f"KL weight: {kl_weight:.2f} | Test loss: {test_loss:.1f}")

    # Load best checkpoint
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    model.eval()
    return model


def eval_avitm_perplexity(model, data_loader):
    """Compute perplexity for AVITM using the full ELBO."""
    model.eval()
    total_elbo = 0.0
    total_word_count = 0.0
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(DEVICE)
            recon, mu, log_var = model(batch)

            # Use the full compute_loss function to get the ELBO
            # (loss is the negative ELBO)
            loss, _, _ = compute_loss(
                recon,
                batch,
                mu,
                log_var,
                model.prior_mean,
                model.prior_log_var
            )

            # We multiply by the batch size because your compute_loss
            # takes the .mean() across the batch, but we need the raw sum
            total_elbo += (loss.item() * batch.shape[0])
            total_word_count += batch.sum().item()

    # Perplexity = exp( Total Negative ELBO / Total Words )
    return float(np.exp(total_elbo / total_word_count))


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    save_dir = Path(f"checkpoints_k{N_TOPICS}")
    save_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("  Loading data...")
    print("=" * 60)
    train_loader, test_loader, vocab = get_dataloaders()
    vocab_size = len(vocab)

    # Build co-occurrence matrix once — shared across all NPMI computations
    print("\nBuilding co-occurrence matrix (used for all NPMI computations)...")
    all_docs, word_counts = build_cooccurrence(train_loader)
    print(f"  Documents: {all_docs.shape[0]}, Vocab: {all_docs.shape[1]}")

    # Reconstruct numpy matrices for sklearn and Gibbs
    X_train = np.vstack([b.numpy() for b in train_loader])
    X_test  = np.vstack([b.numpy() for b in test_loader])

    all_results = {}

    # ── 1. AVITM ProdLDA ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  1. AVITM ProdLDA (product-of-experts decoder)")
    print("=" * 60)
    start_time = time.perf_counter()
    avitm_prod = train_avitm(
        train_loader, test_loader, vocab_size,
        model_type='prodLDA',
        save_path=save_dir / "avitm_prodlda_best.pt",
    )
    elapsed_time = time.perf_counter() - start_time

    avitm_prod_train_ppl = eval_avitm_perplexity(avitm_prod, train_loader)
    avitm_prod_test_ppl = eval_avitm_perplexity(avitm_prod, test_loader)
    avitm_prod_words = avitm_prod.get_topics(vocab, top_n=10)
    avitm_prod_avg_npmi, avitm_prod_topic_npmis = npmi_from_topics(
        avitm_prod_words, all_docs, word_counts, vocab
    )

    print(f"  Train perplexity : {avitm_prod_train_ppl:.1f}")
    print(f"  Test  perplexity : {avitm_prod_test_ppl:.1f}")
    print(f"  Avg NPMI         : {avitm_prod_avg_npmi:.4f}")
    print(f"  Training Time    : {elapsed_time:.1f} s")

    all_results["AVITM-ProdLDA"] = {
        "train_ppl": avitm_prod_train_ppl,
        "test_ppl": avitm_prod_test_ppl,
        "avg_npmi": avitm_prod_avg_npmi,
        "topic_npmis": avitm_prod_topic_npmis,
        "topic_words": avitm_prod_words,
        "time_seconds": elapsed_time
    }

    # ── 2. AVITM LDA-VAE ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  2. AVITM LDA-VAE (mixture model decoder)")
    print("=" * 60)

    start_time = time.perf_counter()
    avitm_lda = train_avitm(
        train_loader, test_loader, vocab_size,
        model_type='LDA',
        save_path=save_dir / "avitm_lda_best.pt",
    )
    elapsed_time = time.perf_counter() - start_time

    avitm_lda_train_ppl = eval_avitm_perplexity(avitm_lda, train_loader)
    avitm_lda_test_ppl = eval_avitm_perplexity(avitm_lda, test_loader)
    avitm_lda_words = avitm_lda.get_topics(vocab, top_n=10)
    avitm_lda_avg_npmi, avitm_lda_topic_npmis = npmi_from_topics(
        avitm_lda_words, all_docs, word_counts, vocab
    )

    print(f"  Train perplexity : {avitm_lda_train_ppl:.1f}")
    print(f"  Test  perplexity : {avitm_lda_test_ppl:.1f}")
    print(f"  Avg NPMI         : {avitm_lda_avg_npmi:.4f}")
    print(f"  Training Time    : {elapsed_time:.1f} s")

    all_results["AVITM-LDA"] = {
        "train_ppl": avitm_lda_train_ppl,
        "test_ppl": avitm_lda_test_ppl,
        "avg_npmi": avitm_lda_avg_npmi,
        "topic_npmis": avitm_lda_topic_npmis,
        "topic_words": avitm_lda_words,
        "time_seconds": elapsed_time
    }

    # ── 3. LDA-DMFVI (sklearn online VI) ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  3. LDA-DMFVI (sklearn online variational inference)")
    print("=" * 60)
    lda_dmfvi = LatentDirichletAllocation(
        n_components=N_TOPICS,
        max_iter=200,
        learning_method='online',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        learning_offset=50.0,
        learning_decay=0.7,
    )
    start_time = time.perf_counter()
    lda_dmfvi.fit(csr_matrix(X_train))
    elapsed_time = time.perf_counter() - start_time

    # Perplexity via log-likelihood (consistent with AVITM)
    dmfvi_train_ppl = float(np.exp(
        -lda_dmfvi.score(csr_matrix(X_train)) / X_train.sum()
    ))
    dmfvi_test_ppl = float(np.exp(
        -lda_dmfvi.score(csr_matrix(X_test)) / X_test.sum()
    ))

    # Topics and NPMI
    dmfvi_topic_words = []
    for topic in lda_dmfvi.components_:
        top_indices = topic.argsort()[-10:][::-1]
        dmfvi_topic_words.append([vocab[i] for i in top_indices])

    dmfvi_avg_npmi, dmfvi_topic_npmis = npmi_from_topics(
        dmfvi_topic_words, all_docs, word_counts, vocab
    )

    print(f"  Train perplexity : {dmfvi_train_ppl:.1f}")
    print(f"  Test  perplexity : {dmfvi_test_ppl:.1f}")
    print(f"  Avg NPMI         : {dmfvi_avg_npmi:.4f}")
    print(f"  Training Time    : {elapsed_time:.1f} s")

    all_results["LDA-DMFVI"] = {
        "train_ppl"   : dmfvi_train_ppl,
        "test_ppl"    : dmfvi_test_ppl,
        "avg_npmi"    : dmfvi_avg_npmi,
        "topic_npmis" : dmfvi_topic_npmis,
        "topic_words" : dmfvi_topic_words,
        "time_seconds": elapsed_time
    }

    # ── 4. Collapsed Gibbs LDA ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  4. Collapsed Gibbs LDA")
    print("=" * 60)
    gibbs = CollapsedGibbsLDA(
        n_topics=N_TOPICS,
        n_iters=20,
        random_state=RANDOM_STATE,
        verbose=True,
    )
    start_time = time.perf_counter()
    gibbs.fit(X_train)
    elapsed_time = time.perf_counter() - start_time

    gibbs_train_ppl = gibbs.perplexity(X_train)
    gibbs_test_ppl  = gibbs.perplexity(X_test)
    gibbs_topic_words = gibbs.get_topics(vocab, top_n=10)
    gibbs_avg_npmi, gibbs_topic_npmis = npmi_from_topics(
        gibbs_topic_words, all_docs, word_counts, vocab
    )

    print(f"  Train perplexity : {gibbs_train_ppl:.1f}")
    print(f"  Test  perplexity : {gibbs_test_ppl:.1f}")
    print(f"  Avg NPMI         : {gibbs_avg_npmi:.4f}")
    print(f"  Training Time    : {elapsed_time:.1f} s")

    all_results["Gibbs-LDA"] = {
        "train_ppl"   : float(gibbs_train_ppl),
        "test_ppl"    : float(gibbs_test_ppl),
        "avg_npmi"    : gibbs_avg_npmi,
        "topic_npmis" : gibbs_topic_npmis,
        "topic_words" : gibbs_topic_words,
        "time_seconds": elapsed_time
    }





    # ── Save all results ───────────────────────────────────────────────────
    results_path = save_dir / "all_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {results_path}")

    # ── Print comparison table ─────────────────────────────────────────────
    print("\n" + "=" * 75)
    print(f"  {'Model':<20} {'Test PPL':>10} {'Avg NPMI':>10} {'Time (s)':>10}")
    print("=" * 75)
    for name, res in all_results.items():
        print(f"  {name:<20} {res['test_ppl']:>10.1f} {res['avg_npmi']:>10.4f} {res['time_seconds']:>10.1f}")
    print("-" * 75)
    print(f"  {'Paper ProdLDA-VAE':<20} {'1172':>10} {'0.24':>10} {'-':>10}")
    print(f"  {'Paper LDA-VAE':<20} {'1059':>10} {'0.11':>10} {'-':>10}")
    print(f"  {'Paper LDA-DMFVI':<20} {'1046':>10} {'0.11':>10} {'-':>10}")
    print(f"  {'Paper Gibbs':<20} {'728':>10} {'0.17':>10} {'-':>10}")
    print("=" * 75)

    return all_results


if __name__ == "__main__":
    main()