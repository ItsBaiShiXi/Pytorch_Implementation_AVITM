"""
gibbs.py
--------
Collapsed Gibbs sampling baseline for LDA.

This module provides a traditional LDA baseline that fits the same bag-of-words
data used by AVITM. The implementation samples only token-level topic
assignments and integrates out the document-topic and topic-word distributions.
"""

from __future__ import annotations

import numpy as np


def _to_numpy(matrix):
    """Convert a torch tensor or array-like object to a NumPy array."""
    if hasattr(matrix, "detach"):
        return matrix.detach().cpu().numpy()
    return np.asarray(matrix)


def _matrix_to_docs(doc_term_matrix):
    """Expand a dense BOW matrix into token-id arrays per document."""
    dense = _to_numpy(doc_term_matrix).astype(np.int32, copy=False)
    docs = []
    for row in dense:
        token_ids = np.repeat(np.arange(row.shape[0], dtype=np.int32), row)
        docs.append(token_ids)
    return docs


class CollapsedGibbsLDA:
    """Collapsed Gibbs LDA suitable for baseline comparisons."""

    def __init__(
        self,
        n_topics=50,
        alpha=0.1,
        beta=0.01,
        n_iters=20,
        doc_infer_iters=20,
        random_state=2333,
        verbose=True,
    ):
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.n_iters = n_iters
        self.doc_infer_iters = doc_infer_iters
        self.random_state = random_state
        self.verbose = verbose

        self.topic_word_counts_ = None
        self.topic_counts_ = None
        self.doc_topic_counts_ = None
        self.topic_word_dist_ = None
        self.doc_topic_dist_ = None
        self.vocab_size_ = None
        self._rng = np.random.default_rng(random_state)

    def fit(self, doc_term_matrix):
        """Fit the collapsed Gibbs sampler on a dense BOW matrix."""
        docs = _matrix_to_docs(doc_term_matrix)
        dense = _to_numpy(doc_term_matrix)
        n_docs, vocab_size = dense.shape

        self.vocab_size_ = vocab_size
        k = self.n_topics

        doc_topic_counts = np.zeros((n_docs, k), dtype=np.int32)
        topic_word_counts = np.zeros((k, vocab_size), dtype=np.int32)
        topic_counts = np.zeros(k, dtype=np.int32)

        topic_assignments = []

        for d, doc in enumerate(docs):
            if doc.size == 0:
                topic_assignments.append(np.empty(0, dtype=np.int32))
                continue

            assignments = self._rng.integers(0, k, size=doc.size, dtype=np.int32)
            topic_assignments.append(assignments)
            for word_id, topic_id in zip(doc, assignments):
                doc_topic_counts[d, topic_id] += 1
                topic_word_counts[topic_id, word_id] += 1
                topic_counts[topic_id] += 1

        for iteration in range(self.n_iters):
            if self.verbose:
                print(f"Collapsed Gibbs LDA iteration {iteration + 1}/{self.n_iters}")

            for d, doc in enumerate(docs):
                assignments = topic_assignments[d]
                if doc.size == 0:
                    continue

                for n, word_id in enumerate(doc):
                    old_topic = assignments[n]

                    doc_topic_counts[d, old_topic] -= 1
                    topic_word_counts[old_topic, word_id] -= 1
                    topic_counts[old_topic] -= 1

                    topic_likelihood = (
                        topic_word_counts[:, word_id] + self.beta
                    ) / (topic_counts + vocab_size * self.beta)
                    doc_likelihood = doc_topic_counts[d] + self.alpha
                    probs = topic_likelihood * doc_likelihood
                    probs = probs / probs.sum()

                    new_topic = self._rng.choice(k, p=probs)
                    assignments[n] = new_topic

                    doc_topic_counts[d, new_topic] += 1
                    topic_word_counts[new_topic, word_id] += 1
                    topic_counts[new_topic] += 1

        self.topic_word_counts_ = topic_word_counts
        self.topic_counts_ = topic_counts
        self.doc_topic_counts_ = doc_topic_counts

        self.topic_word_dist_ = (topic_word_counts + self.beta) / (
            topic_counts[:, None] + vocab_size * self.beta
        )
        self.doc_topic_dist_ = (doc_topic_counts + self.alpha) / (
            doc_topic_counts.sum(axis=1, keepdims=True) + k * self.alpha
        )

        return self

    def infer_document_topic_distribution(self, doc_term_row):
        """Infer a posterior mean topic distribution for one held-out document."""
        if self.topic_word_dist_ is None:
            raise RuntimeError("Model must be fit before inference.")

        row = _to_numpy(doc_term_row).astype(np.int32, copy=False)
        token_ids = np.repeat(np.arange(row.shape[0], dtype=np.int32), row)
        if token_ids.size == 0:
            return np.ones(self.n_topics, dtype=np.float64) / self.n_topics

        assignments = self._rng.integers(
            0, self.n_topics, size=token_ids.size, dtype=np.int32
        )
        doc_topic_counts = np.zeros(self.n_topics, dtype=np.int32)

        for topic_id in assignments:
            doc_topic_counts[topic_id] += 1

        for _ in range(self.doc_infer_iters):
            for n, word_id in enumerate(token_ids):
                old_topic = assignments[n]
                doc_topic_counts[old_topic] -= 1

                probs = self.topic_word_dist_[:, word_id] * (
                    doc_topic_counts + self.alpha
                )
                probs = probs / probs.sum()

                new_topic = self._rng.choice(self.n_topics, p=probs)
                assignments[n] = new_topic
                doc_topic_counts[new_topic] += 1

        return (doc_topic_counts + self.alpha) / (
            token_ids.size + self.n_topics * self.alpha
        )

    def transform(self, doc_term_matrix):
        """Infer topic proportions for every document in a matrix."""
        dense = _to_numpy(doc_term_matrix)
        return np.vstack([self.infer_document_topic_distribution(row) for row in dense])

    def perplexity(self, doc_term_matrix):
        """Approximate held-out perplexity using fold-in Gibbs inference."""
        dense = _to_numpy(doc_term_matrix)
        total_log_likelihood = 0.0
        total_word_count = 0.0

        for row in dense:
            theta = self.infer_document_topic_distribution(row)
            word_probs = theta @ self.topic_word_dist_
            counts = row.astype(np.float64, copy=False)
            mask = counts > 0
            if np.any(mask):
                total_log_likelihood += np.sum(
                    counts[mask] * np.log(np.clip(word_probs[mask], 1e-12, 1.0))
                )
                total_word_count += counts[mask].sum()

        if total_word_count == 0:
            return float("inf")

        return float(np.exp(-total_log_likelihood / total_word_count))

    def get_topics(self, vocab, top_n=10):
        """Return top words for each topic."""
        if self.topic_word_dist_ is None:
            raise RuntimeError("Model must be fit before extracting topics.")

        vocab = np.asarray(vocab)
        topics = []
        for topic_id in range(self.n_topics):
            top_indices = np.argsort(self.topic_word_dist_[topic_id])[::-1][:top_n]
            topics.append([vocab[idx] for idx in top_indices])
        return topics


__all__ = ["CollapsedGibbsLDA"]