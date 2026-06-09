"""
model.py
--------
AVITM: Autoencoded Variational Inference for Topic Models
Srivastava & Sutton (2017) https://arxiv.org/pdf/1703.01488

Architecture:
    Encoder : BOW (vocab_size,) -> MLP -> mu (n_topics,), log_var (n_topics,)
    Decoder : theta (n_topics,) -> Linear -> log_softmax -> BOW reconstruction

Loss (ELBO):
    reconstruction loss : multinomial log-likelihood
    KL loss             : KL(q(z|x) || p(z)) under Laplace approximation
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
    """
    Inference network: maps a BOW vector to approximate posterior parameters.

    Input  : x         (batch_size, vocab_size)  — raw word counts
    Output : mu        (batch_size, n_topics)     — posterior mean
             log_var   (batch_size, n_topics)     — posterior log variance
    """

    def __init__(self, vocab_size, n_topics, hidden_size=100):
        super().__init__()

        self.fc1      = nn.Linear(vocab_size, hidden_size)  #(200,2000) * (2000,100) = (200,100)
        self.fc2      = nn.Linear(hidden_size, hidden_size) #(200,100) * (100,100) = (200,100)
        self.fc_mu    = nn.Linear(hidden_size, n_topics)    #(200,100) * (100,50) = (200,50)
        self.fc_logvar = nn.Linear(hidden_size, n_topics)   #(200,100) * (100,50) = (200,50)

        self.activation = nn.Softplus()                     #(200,100) -> (200,100)

        # Batch norm on hidden layers — helps prevent component collapsing
        self.bn_logvar = nn.BatchNorm1d(n_topics)  #(200,100) -> (200,100)

    def forward(self, x):
        # 1. Pass through shared hidden layers with Softplus
        h = self.activation(self.fc1(x))
        h = self.activation(self.fc2(h))

        # 2. Calculate Mean
        mu = self.fc_mu(h)

        # 3. Calculate Variance and apply Batch Normalization
        log_var = self.bn_logvar(self.fc_logvar(h))

        return mu, log_var


class Decoder(nn.Module):
    """
    Generative network: maps topic proportions to a distribution over words.

    Input  : theta  (batch_size, n_topics)  — topic proportions
    Output : recon  (batch_size, vocab_size) — log prob over vocabulary
    """

    def __init__(self, n_topics, vocab_size, model_type="prodLDA"):
        super().__init__()
        self.model_type = model_type
        # Beta matrix: each row is a topic's word distribution
        # No bias — keeps the interpretation clean (pure topic-word weights)
        self.fc      = nn.Linear(n_topics, vocab_size, bias=False)  # beta matrix
        self.bn      = nn.BatchNorm1d(vocab_size)

    def forward(self, theta):
        if self.model_type == 'LDA':
            # Normalize beta per topic first, then mix
            beta  = F.softmax(self.fc.weight, dim=0)  # (vocab_size, n_topics)
            word_probs = theta @ beta.T
            recon = torch.log(word_probs + 1e-8)
        else:
            # ProdLDA: mix in logit space, softmax after
            recon = F.log_softmax(self.bn(self.fc(theta)), dim=1)
        return recon


class AVITM(nn.Module):
    """
    Full AVITM model combining encoder and decoder.

    Key design choice: Laplace approximation for the Dirichlet prior.
    Instead of sampling z ~ Dirichlet(alpha), we approximate the Dirichlet
    with a Gaussian in the logistic-normal space:
        z = softmax(mu + eps * sigma),  eps ~ N(0, I)
    This enables the reparameterization trick for gradient-based training.

    Parameters
    ----------
    vocab_size  : int   — size of vocabulary
    n_topics    : int   — number of latent topics (K)
    hidden_size : int   — hidden layer size in encoder MLP
    dropout     : float — dropout rate in encoder
    """

    def __init__(
        self,
        vocab_size,
        n_topics=50,
        hidden_size=100,
        dropout_rate=0.2,
        model_type="prodLDA",
        alpha = None
    ):
        super().__init__()

        self.n_topics   = n_topics
        self.vocab_size = vocab_size

        self.encoder = Encoder(vocab_size, n_topics, hidden_size)
        self.decoder = Decoder(n_topics, vocab_size, model_type)
        self.dropout_rate = dropout_rate
        # If no alpha is provided, default to 1/K
        if alpha is None:
            alpha = 1.0 / n_topics
        # # Prior: N(0, I) in the Laplace approximation space
        # # (approximates a symmetric Dirichlet prior)
        # self.prior_mean    = torch.zeros(n_topics)
        # self.prior_log_var = torch.zeros(n_topics)
        #
        # # Laplace approximation prior parameters (Equation 6, alpha=1)
        # # mu1 = 0 (all zeros when alpha=1)
        # # var1 = 1 - 1/K per dimension
        # self.prior_var = 1.0 - 1.0 / n_topics  # scalar, same for all dims

        # Laplace approximation prior parameters (Equation 6)
        # For a symmetric Dirichlet prior, mu is always 0.
        self.register_buffer('prior_mean', torch.zeros(1))

        # Variance formula: (1/alpha) * (1 - 1/K)
        prior_var = (1.0 / alpha) * (1.0 - (1.0 / n_topics))

        # We store the log variance to match the ELBO calculation later
        self.register_buffer('prior_log_var', torch.full((1,), math.log(prior_var)))

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick: z = mu + eps * sigma, eps ~ N(0, I)
        Enables gradients to flow through the sampling step.
        Only applied during training; at eval time we use the mean directly.
        """
        if self.training:
            std = torch.exp(0.5 * log_var)          # sigma
            eps = torch.randn_like(std)              # eps ~ N(0, I)
            return mu + eps * std
        else:
            return mu                                # use mean at eval time

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : (batch_size, vocab_size) — raw word count tensor

        Returns
        -------
        recon   : (batch_size, vocab_size) — log prob reconstruction
        mu      : (batch_size, n_topics)   — posterior mean
        log_var : (batch_size, n_topics)   — posterior log variance
        """
        # Encode
        mu, log_var = self.encoder(x)

        # Sample z via reparameterization
        z = self.reparameterize(mu, log_var)

        # Convert to topic proportions via softmax
        # This is the Laplace approximation: softmax(Gaussian) ≈ Dirichlet
        theta = F.softmax(z, dim=1)
        theta = F.dropout(theta, p=self.dropout_rate, training=self.training)

        # Decode
        recon = self.decoder(theta)

        return recon, mu, log_var

    def get_topics(self, vocab, top_n=10):
        """
        Extract top_n words for each topic from the decoder weights.
        Used for qualitative evaluation.

        Parameters
        ----------
        vocab : np.ndarray of shape (vocab_size,)
        top_n : int — number of top words to show per topic

        Returns
        -------
        topics : list of lists, shape (n_topics, top_n)
        """
        # Decoder weight matrix: (vocab_size, n_topics)
        # Each column = word distribution for one topic
        beta = self.decoder.fc.weight.detach().cpu()  # (vocab_size, n_topics)
        beta = F.softmax(beta, dim=0)                 # normalize over vocab

        topics = []
        for k in range(self.n_topics):
            top_indices = beta[:, k].argsort(descending=True)[:top_n]
            top_words   = [vocab[i] for i in top_indices]
            topics.append(top_words)

        return topics


def compute_loss(recon, x, mu, log_var, prior_mean, prior_log_var):
    """
    ELBO loss = reconstruction loss + KL divergence.

    Reconstruction loss:
        Multinomial log-likelihood: sum over vocab of (count * log_prob)
        This is equivalent to cross-entropy weighted by word counts.

    KL divergence (Laplace approximation):
        KL(N(mu, sigma^2) || N(0, I))
        = 0.5 * sum(sigma^2 + mu^2 - 1 - log(sigma^2))
        Closed-form solution — no sampling needed for this term.

    Parameters
    ----------
    recon   : (batch_size, vocab_size) — log probs from decoder
    x       : (batch_size, vocab_size) — raw word counts
    mu      : (batch_size, n_topics)   — posterior mean
    log_var : (batch_size, n_topics)   — posterior log variance

    Returns
    -------
    loss       : scalar — total ELBO loss (negated, for minimization)
    recon_loss : scalar — reconstruction term
    kl_loss    : scalar — KL divergence term
    """
    # Reconstruction: sum(BOW * log_prob), averaged over batch
    # x acts as weights — words that appear more contribute more to the loss
    recon_loss = -(x * recon).sum(dim=1).mean()

    # Full KL: KL(N(mu0, sigma0^2) || N(0, prior_var * I))
    # = 0.5 * sum(sigma0^2/prior_var + mu0^2/prior_var - 1 + log(prior_var) - log(sigma0^2))
    # We use the pre-calculated prior_log_var from the model buffers
    # instead of recalculating Equation 6 on the fly.
    prior_var = prior_log_var.exp()

    kl_loss = 0.5 * (
            log_var.exp() / prior_var
            + (mu - prior_mean).pow(2) / prior_var
            - 1
            + prior_log_var
            - log_var
    ).sum(dim=1).mean()

    loss = recon_loss + kl_loss

    return loss, recon_loss, kl_loss


# ── Sanity check ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    VOCAB_SIZE = 2000
    N_TOPICS   = 50
    BATCH_SIZE = 200

    # Fake batch of BOW vectors
    x = torch.randint(0, 10, (BATCH_SIZE, VOCAB_SIZE)).float()

    # Build model
    model = AVITM(vocab_size=VOCAB_SIZE, n_topics=N_TOPICS)
    print(model)

    # Forward pass
    recon, mu, log_var = model(x)
    print(f"\nForward pass:")
    print(f"  recon   : {recon.shape}")    # (200, 2000)
    print(f"  mu      : {mu.shape}")       # (200, 50)
    print(f"  log_var : {log_var.shape}")  # (200, 50)

    # Loss
    loss, recon_loss, kl_loss = compute_loss(
        recon,
        x,
        mu,
        log_var,
        model.prior_mean,
        model.prior_log_var
    )
    print(f"\nLoss:")
    print(f"  Total : {loss.item():.4f}")
    print(f"  Recon : {recon_loss.item():.4f}")
    print(f"  KL    : {kl_loss.item():.4f}")

    # Topics
    vocab = np.array([f"word_{i}" for i in range(VOCAB_SIZE)])
    topics = model.get_topics(vocab, top_n=5)
    print(f"\nSample topic (topic 0): {topics[0]}")