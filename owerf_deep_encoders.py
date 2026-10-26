"""
OW-ERF deep encoders (PyTorch). STATUS: written and syntax-checked only.
NOT trained, NOT evaluated, and not run in the environment where it was written
(PyTorch was unavailable there). Run the smoke test first:  python owerf_deep_encoders.py

Branches (each must be justified by an ablation before you claim it helps):
  - LSTM        : sequence of the provider's / beneficiary's recent claims (temporal behaviour)
  - 1D-CNN      : ordered procedure / line-item code sequence inside one claim (local code patterns)
  - Transformer : self-attention over relational-context tokens (claim, hospital, doctor,
                  package, diagnosis, neighbours). In the full graph version, replace with
                  PyG HGTConv / RGCNConv on a time-stamped heterogeneous graph.
Heads:
  - classifier head   : logits over KNOWN classes -> energy score
  - expectation head  : mean and log-sigma of log-amount and LOS from EXOGENOUS inputs only
                        (never from fields a fraudster controls)
Losses:
  - cross-entropy + squared-hinge energy-margin loss with episodic pseudo-novel masking
  - Gaussian negative log-likelihood for the expectation head (benign claims only)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OWERFEncoder(nn.Module):
    def __init__(self, hist_dim=6, n_codes=500, ctx_dim=16, d=64, n_classes=4, exo_dim=3, n_targets=2):
        super().__init__()
        # --- LSTM over claim history: (B, L, hist_dim) -> (B, d)
        self.lstm = nn.LSTM(hist_dim, d, batch_first=True)
        # --- 1D-CNN over code sequence: (B, Lc) ints -> (B, d)
        self.code_emb = nn.Embedding(n_codes, 32, padding_idx=0)
        self.conv3 = nn.Conv1d(32, d // 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(32, d // 2, kernel_size=5, padding=2)
        # --- Transformer encoder over context tokens: (B, T, ctx_dim) -> (B, d)
        self.tok_proj = nn.Linear(ctx_dim, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=2 * d, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        # --- fusion + heads
        self.fuse = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Dropout(0.1))
        self.classifier = nn.Linear(d, n_classes)
        self.expect = nn.Sequential(nn.Linear(exo_dim, 32), nn.ReLU(), nn.Linear(32, 2 * n_targets))
        self.n_targets = n_targets

    def embed(self, hist, codes, ctx, ctx_pad_mask):
        _, (h_n, _) = self.lstm(hist)
        z_lstm = h_n[-1]                                            # (B, d)
        e = self.code_emb(codes).transpose(1, 2)                    # (B, 32, Lc)
        c = torch.cat([F.relu(self.conv3(e)), F.relu(self.conv5(e))], dim=1)
        z_cnn = c.max(dim=2).values                                 # (B, d)
        tok = self.tok_proj(ctx)                                    # (B, T, d)
        cls = self.cls.expand(tok.size(0), -1, -1)
        tok = torch.cat([cls, tok], dim=1)
        pad = torch.cat([torch.zeros_like(ctx_pad_mask[:, :1]), ctx_pad_mask], dim=1)
        z_tr = self.transformer(tok, src_key_padding_mask=pad)[:, 0]  # CLS token
        return self.fuse(torch.cat([z_lstm, z_cnn, z_tr], dim=1))

    def forward(self, hist, codes, ctx, ctx_pad_mask, exo):
        h = self.embed(hist, codes, ctx, ctx_pad_mask)
        logits = self.classifier(h)
        mu, log_sigma = self.expect(exo).chunk(2, dim=1)
        log_sigma = log_sigma.clamp(-3.0, 2.0)  # keep heteroscedastic NLL numerically stable
        return h, logits, mu, log_sigma


def energy(logits, T=1.0):
    return -T * torch.logsumexp(logits / T, dim=1)


def episodic_loss(logits, y, known_fraud_ids, m_in=-25.0, m_out=-7.0, lam=0.1):
    """Mask one known fraud class as pseudo-novel for this episode.
    y: (B,) class ids (0 = benign). Returns CE on remaining classes + energy-margin loss."""
    k_star = known_fraud_ids[torch.randint(len(known_fraud_ids), (1,)).item()]
    col_mask = torch.zeros(logits.size(1), dtype=torch.bool, device=logits.device)
    col_mask[k_star] = True
    logits_ep = logits.masked_fill(col_mask, -1e4)           # classifier "does not know" k_star
    known = y != k_star
    novel = y == k_star
    ce = F.cross_entropy(logits_ep[known], y[known]) if known.any() else logits.sum() * 0
    E = energy(logits_ep)
    l_in = F.relu(E[known] - m_in).pow(2).mean() if known.any() else E.sum() * 0
    l_out = F.relu(m_out - E[novel]).pow(2).mean() if novel.any() else E.sum() * 0
    return ce + lam * (l_in + l_out)


def gaussian_nll(mu, log_sigma, target):
    """Heteroscedastic Gaussian NLL: ln(sigma) + (o - mu)^2 / (2 sigma^2). Train on BENIGN claims only."""
    return (log_sigma + (target - mu).pow(2) / (2 * torch.exp(2 * log_sigma))).mean()


def inconsistency_ratio(mu, log_sigma, target):
    """a_v = max_j max(0, (o_j - mu_j) / sigma_j). Compare with a threshold tau calibrated on benign claims."""
    return torch.clamp((target - mu) / torch.exp(log_sigma), min=0).max(dim=1).values


if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, Lc, T = 8, 10, 12, 6
    model = OWERFEncoder()
    hist = torch.randn(B, L, 6)
    codes = torch.randint(1, 500, (B, Lc))
    ctx = torch.randn(B, T, 16)
    pad = torch.zeros(B, T, dtype=torch.bool)
    pad[:, -2:] = True                                          # last 2 tokens are padding
    exo = torch.randn(B, 3)
    y = torch.randint(0, 4, (B,))
    h, logits, mu, log_sigma = model(hist, codes, ctx, pad, exo)
    loss = episodic_loss(logits, y, known_fraud_ids=[1, 2, 3]) + gaussian_nll(mu, log_sigma, torch.randn(B, 2))
    loss.backward()
    print("shapes OK:", h.shape, logits.shape, mu.shape, "| loss:", float(loss))
