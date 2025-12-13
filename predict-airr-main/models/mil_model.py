# models/mil_model.py
"""
MIL model implementing:
 - Encoder wrapper: optional ESM (if available) or trainable embedding
 - Small 1D CNN to capture local/gapped motifs
 - Concatenate V/J one-hot -> projection -> gated attention aggregator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# optional import for fair ESM use (if user installed facebookresearch/esm)
try:
    import esm
    _ESM_AVAILABLE = True
except Exception:
    _ESM_AVAILABLE = False

class ESMFallbackEncoder(nn.Module):
    """
    Wrapper that uses ESM if available; otherwise a small character-level embedding + CNN.
    Interface:
      - encode_batch(list[str]) -> torch.FloatTensor (N, output_dim)
    """
    def __init__(self, embed_dim: int = 512, device: str = "cpu", use_pretrained: bool = False):
        super().__init__()
        self.device = device
        self.output_dim = embed_dim
        self.use_pretrained = use_pretrained and _ESM_AVAILABLE
        if self.use_pretrained:
            # load ESM model (small / faster variant). user must install esm.
            self.esm_model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            self.esm_model = self.esm_model.eval().to(self.device)
            # projection to desired dim
            self.proj = nn.Linear(self.esm_model.args.embed_dim, embed_dim)
        else:
            # fallback trainable embedding: amino-acid tokens + CNN pool
            aa_letters = "ACDEFGHIKLMNPQRSTVWYBXZJUO"  # include possible extras
            self.aa_to_idx = {c: i+1 for i, c in enumerate(aa_letters)}  # 0 reserved for PAD/UNK
            vocab_size = len(self.aa_to_idx) + 2
            self.token_emb = nn.Embedding(vocab_size, 64, padding_idx=0)
            self.conv = nn.Conv1d(in_channels=64, out_channels=embed_dim, kernel_size=5, padding=2)
            self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        raise RuntimeError("Use encode_batch(list[str]) instead for convenience.")

    def encode_batch(self, seqs: list):
        """
        seqs: list[str] amino-acid sequences (junction_aa)
        returns: tensor (N, output_dim) on CPU (detached). Model weights live on self.device.
        """
        if self.use_pretrained:
            # esm batch encode
            # Note: ESM expects batched tokens via alphabet.get_batch_converter()
            batch_converter = self.alphabet.get_batch_converter()
            # prepare tuple list
            batch = [("seq"+str(i), s) for i, s in enumerate(seqs)]
            labels, strs, toks = batch_converter(batch)  # toks on CPU
            toks = toks.to(self.device)
            with torch.no_grad():
                results = self.esm_model(toks, repr_layers=[len(self.esm_model.layers)], return_contacts=False)
                # use LM representation (per-residue) -> average pooling
                token_reprs = results["representations"][len(self.esm_model.layers)]
                # remove start token; pool over residues where token != pad (token 1?)
                mask = (toks != self.alphabet.padding_idx).float().unsqueeze(-1)
                summed = (token_reprs * mask).sum(1)
                lens = mask.sum(1)
                pooled = summed / lens
                emb = self.proj(pooled)  # (N, embed_dim)
            return emb.cpu()
        else:
            # fallback: simple char-tokenization -> token_emb -> conv -> pool -> (N, out_dim)
            max_len = max([len(s) for s in seqs]) if seqs else 1
            toks = []
            for s in seqs:
                idxs = [self.aa_to_idx.get(ch, 0) for ch in s]
                # pad to max_len
                idxs = idxs + [0]*(max_len - len(idxs))
                toks.append(idxs)
            toks = torch.tensor(toks, dtype=torch.long, device=self.device)  # (N, L)
            emb = self.token_emb(toks)  # (N, L, emb_dim)
            emb = emb.transpose(1,2)   # (N, emb_dim, L) for conv1d
            conv_out = self.conv(emb)  # (N, output_dim, L)
            pooled = self.pool(conv_out).squeeze(-1)  # (N, output_dim)
            return pooled.cpu()

    # make parameters iterable for optimizer
    def parameters(self, recurse=True):
        if self.use_pretrained:
            return list(self.esm_model.parameters()) + list(self.proj.parameters())
        else:
            return super().parameters(recurse=recurse)

class GatedAttention(nn.Module):
    """
    Gated attention mechanism from Ilse et al. (2018) for MIL.
    Produces attention weights (N, 1) and aggregated representation.
    """
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.attn_v = nn.Linear(in_dim, hidden_dim)
        self.attn_w = nn.Linear(in_dim, hidden_dim)
        self.attn_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (N_instances, feat_dim)
        A = torch.tanh(self.attn_v(x)) * torch.sigmoid(self.attn_w(x))  # (N, hidden)
        a = self.attn_out(A)  # (N, 1)
        a_soft = torch.softmax(a, dim=0)  # normalized across instances
        M = (a_soft * x).sum(dim=0)  # (feat_dim,)
        return M, a_soft

class MILModel(nn.Module):
    """
    Full MIL model:
      - input: instance embeddings (N, embed_dim), v_onehot (N, v_dim), j_onehot (N, j_dim)
      - instance projector: small MLP with CNN residual (optional)
      - gated attention aggregator -> repertoire representation -> classifier
    """
    def __init__(self, input_dim: int = 512, v_dim: int = 32, j_dim: int = 16, hidden_dim: int = 256):
        super().__init__()
        # instance projection after concatenation
        self.input_dim = input_dim + v_dim + j_dim
        self.instance_proj = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # gated attention
        self.attention = GatedAttention(in_dim=hidden_dim, hidden_dim=hidden_dim // 2)
        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, seq_embeds, v_onehot, j_onehot):
        """
        seq_embeds: torch.FloatTensor (N, embed_dim)  # CPU or device; ensure on same device as model
        v_onehot: (N, v_dim)
        j_onehot: (N, j_dim)
        returns: repertoire_logit (1,) and attention_scores (N,1)
        """
        # ensure tensors are on same device as model
        device = next(self.parameters()).device
        seq_embeds = seq_embeds.to(device)
        v_onehot = v_onehot.to(device)
        j_onehot = j_onehot.to(device)

        x = torch.cat([seq_embeds, v_onehot, j_onehot], dim=1)  # (N, input_dim)
        inst = self.instance_proj(x)  # (N, hidden)
        rep_repr, attn = self.attention(inst)  # rep_repr (hidden,), attn (N,1)
        logit = self.classifier(rep_repr)  # (1,)
        return logit, attn

# End of file
