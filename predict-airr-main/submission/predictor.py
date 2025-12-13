# submission/predictor.py
"""
Enhanced end-to-end pipeline for the predict-airr template:
- MIL model (ESM optional / fallback)
- Precompute & cache sequence embeddings for speed
- Gated attention MIL aggregator -> repertoire probability + attention-based ranking
- Checkpointing, freezing encoder, prefiltering, and robust dataset handling
"""

import os
import sys
import argparse
import math
import random
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

# Try template imports; fallback to local utils if needed
try:
    from submission.utils import load_data_generator, load_full_dataset, get_repertoire_ids, generate_random_top_sequences_df
except Exception:
    try:
        from utils import load_data_generator, load_full_dataset, get_repertoire_ids, generate_random_top_sequences_df  # type: ignore
    except Exception:
        # minimal fallbacks (used only if repo utils are missing)
        def load_full_dataset(path: str) -> pd.DataFrame:
            # attempt to read tsvs in the directory
            files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.tsv')]
            if not files:
                raise FileNotFoundError(f"No TSV files found in {path}")
            dfs = [pd.read_csv(f, sep='\t', low_memory=False) for f in files]
            return pd.concat(dfs, ignore_index=True)

        def get_repertoire_ids(path: str) -> List[str]:
            df = load_full_dataset(path)
            for c in ['repertoire_id', 'repertoire', 'sample_id', 'id']:
                if c in df.columns:
                    return sorted(df[c].unique().tolist())
            # fallback: generate artificial ids
            return [f"rep_{i}" for i in range(100)]

        def generate_random_top_sequences_df(n_seq: int = 1000) -> pd.DataFrame:
            import random, string
            rows = []
            for i in range(n_seq):
                seq = ''.join(random.choices(list("ACDEFGHIKLMNPQRSTVWY"), k=12))
                rows.append({'junction_aa': seq, 'v_call': 'V1', 'j_call': 'J1', 'importance_score': random.random()})
            return pd.DataFrame(rows)

# ----------------------------
# Determinism & device
# ----------------------------
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Utils: robust column detection
# ----------------------------
def _col(df: pd.DataFrame, candidates: List[str], default: Optional[str] = None) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    if default is not None:
        return default
    raise KeyError(f"None of {candidates} found. Columns present: {list(df.columns)}")

# ----------------------------
# Encoder + MIL model
# ----------------------------
try:
    import esm  # type: ignore
    _ESM_AVAILABLE = True
except Exception:
    _ESM_AVAILABLE = False

class ESMFallbackEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256, use_esm: bool = False, device: str = "cpu"):
        super().__init__()
        self.embed_dim = embed_dim
        self.device = device
        self.use_esm = use_esm and _ESM_AVAILABLE
        if self.use_esm:
            # load smaller ESM if available
            self.esm_model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            self.esm_model = self.esm_model.eval().to(self.device)
            self.proj = nn.Linear(self.esm_model.args.embed_dim, embed_dim)
        else:
            aa_letters = "ACDEFGHIKLMNPQRSTVWYBXZJUO"
            self.aa_to_idx = {c: i+1 for i, c in enumerate(aa_letters)}
            vocab_size = len(self.aa_to_idx) + 2
            self.tok_emb = nn.Embedding(vocab_size, 64, padding_idx=0)
            self.conv = nn.Conv1d(in_channels=64, out_channels=embed_dim, kernel_size=5, padding=2)
            self.pool = nn.AdaptiveAvgPool1d(1)

    def encode_batch(self, seqs: List[str]) -> torch.FloatTensor:
        if self.use_esm:
            if len(seqs) == 0:
                return torch.zeros((0, self.embed_dim), dtype=torch.float32)
            batch = [(f"s{i}", s) for i, s in enumerate(seqs)]
            batch_converter = self.alphabet.get_batch_converter()
            _, _, toks = batch_converter(batch)
            toks = toks.to(self.device)
            with torch.no_grad():
                reps = self.esm_model(toks, repr_layers=[len(self.esm_model.layers)], return_contacts=False)
                token_reprs = reps["representations"][len(self.esm_model.layers)]
                mask = (toks != self.alphabet.padding_idx).float().unsqueeze(-1)
                summed = (token_reprs * mask).sum(1)
                lens = mask.sum(1)
                pooled = summed / (lens + 1e-8)
                emb = self.proj(pooled)
            return emb.cpu()
        else:
            if len(seqs) == 0:
                return torch.zeros((0, self.embed_dim), dtype=torch.float32)
            max_len = max(len(s) for s in seqs)
            toks = []
            for s in seqs:
                idxs = [self.aa_to_idx.get(ch, 0) for ch in s]
                idxs = idxs + [0] * (max_len - len(idxs))
                toks.append(idxs)
            toks = torch.tensor(toks, dtype=torch.long, device=self.device)
            emb = self.tok_emb(toks)  # (N, L, emb)
            emb = emb.transpose(1, 2)
            conv = self.conv(emb)
            pooled = self.pool(conv).squeeze(-1)
            return pooled.cpu()

    def parameters(self, recurse=True):
        if self.use_esm:
            return list(self.esm_model.parameters()) + list(self.proj.parameters())
        else:
            return super().parameters(recurse=recurse)

class GatedAttention(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.attn_v = nn.Linear(in_dim, hidden_dim)
        self.attn_w = nn.Linear(in_dim, hidden_dim)
        self.attn_out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        A = torch.tanh(self.attn_v(x)) * torch.sigmoid(self.attn_w(x))
        a = self.attn_out(A)
        a_soft = torch.softmax(a, dim=0)
        M = (a_soft * x).sum(dim=0)
        return M, a_soft

class MILModel(nn.Module):
    def __init__(self, seq_emb_dim: int, v_dim: int, j_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = seq_emb_dim + v_dim + j_dim
        self.inst_proj = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.attn = GatedAttention(in_dim=hidden_dim, hidden_dim=max(16, hidden_dim // 2))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, seq_embs: torch.Tensor, v_onehot: torch.Tensor, j_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        seq_embs = seq_embs.to(device)
        v_onehot = v_onehot.to(device)
        j_onehot = j_onehot.to(device)
        x = torch.cat([seq_embs, v_onehot, j_onehot], dim=1)
        inst = self.inst_proj(x)
        rep_repr, attn = self.attn(inst)
        logit = self.classifier(rep_repr)
        return logit.view(-1), attn.view(-1)

# ----------------------------
# Helper: build repertoires from a long-format df
# ----------------------------
def build_repertoires_from_df(df: pd.DataFrame, prefilter_top: Optional[int] = None) -> List[dict]:
    seq_col = _col(df, ['junction_aa', 'cdr3', 'cdr3_aa', 'sequence'], default='junction_aa')
    v_col = _col(df, ['v_call', 'v_gene', 'v'], default='v_call')
    j_col = _col(df, ['j_call', 'j_gene', 'j'], default='j_call')
    count_col = next((c for c in ['template_count', 'duplicate_count', 'count', 'templates'] if c in df.columns), None)
    label_col = next((c for c in ['label', 'label_positive', 'y', 'is_case'] if c in df.columns), None)
    rep_col = next((c for c in ['repertoire_id', 'repertoire', 'sample_id', 'subject_id', 'id'] if c in df.columns), None)
    if rep_col is None:
        raise KeyError("No repertoire identifier column found.")

    repertoires = []
    grouped = df.groupby(rep_col)
    for rid, g in grouped:
        label = 0
        if label_col and label_col in g.columns:
            try:
                vals = g[label_col].unique()
                label = int(vals[0]) if len(vals) > 0 and not pd.isna(vals[0]) else 0
            except Exception:
                label = 0
        # optionally prefilter by top counts
        if count_col and prefilter_top is not None:
            # keep top prefilter_top rows by count
            g = g.sort_values(count_col, ascending=False).head(prefilter_top)
        seqs = []
        for _, row in g.iterrows():
            seqs.append({
                'junction_aa': str(row.get(seq_col, "")),
                'v_call': str(row.get(v_col, "UNK")),
                'j_call': str(row.get(j_col, "UNK")),
                'count': int(row[count_col]) if count_col is not None and not pd.isna(row[count_col]) else 1
            })
        repertoires.append({'repertoire_id': str(rid), 'label': int(label), 'sequences': seqs})
    return repertoires

# ----------------------------
# Embedding precompute cache
# ----------------------------
def precompute_embeddings_cache(all_seqs: List[str], encoder: ESMFallbackEncoder, cache_path: str, batch_size: int = 256) -> Dict[str, np.ndarray]:
    """
    Compute embeddings for unique sequences and store as a dict {seq: ndarray}.
    Will save to cache_path (npz) for reuse.
    """
    if os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            emb_map = {k: data[k].tolist() if isinstance(data[k].tolist(), list) else data[k] for k in data.files}
            # convert arrays to numpy arrays if stored oddly
            emb_map = {k: np.array(v) for k, v in emb_map.items()}
            print(f"Loaded cached embeddings from {cache_path} ({len(emb_map)} sequences).")
            return emb_map
        except Exception:
            print(f"Could not load cache at {cache_path}, will recompute.")

    unique = sorted(set(all_seqs))
    emb_map: Dict[str, np.ndarray] = {}
    # batch encode
    for i in tqdm(range(0, len(unique), batch_size), desc="Computing embeddings"):
        batch = unique[i:i+batch_size]
        with torch.no_grad():
            embs = encoder.encode_batch(batch)  # CPU tensor
        # ensure numpy arrays
        for s, e in zip(batch, embs.numpy()):
            emb_map[s] = e.astype(np.float32)
    # save as npz (store each sequence as an array key; keys must be valid filenames but np.savez supports strings)
    try:
        # np.savez supports mapping but to avoid giant file we use savez_compressed
        np.savez_compressed(cache_path, **{k: v for k, v in emb_map.items()})
        print(f"Saved embedding cache to {cache_path}")
    except Exception as e:
        print(f"Warning: failed to save cache to {cache_path}: {e}")
    return emb_map

# ----------------------------
# Predictor class with enhancements
# ----------------------------
class ImmuneStatePredictor:
    def __init__(self, n_jobs: int = 1, device: str = DEVICE, cfg: Optional[dict] = None):
        total_cores = os.cpu_count() or 1
        self.n_jobs = total_cores if n_jobs == -1 else min(n_jobs, total_cores)
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: requested cuda but not available; falling back to cpu.")
            device = 'cpu'
        self.device = device
        self.cfg = cfg or {}
        self.seq_embed_dim = self.cfg.get('seq_embed_dim', 256)
        self.hidden_dim = self.cfg.get('hidden_dim', 256)
        self.use_esm = self.cfg.get('use_esm', False) and _ESM_AVAILABLE
        self.lambda_sparsity = self.cfg.get('lambda_sparsity', 1e-3)
        self.lr = self.cfg.get('lr', 1e-4)
        self.epochs = self.cfg.get('epochs', 3)
        self.batch_size = self.cfg.get('batch_size', 1)
        self.prefilter_top = self.cfg.get('prefilter_top', None)
        self.embed_cache_path = self.cfg.get('embed_cache_path', '/kaggle/working/embeddings_cache.npz')
        self.freeze_encoder = self.cfg.get('freeze_encoder', False)

        # instantiate encoder
        self.encoder = ESMFallbackEncoder(embed_dim=self.seq_embed_dim, use_esm=self.use_esm, device=self.device)
        self.model: Optional[MILModel] = None
        self.v_map: Dict[str, int] = {}
        self.j_map: Dict[str, int] = {}
        self.important_sequences_: Optional[pd.DataFrame] = None
        self.embedding_cache: Optional[Dict[str, np.ndarray]] = None

    def fit(self, train_dir_path: str, ckpt_path: Optional[str] = None, precompute_embeddings: bool = False):
        print("Loading training data...")
        full_df = load_full_dataset(train_dir_path)
        # gather unique sequences for cache if requested (include test? only train)
        if precompute_embeddings:
            all_seqs = list(full_df[_col(full_df, ['junction_aa', 'cdr3', 'cdr3_aa', 'sequence'])].astype(str).unique())
            # precompute using current encoder; if ESM is used this will be slower but cached
            self.embedding_cache = precompute_embeddings_cache(all_seqs, self.encoder, self.embed_cache_path, batch_size=256)

        repertoires = build_repertoires_from_df(full_df, prefilter_top=self.prefilter_top)

        # build v/j maps
        v_set, j_set = set(), set()
        for r in repertoires:
            for s in r['sequences']:
                v_set.add(s.get('v_call', 'UNK'))
                j_set.add(s.get('j_call', 'UNK'))
        self.v_map = {g: i for i, g in enumerate(sorted(list(v_set)))}
        self.j_map = {g: i for i, g in enumerate(sorted(list(j_set)))}
        v_dim, j_dim = len(self.v_map), len(self.j_map)

        self.model = MILModel(seq_emb_dim=self.seq_embed_dim, v_dim=v_dim, j_dim=j_dim, hidden_dim=self.hidden_dim).to(self.device)

        # optimizer: include encoder params if not frozen
        params = list(self.model.parameters())
        if not self.freeze_encoder:
            try:
                params += list(self.encoder.parameters())
            except Exception:
                pass

        optimizer = optim.AdamW(params, lr=self.lr)
        bce_loss = nn.BCEWithLogitsLoss()

        print(f"Training for {self.epochs} epochs on {len(repertoires)} repertoires (prefilter_top={self.prefilter_top}).")
        best_loss = float('inf')
        for epoch in range(self.epochs):
            random.shuffle(repertoires)
            epoch_loss = 0.0
            self.model.train()
            for repo in tqdm(repertoires, desc=f"Epoch {epoch+1}/{self.epochs}"):
                optimizer.zero_grad()
                seqs = repo['sequences']
                if len(seqs) == 0:
                    continue
                # optionally limit per repertoire size (prefilter_top already applied)
                junctions = [s['junction_aa'] for s in seqs]
                v_calls = [s.get('v_call', 'UNK') for s in seqs]
                j_calls = [s.get('j_call', 'UNK') for s in seqs]

                # get embeddings (prefer cache if available)
                if self.embedding_cache is not None:
                    emb_list = []
                    for s in junctions:
                        emb = self.embedding_cache.get(s)
                        if emb is None:
                            # compute single
                            with torch.no_grad():
                                emb = self.encoder.encode_batch([s]).numpy()[0]
                            self.embedding_cache[s] = emb
                        emb_list.append(emb)
                    seq_embs = torch.tensor(np.stack(emb_list, axis=0), dtype=torch.float32)
                else:
                    seq_embs = self.encoder.encode_batch(junctions)

                if seq_embs.numel() == 0:
                    continue

                # v/j one-hot
                v_onehot = torch.zeros((len(v_calls), len(self.v_map)), dtype=torch.float32)
                j_onehot = torch.zeros((len(j_calls), len(self.j_map)), dtype=torch.float32)
                for i, (v, j) in enumerate(zip(v_calls, j_calls)):
                    v_onehot[i, self.v_map.get(v, 0)] = 1.0
                    j_onehot[i, self.j_map.get(j, 0)] = 1.0

                seq_embs = seq_embs.to(self.device)
                v_onehot = v_onehot.to(self.device)
                j_onehot = j_onehot.to(self.device)

                logit, attn = self.model(seq_embs, v_onehot, j_onehot)
                label = torch.tensor(repo.get('label', 0), dtype=torch.float32, device=self.device)
                loss_bce = bce_loss(logit.view(-1), label.view(-1))

                p = attn + 1e-12
                entropy = - (p * torch.log(p)).sum()
                loss_sparsity = self.lambda_sparsity * entropy
                loss = loss_bce + loss_sparsity

                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())

            print(f"Epoch {epoch+1}/{self.epochs} total_loss={epoch_loss:.4f}")
            # simple checkpoint by loss
            if ckpt_path is not None:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    self.save_checkpoint(ckpt_path)
                    print(f"Saved best checkpoint to {ckpt_path} (loss={best_loss:.4f})")
        print("Training complete. Computing ranked important sequences.")
        # compute important sequences and store
        self.important_sequences_ = self.identify_associated_sequences(train_df=full_df, top_k=50000, dataset_name=os.path.basename(train_dir_path))
        return self

    def save_checkpoint(self, path: str):
        d = {
            'model_state': self.model.state_dict() if self.model else None,
            'v_map': self.v_map,
            'j_map': self.j_map,
            'cfg': self.cfg
        }
        try:
            torch.save(d, path)
        except Exception as e:
            print(f"Checkpoint save failed: {e}")

    def load_checkpoint(self, path: str):
        d = torch.load(path, map_location=self.device)
        self.v_map = d.get('v_map', {})
        self.j_map = d.get('j_map', {})
        cfg = d.get('cfg', {})
        # re-init model if needed
        v_dim = len(self.v_map)
        j_dim = len(self.j_map)
        self.model = MILModel(seq_emb_dim=self.seq_embed_dim, v_dim=v_dim, j_dim=j_dim, hidden_dim=self.hidden_dim).to(self.device)
        self.model.load_state_dict(d['model_state'])
        print(f"Loaded model checkpoint from {path}")

    def _prepare_rep_tensors_from_meta(self, sequences: List[dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[dict]]:
        junctions = [s['junction_aa'] for s in sequences]
        v_calls = [s.get('v_call', 'UNK') for s in sequences]
        j_calls = [s.get('j_call', 'UNK') for s in sequences]

        if self.embedding_cache is not None:
            emb_list = []
            for s in junctions:
                emb = self.embedding_cache.get(s)
                if emb is None:
                    with torch.no_grad():
                        emb = self.encoder.encode_batch([s]).numpy()[0]
                    self.embedding_cache[s] = emb
                emb_list.append(emb)
            seq_embs = torch.tensor(np.stack(emb_list, axis=0), dtype=torch.float32)
        else:
            seq_embs = self.encoder.encode_batch(junctions)

        v_onehot = torch.zeros((len(v_calls), len(self.v_map)), dtype=torch.float32)
        j_onehot = torch.zeros((len(j_calls), len(self.j_map)), dtype=torch.float32)
        for i, (v, j) in enumerate(zip(v_calls, j_calls)):
            v_onehot[i, self.v_map.get(v, 0)] = 1.0
            j_onehot[i, self.j_map.get(j, 0)] = 1.0

        meta = [{'junction_aa': a, 'v_call': v, 'j_call': j} for a, v, j in zip(junctions, v_calls, j_calls)]
        return seq_embs, v_onehot, j_onehot, meta

    def predict_proba(self, test_dir_path: str) -> pd.DataFrame:
        print("Running inference on test set...")
        try:
            df_test = load_full_dataset(test_dir_path)
            test_reps = build_repertoires_from_df(df_test, prefilter_top=self.prefilter_top)
        except Exception:
            print("Could not load full test dataset; trying get_repertoire_ids fallback.")
            rep_ids = get_repertoire_ids(test_dir_path)
            test_reps = [{'repertoire_id': rid, 'label': 0, 'sequences': []} for rid in rep_ids]

        results = []
        self.model.eval()
        with torch.no_grad():
            for rep in tqdm(test_reps, desc="Predicting repertoires"):
                seqs = rep.get('sequences', [])
                if not seqs:
                    prob = 0.0
                else:
                    seq_embs, v_onehot, j_onehot, _ = self._prepare_rep_tensors_from_meta(seqs)
                    if seq_embs.numel() == 0:
                        prob = 0.0
                    else:
                        seq_embs = seq_embs.to(self.device)
                        v_onehot = v_onehot.to(self.device)
                        j_onehot = j_onehot.to(self.device)
                        logit, attn = self.model(seq_embs, v_onehot, j_onehot)
                        prob = float(torch.sigmoid(logit).cpu().numpy().item())
                results.append({
                    'ID': rep.get('repertoire_id', 'unknown'),
                    'dataset': os.path.basename(test_dir_path),
                    'label_positive_probability': prob,
                    'junction_aa': -999.0,
                    'v_call': -999.0,
                    'j_call': -999.0
                })
        pred_df = pd.DataFrame(results, columns=['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call'])
        print(f"Completed inference for {len(pred_df)} repertoires.")
        return pred_df

    def identify_associated_sequences(self, train_df: Optional[pd.DataFrame] = None, top_k: int = 50000, dataset_name: str = "train") -> pd.DataFrame:
        print("Computing sequence importance ranking...")
        if train_df is None:
            print("No train_df provided; falling back to random generator.")
            all_df = generate_random_top_sequences_df(n_seq=top_k)
            top_df = all_df.nlargest(top_k, 'importance_score')
            top_df = top_df[['junction_aa', 'v_call', 'j_call']]
            top_df['dataset'] = dataset_name
            top_df['ID'] = [f"{dataset_name}_seq_top_{i+1}" for i in range(len(top_df))]
            top_df['label_positive_probability'] = -999.0
            return top_df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

        reps = build_repertoires_from_df(train_df, prefilter_top=self.prefilter_top)
        seq_scores: Dict[Tuple[str, str, str], List[float]] = {}
        self.model.eval()
        with torch.no_grad():
            for rep in tqdm(reps, desc="Scoring sequences"):
                seqs = rep['sequences']
                if not seqs:
                    continue
                seq_embs, v_onehot, j_onehot, meta = self._prepare_rep_tensors_from_meta(seqs)
                if seq_embs.numel() == 0:
                    continue
                seq_embs = seq_embs.to(self.device)
                v_onehot = v_onehot.to(self.device)
                j_onehot = j_onehot.to(self.device)
                logit, attn = self.model(seq_embs, v_onehot, j_onehot)
                attn_list = attn.cpu().numpy().tolist()
                for m, a in zip(meta, attn_list):
                    key = (m['junction_aa'], m['v_call'], m['j_call'])
                    seq_scores.setdefault(key, []).append(float(a))
        rows = []
        for (junction, v_call, j_call), scores in seq_scores.items():
            rows.append({'junction_aa': junction, 'v_call': v_call, 'j_call': j_call, 'score': float(np.mean(scores))})
        if not rows:
            print("No sequences were scored; falling back to random generator.")
            all_df = generate_random_top_sequences_df(n_seq=top_k)
            top_df = all_df.nlargest(top_k, 'importance_score')
            top_df['dataset'] = dataset_name
            top_df['ID'] = [f"{dataset_name}_seq_top_{i+1}" for i in range(len(top_df))]
            top_df['label_positive_probability'] = -999.0
            return top_df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

        seq_df = pd.DataFrame(rows).sort_values('score', ascending=False).head(top_k).reset_index(drop=True)
        seq_df['dataset'] = dataset_name
        seq_df['ID'] = [f"{dataset_name}_seq_top_{i+1}" for i in range(len(seq_df))]
        seq_df['label_positive_probability'] = -999.0
        out_df = seq_df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]
        print(f"Produced top-{min(top_k, len(out_df))} sequences.")
        return out_df

    def export_submission(self, test_df: pd.DataFrame, top_sequences_df: pd.DataFrame, out_path: str):
        cols = ['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']
        test_df = test_df[cols].copy()
        top_sequences_df = top_sequences_df[cols].copy()
        final = pd.concat([test_df, top_sequences_df], axis=0, ignore_index=True)
        final.to_csv(out_path, index=False)
        print(f"Wrote submission to {out_path}")

# ----------------------------
# CLI
# ----------------------------
def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", type=str, required=True, help="Path to training directory.")
    p.add_argument("--test-dir", type=str, required=True, help="Path to test directory.")
    p.add_argument("--out", type=str, default="submission.csv", help="Output submission CSV path.")
    p.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    p.add_argument("--use-esm", action="store_true", help="Enable ESM encoder if available.")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size (repertoires per step).")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    p.add_argument("--no-train", action="store_true", help="Skip training; load checkpoint if provided.")
    p.add_argument("--ckpt-path", type=str, default=None, help="Checkpoint path for saving / loading model.")
    p.add_argument("--precompute-embeddings", action="store_true", help="Precompute embeddings for all unique sequences and cache.")
    p.add_argument("--embed-cache-path", type=str, default="/kaggle/working/embeddings_cache.npz", help="Path to save/load embedding cache.")
    p.add_argument("--freeze-encoder", action="store_true", help="Freeze encoder weights during training.")
    p.add_argument("--prefilter-top", type=int, default=None, help="Per-repertoire: keep only top-N sequences by count to reduce size.")
    return p.parse_args(argv)

def main(argv):
    args = parse_args(argv)
    cfg = {
        'seq_embed_dim': 256,
        'hidden_dim': 256,
        'use_esm': args.use_esm,
        'lambda_sparsity': 1e-3,
        'lr': args.lr,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'prefilter_top': args.prefilter_top,
        'embed_cache_path': args.embed_cache_path,
        'freeze_encoder': args.freeze_encoder
    }
    predictor = ImmuneStatePredictor(n_jobs=1, device=DEVICE, cfg=cfg)

    if args.no_train:
        if args.ckpt_path and os.path.exists(args.ckpt_path):
            predictor.load_checkpoint(args.ckpt_path)
        else:
            print("No checkpoint provided; proceeding without training (model uninitialized).")

    if not args.no_train:
        predictor.fit(args.train_dir, ckpt_path=args.ckpt_path, precompute_embeddings=args.precompute_embeddings)

    test_preds = predictor.predict_proba(args.test_dir)
    # use train df to identify top sequences if available
    try:
        train_df = load_full_dataset(args.train_dir)
    except Exception:
        train_df = None
    top_seqs = predictor.identify_associated_sequences(train_df=train_df, top_k=50000, dataset_name=os.path.basename(args.train_dir))
    predictor.export_submission(test_preds, top_seqs, args.out)
    print("All done.")

if __name__ == "__main__":
    main(sys.argv[1:])
