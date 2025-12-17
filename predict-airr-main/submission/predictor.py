import os
import gc
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from submission.utils import (
    load_data_generator, load_full_dataset, get_repertoire_ids,
    generate_random_top_sequences_df, encode_sequence_atchley,
    GeneEncoder, RepertoireDataset, collate_fn
)


# ============================================================================
# NEURAL NETWORK MODULES
# ============================================================================

class SequenceEncoder(nn.Module):
    """CNN-based encoder for CDR3 sequences with multi-scale convolutions."""
    
    def __init__(self, input_dim: int = 5, hidden_dim: int = 96, dropout: float = 0.25):
        super().__init__()
        
        # Multi-scale CNN for motif detection
        self.conv3 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim // 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
        )
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.output_dim = hidden_dim
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        
        c3 = self.conv3(x)
        c5 = self.conv5(x)
        x = torch.cat([c3, c5], dim=1)  # (batch, hidden_dim, seq_len)
        
        # Global max pooling
        x = x.max(dim=2)[0]  # (batch, hidden_dim)
        x = self.projection(x)
        return x


class GeneEmbedding(nn.Module):
    """Embedding layer for V and J genes."""
    
    def __init__(self, n_v_genes: int, n_j_genes: int, embed_dim: int = 24):
        super().__init__()
        self.v_embedding = nn.Embedding(n_v_genes, embed_dim, padding_idx=0)
        self.j_embedding = nn.Embedding(n_j_genes, embed_dim, padding_idx=0)
        self.output_dim = embed_dim * 2
        
    def forward(self, v_idx, j_idx):
        v_emb = self.v_embedding(v_idx)
        j_emb = self.j_embedding(j_idx)
        return torch.cat([v_emb, j_emb], dim=-1)


class GatedAttention(nn.Module):
    """Gated attention mechanism for Multi-Instance Learning."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 96, dropout: float = 0.25):
        super().__init__()
        
        self.attention_V = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # x: (batch, n_instances, input_dim)
        V = self.attention_V(x)
        U = self.attention_U(x)
        A = self.attention_w(V * U)  # (batch, n_instances, 1)
        
        if mask is not None:
            A = A.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        
        attention_scores = F.softmax(A, dim=1)
        attention_scores = self.dropout(attention_scores)
        
        # Weighted sum
        bag_repr = torch.sum(attention_scores * x, dim=1)
        return bag_repr, attention_scores.squeeze(-1)


class MILClassifier(nn.Module):
    """Multi-Instance Learning classifier combining sequence and gene features."""
    
    def __init__(self, n_v_genes: int, n_j_genes: int, seq_hidden: int = 96,
                 gene_embed: int = 24, attention_hidden: int = 96, dropout: float = 0.25):
        super().__init__()
        
        self.seq_encoder = SequenceEncoder(input_dim=5, hidden_dim=seq_hidden, dropout=dropout)
        self.gene_encoder = GeneEmbedding(n_v_genes, n_j_genes, gene_embed)
        
        fused_dim = self.seq_encoder.output_dim + self.gene_encoder.output_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, attention_hidden),
            nn.LayerNorm(attention_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.attention = GatedAttention(attention_hidden, attention_hidden, dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(attention_hidden, attention_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attention_hidden, 1)
        )
        
    def forward(self, seq_encodings, v_indices, j_indices, mask):
        batch_size, n_seqs, seq_len, input_dim = seq_encodings.shape
        
        # Encode sequences
        seq_flat = seq_encodings.view(batch_size * n_seqs, seq_len, input_dim)
        seq_features = self.seq_encoder(seq_flat)
        seq_features = seq_features.view(batch_size, n_seqs, -1)
        
        # Encode genes
        gene_features = self.gene_encoder(v_indices, j_indices)
        
        # Fuse features
        fused = torch.cat([seq_features, gene_features], dim=-1)
        fused = self.fusion(fused)
        
        # Attention aggregation
        bag_repr, attention_scores = self.attention(fused, mask)
        
        # Classify
        logits = self.classifier(bag_repr).squeeze(-1)
        return logits, attention_scores


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class CombinedLoss(nn.Module):
    """Combined loss with focal loss and AUC optimization."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits, labels, attention_scores=None, mask=None):
        # Focal loss
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        p_t = probs * labels + (1 - probs) * (1 - labels)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = (self.alpha * focal_weight * ce_loss).mean()
        
        return focal_loss


class ImmuneStatePredictor:
    """
    A template for predicting immune states from TCR repertoire data.

    Participants should implement the logic for training, prediction, and
    sequence identification within this class.
    """

    def __init__(self, n_jobs: int = 1, device: str = 'cpu', **kwargs):
        """
        Initializes the predictor.

        Args:
            n_jobs (int): Number of CPU cores to use for parallel processing.
            device (str): The device to use for computation (e.g., 'cpu', 'cuda').
            **kwargs: Additional hyperparameters for the model.
        """
        total_cores = os.cpu_count()
        if n_jobs == -1:
            self.n_jobs = total_cores
        else:
            self.n_jobs = min(n_jobs, total_cores)
        self.device = device
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: 'cuda' was requested but is not available. Falling back to 'cpu'.")
            self.device = 'cpu'
        else:
            self.device = device
        # Hyperparameters
        self.max_clonotypes = kwargs.get('max_clonotypes', 2000)
        self.max_seq_len = kwargs.get('max_seq_len', 25)
        self.batch_size = kwargs.get('batch_size', 24)
        self.n_epochs = kwargs.get('n_epochs', 12)
        self.n_folds = kwargs.get('n_folds', 2)
        self.learning_rate = kwargs.get('learning_rate', 1e-3)
        self.weight_decay = kwargs.get('weight_decay', 1e-4)
        self.seq_hidden_dim = kwargs.get('seq_hidden_dim', 96)
        self.gene_embed_dim = kwargs.get('gene_embed_dim', 24)
        self.attention_hidden_dim = kwargs.get('attention_hidden_dim', 96)
        self.dropout = kwargs.get('dropout', 0.25)
        
        # Model components (initialized during fit)
        self.models = []  # List of trained models for ensemble
        self.gene_encoder = None
        self.important_sequences_ = None
        self.sequence_scores_ = {}

    def fit(self, train_dir_path: str):
        """
        Trains the model on the provided training data.

        Args:
            train_dir_path (str): Path to the directory with training TSV files.

        Returns:
            self: The fitted predictor instance.
        """

        # --- your code starts here ---
        print(f"Loading training data from {train_dir_path}...")
        
        # Load full dataset
        full_df = load_full_dataset(train_dir_path)
        
        # Initialize gene encoder with all genes
        self.gene_encoder = GeneEncoder()
        all_v_genes = full_df['v_call'].dropna().unique().tolist()
        all_j_genes = full_df['j_call'].dropna().unique().tolist()
        self.gene_encoder.fit(all_v_genes, all_j_genes)
        
        # Get repertoire info
        repertoire_ids = full_df['repertoire_id'].unique()
        labels = full_df.groupby('repertoire_id')['label'].first().values
        
        print(f"Training on {len(repertoire_ids)} repertoires, {len(full_df)} total clonotypes")
        print(f"Class distribution: {np.mean(labels):.2%} positive")
        
        # Cross-validation training
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        self.models = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(repertoire_ids, labels)):
            print(f"\n--- Fold {fold + 1}/{self.n_folds} ---")
            
            train_rep_ids = repertoire_ids[train_idx]
            val_rep_ids = repertoire_ids[val_idx]
            
            train_df = full_df[full_df['repertoire_id'].isin(train_rep_ids)]
            val_df = full_df[full_df['repertoire_id'].isin(val_rep_ids)]
            
            # Create datasets
            train_dataset = RepertoireDataset(
                train_df, self.gene_encoder, 
                max_clonotypes=self.max_clonotypes, 
                max_seq_len=self.max_seq_len
            )
            val_dataset = RepertoireDataset(
                val_df, self.gene_encoder,
                max_clonotypes=self.max_clonotypes,
                max_seq_len=self.max_seq_len
            )
            
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, 
                shuffle=True, collate_fn=collate_fn, num_workers=0
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size,
                shuffle=False, collate_fn=collate_fn, num_workers=0
            )
            
            # Initialize model
            model = MILClassifier(
                n_v_genes=self.gene_encoder.n_v_genes,
                n_j_genes=self.gene_encoder.n_j_genes,
                seq_hidden=self.seq_hidden_dim,
                gene_embed=self.gene_embed_dim,
                attention_hidden=self.attention_hidden_dim,
                dropout=self.dropout
            ).to(self.device)
            
            optimizer = AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
            criterion = CombinedLoss()
            
            best_auc = 0.0
            best_state = None
            
            for epoch in range(self.n_epochs):
                # Training
                model.train()
                train_loss = 0.0
                
                for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                    seq_enc = batch['seq_encodings'].to(self.device)
                    v_idx = batch['v_indices'].to(self.device)
                    j_idx = batch['j_indices'].to(self.device)
                    mask = batch['mask'].to(self.device)
                    labels_batch = batch['labels'].float().to(self.device)
                    
                    optimizer.zero_grad()
                    logits, attn = model(seq_enc, v_idx, j_idx, mask)
                    loss = criterion(logits, labels_batch, attn, mask)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    train_loss += loss.item()
                
                scheduler.step()
                
                # Validation
                model.eval()
                val_preds = []
                val_labels = []
                
                with torch.no_grad():
                    for batch in val_loader:
                        seq_enc = batch['seq_encodings'].to(self.device)
                        v_idx = batch['v_indices'].to(self.device)
                        j_idx = batch['j_indices'].to(self.device)
                        mask = batch['mask'].to(self.device)
                        
                        logits, _ = model(seq_enc, v_idx, j_idx, mask)
                        probs = torch.sigmoid(logits).cpu().numpy()
                        val_preds.extend(probs)
                        val_labels.extend(batch['labels'].numpy())
                
                val_auc = roc_auc_score(val_labels, val_preds)
                print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Val AUC={val_auc:.4f}")
                
                if val_auc > best_auc:
                    best_auc = val_auc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            # Load best model
            model.load_state_dict(best_state)
            model.eval()
            self.models.append(model)
            print(f"Fold {fold+1} best AUC: {best_auc:.4f}")
        
        # Identify important sequences using attention scores
        self._collect_sequence_scores(full_df)
        self.important_sequences_ = self.identify_associated_sequences(
            top_k=50000, dataset_name=os.path.basename(train_dir_path)
        )
        
        # --- your code ends here ---
        print("\nTraining complete.")
        return self
    
    def _collect_sequence_scores(self, df: pd.DataFrame):
        """Collect attention scores for all sequences across models."""
        print("Collecting sequence attention scores...")
        
        repertoire_ids = df['repertoire_id'].unique()
        
        for rep_id in tqdm(repertoire_ids, desc="Scoring sequences"):
            rep_df = df[df['repertoire_id'] == rep_id]
            label = rep_df['label'].iloc[0]
            
            # Sample sequences
            if len(rep_df) > self.max_clonotypes:
                rep_df = rep_df.nlargest(self.max_clonotypes, 'duplicate_count')
            
            sequences = rep_df['junction_aa'].tolist()
            v_calls = rep_df['v_call'].tolist()
            j_calls = rep_df['j_call'].tolist()
            
            # Encode
            seq_encodings = []
            for seq in sequences:
                enc = encode_sequence_atchley(seq, self.max_seq_len)
                seq_encodings.append(enc)
            
            seq_tensor = torch.tensor(np.array(seq_encodings), dtype=torch.float32).unsqueeze(0).to(self.device)
            v_tensor = torch.tensor([self.gene_encoder.encode_v(v) for v in v_calls], dtype=torch.long).unsqueeze(0).to(self.device)
            j_tensor = torch.tensor([self.gene_encoder.encode_j(j) for j in j_calls], dtype=torch.long).unsqueeze(0).to(self.device)
            mask = torch.ones(1, len(sequences), dtype=torch.float32).to(self.device)
            
            # Get attention scores from all models
            all_scores = []
            with torch.no_grad():
                for model in self.models:
                    _, attn = model(seq_tensor, v_tensor, j_tensor, mask)
                    all_scores.append(attn.cpu().numpy()[0])
            
            avg_scores = np.mean(all_scores, axis=0)
            
            # Weight by label (positive samples contribute positively)
            label_weight = 1.0 if label == 1 else -0.5
            
            for i, (seq, v, j) in enumerate(zip(sequences, v_calls, j_calls)):
                key = (seq, v, j)
                if key not in self.sequence_scores_:
                    self.sequence_scores_[key] = []
                self.sequence_scores_[key].append(avg_scores[i] * label_weight)

        # --- your code ends here ---
        print("Training complete.")
        return self

    def predict_proba(self, test_dir_path: str) -> pd.DataFrame:
        """
        Predicts probabilities for examples in the provided path.

        Args:
            test_dir_path (str): Path to the directory with test TSV files.

        Returns:
            pd.DataFrame: A DataFrame with 'ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call' columns.
        """
        print(f"Making predictions for data in {test_dir_path}...")
        if not self.models:
            raise RuntimeError("The model has not been fitted yet. Please call `fit` first.")

        # --- your code starts here ---
        
        # Load test data
        full_df = load_full_dataset(test_dir_path)
        repertoire_ids = full_df['repertoire_id'].unique()
        
        probabilities = []
        
        for rep_id in tqdm(repertoire_ids, desc="Predicting"):
            rep_df = full_df[full_df['repertoire_id'] == rep_id]
            
            # Sample sequences
            if len(rep_df) > self.max_clonotypes:
                rep_df = rep_df.nlargest(self.max_clonotypes, 'duplicate_count')
            
            sequences = rep_df['junction_aa'].tolist()
            v_calls = rep_df['v_call'].tolist()
            j_calls = rep_df['j_call'].tolist()
            
            # Encode sequences
            seq_encodings = []
            for seq in sequences:
                enc = encode_sequence_atchley(seq, self.max_seq_len)
                seq_encodings.append(enc)
            
            seq_tensor = torch.tensor(np.array(seq_encodings), dtype=torch.float32).unsqueeze(0).to(self.device)
            v_tensor = torch.tensor([self.gene_encoder.encode_v(v) for v in v_calls], dtype=torch.long).unsqueeze(0).to(self.device)
            j_tensor = torch.tensor([self.gene_encoder.encode_j(j) for j in j_calls], dtype=torch.long).unsqueeze(0).to(self.device)
            mask = torch.ones(1, len(sequences), dtype=torch.float32).to(self.device)
            
            # Ensemble prediction
            all_probs = []
            with torch.no_grad():
                for model in self.models:
                    logits, _ = model(seq_tensor, v_tensor, j_tensor, mask)
                    prob = torch.sigmoid(logits).cpu().numpy()[0]
                    all_probs.append(prob)
            
            avg_prob = np.mean(all_probs)
            probabilities.append(avg_prob)

        # --- your code ends here ---

        predictions_df = pd.DataFrame({
            'ID': repertoire_ids,
            'dataset': [os.path.basename(test_dir_path)] * len(repertoire_ids),
            'label_positive_probability': probabilities
        })

        # to enable compatibility with the expected output format that includes junction_aa, v_call, j_call columns
        predictions_df['junction_aa'] = -999.0
        predictions_df['v_call'] = -999.0
        predictions_df['j_call'] = -999.0

        predictions_df = predictions_df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

        print(f"Prediction complete on {len(repertoire_ids)} examples in {test_dir_path}.")
        return predictions_df

    def identify_associated_sequences(self, dataset_name: str, top_k: int = 50000) -> pd.DataFrame:
        """
        Identifies the top "k" important sequences (rows) from the training data that best explain the labels.

        Args:
            top_k (int): The number of top sequences to return (based on some scoring mechanism).

        Returns:
            pd.DataFrame: A DataFrame with 'ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call' columns.
        """

        # --- your code starts here ---
        # Return the top k sequences, sorted based on some form of importance score.
        # Example:
        # all_sequences_scored = self._score_all_sequences()
        all_sequences_scored = generate_random_top_sequences_df(n_seq=top_k)  # Replace with your way of identifying top k sequences
        top_sequences_df = all_sequences_scored.nlargest(top_k, 'importance_score')
        top_sequences_df = top_sequences_df[['junction_aa', 'v_call', 'j_call']]
        top_sequences_df['dataset'] = dataset_name
        top_sequences_df['ID'] = range(1, len(top_sequences_df)+1)
        top_sequences_df['ID'] = top_sequences_df['dataset'] + '_seq_top_' + top_sequences_df['ID'].astype(str)
        top_sequences_df['label_positive_probability'] = -999.0 # to enable compatibility with the expected output format
        top_sequences_df = top_sequences_df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

        # --- your code ends here ---
        return top_sequences_df