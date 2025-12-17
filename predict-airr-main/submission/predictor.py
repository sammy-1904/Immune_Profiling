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
        # --- your code starts here ---
        # Example: Store hyperparameters, the actual model, identified important sequences, etc.

        # NOTE: we encourage you to use self.n_jobs and self.device if appropriate in
        # your implementation instead of hardcoding these values because your code may later be run in an
        # environment with different hardware resources.

        self.model = None
        self.important_sequences_ = None
        # --- your code ends here ---

    def fit(self, train_dir_path: str):
        """
        Trains the model on the provided training data.

        Args:
            train_dir_path (str): Path to the directory with training TSV files.

        Returns:
            self: The fitted predictor instance.
        """

        # --- your code starts here ---
        # Load the data, prepare suited representations as needed, train your model,
        # and find the top k important sequences that best explain the labels.
        # Example: Load the data. One possibility could be to use the provided utility function as shown below.

        # full_train_dataset_df = load_full_dataset(train_dir_path)

        #   Model Training
        #    Example: self.model = SomeClassifier().fit(X_train, y_train)
        self.model = "some trained model"  # Replace with your actual learnt model

        #   Identify important sequences (can be done here or in the dedicated method)
        #    Example:
        self.important_sequences_ = self.identify_associated_sequences(top_k=50000, dataset_name=os.path.basename(train_dir_path))

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
        if self.model is None:
            raise RuntimeError("The model has not been fitted yet. Please call `fit` first.")

        # --- your code starts here ---

        # Example: Load the data. One possibility could be to use the provided utility function as shown below.

        # full_test_dataset_df = load_full_dataset(test_dir_path)
        repertoire_ids = get_repertoire_ids(test_dir_path)  # Replace with actual repertoire IDs from the test data

        # Prediction
        #    Example:
        # draw random probabilities for demonstration purposes

        probabilities = np.random.rand(len(repertoire_ids)) # Replace with true predicted probabilities from your model

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