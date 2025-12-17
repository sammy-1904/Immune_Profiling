import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob
import sys
from collections import defaultdict, Counter
from typing import Iterator, Tuple, Union, List

# ============================================================================
# AMINO ACID ENCODING - Atchley Factors
# Reference: Atchley et al. (2005) PNAS - "Solving the protein sequence metric problem"
# Each amino acid is represented by 5 physicochemical factors:
# Factor 1: Polarity, Factor 2: Secondary structure, Factor 3: Molecular size,
# Factor 4: Codon diversity, Factor 5: Electrostatic charge
# ============================================================================

ATCHLEY_FACTORS = {
    'A': [-0.591, -1.302, -0.733,  1.570, -0.146],
    'C': [-1.343,  0.465, -0.862, -1.020, -0.255],
    'D': [ 1.050,  0.302, -3.656, -0.259, -3.242],
    'E': [ 1.357, -1.453,  1.477,  0.113, -0.837],
    'F': [-1.006, -0.590,  1.891, -0.397,  0.412],
    'G': [-0.384,  1.652,  1.330,  1.045,  2.064],
    'H': [ 0.336, -0.417, -1.673, -1.474, -0.078],
    'I': [-1.239, -0.547,  2.131,  0.393,  0.816],
    'K': [ 1.831, -0.561,  0.533, -0.277,  1.648],
    'L': [-1.019, -0.987, -1.505,  1.266, -0.912],
    'M': [-0.663, -1.524,  2.219, -1.005,  1.212],
    'N': [ 0.945,  0.828,  1.299, -0.169,  0.933],
    'P': [ 0.189,  2.081, -1.628,  0.421, -1.392],
    'Q': [ 0.931, -0.179, -3.005, -0.503, -1.853],
    'R': [ 1.538, -0.055,  1.502,  0.440,  2.897],
    'S': [-0.228,  1.399, -4.760,  0.670, -2.647],
    'T': [-0.032,  0.326,  2.213,  0.908,  1.313],
    'V': [-1.337, -0.279, -0.544,  1.242, -1.262],
    'W': [-0.595,  0.009,  0.672, -2.128, -0.184],
    'Y': [ 0.260,  0.830,  3.097, -0.838,  1.512],
    'X': [ 0.000,  0.000,  0.000,  0.000,  0.000],  # Unknown amino acid
}

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWYX'
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}


def encode_sequence_atchley(seq: str, max_len: int = 25) -> np.ndarray:
    """
    Encode amino acid sequence using Atchley factors with padding/truncation.
    
    Args:
        seq: Amino acid sequence string
        max_len: Maximum sequence length (will pad or truncate)
    
    Returns:
        np.ndarray: Shape (max_len, 5) - Atchley factor encoding
    """
    encoding = []
    seq = seq[:max_len]  # Truncate if too long
    
    for aa in seq:
        if aa in ATCHLEY_FACTORS:
            encoding.append(ATCHLEY_FACTORS[aa])
        else:
            encoding.append(ATCHLEY_FACTORS['X'])  # Unknown amino acid
    
    # Pad with zeros if too short
    while len(encoding) < max_len:
        encoding.append([0.0] * 5)
    
    return np.array(encoding, dtype=np.float32)


class GeneEncoder:
    """Encoder for V/J gene calls to numerical indices."""
    
    def __init__(self):
        self.v_encoder = None
        self.j_encoder = None
        self.v_classes = []
        self.j_classes = []
        self.n_v_genes = 0
        self.n_j_genes = 0
        
    def fit(self, v_genes: List[str], j_genes: List[str]):
        """Fit encoder on V and J gene lists."""
        # Clean and get unique genes
        v_genes_clean = list(set([str(g) if pd.notna(g) else 'UNK' for g in v_genes]))
        j_genes_clean = list(set([str(g) if pd.notna(g) else 'UNK' for g in j_genes]))
        
        # Add UNK if not present
        if 'UNK' not in v_genes_clean:
            v_genes_clean.append('UNK')
        if 'UNK' not in j_genes_clean:
            j_genes_clean.append('UNK')
        
        self.v_classes = sorted(v_genes_clean)
        self.j_classes = sorted(j_genes_clean)
        
        self.v_encoder = {g: i for i, g in enumerate(self.v_classes)}
        self.j_encoder = {g: i for i, g in enumerate(self.j_classes)}
        
        self.n_v_genes = len(self.v_classes)
        self.n_j_genes = len(self.j_classes)
        
    def transform_v(self, v_gene: str) -> int:
        """Transform V gene to index."""
        v_gene = str(v_gene) if pd.notna(v_gene) else 'UNK'
        return self.v_encoder.get(v_gene, self.v_encoder['UNK'])
            
    def transform_j(self, j_gene: str) -> int:
        """Transform J gene to index."""
        j_gene = str(j_gene) if pd.notna(j_gene) else 'UNK'
        return self.j_encoder.get(j_gene, self.j_encoder['UNK'])


def load_data_generator(data_dir: str, metadata_filename='metadata.csv') -> Iterator[
    Union[Tuple[str, pd.DataFrame, bool], Tuple[str, pd.DataFrame]]]:
    """
    A generator to load immune repertoire data.

    This function operates in two modes:
    1.  If metadata is found, it yields data based on the metadata file.
    2.  If metadata is NOT found, it uses glob to find and yield all '.tsv'
        files in the directory.

    Args:
        data_dir (str): The path to the directory containing the data.

    Yields:
        An iterator of tuples. The format depends on the mode:
        - With metadata: (repertoire_id, pd.DataFrame, label_positive)
        - Without metadata: (filename, pd.DataFrame)
    """
    metadata_path = os.path.join(data_dir, metadata_filename)

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        for row in metadata_df.itertuples(index=False):
            file_path = os.path.join(data_dir, row.filename)
            try:
                repertoire_df = pd.read_csv(file_path, sep='\t')
                yield row.repertoire_id, repertoire_df, row.label_positive
            except FileNotFoundError:
                print(f"Warning: File '{row.filename}' listed in metadata not found. Skipping.")
                continue
    else:
        search_pattern = os.path.join(data_dir, '*.tsv')
        tsv_files = glob.glob(search_pattern)
        for file_path in sorted(tsv_files):
            try:
                filename = os.path.basename(file_path)
                repertoire_df = pd.read_csv(file_path, sep='\t')
                yield filename, repertoire_df
            except Exception as e:
                print(f"Warning: Could not read file '{file_path}'. Error: {e}. Skipping.")
                continue


def load_full_dataset(data_dir: str) -> pd.DataFrame:
    """
    Loads all TSV files from a directory and concatenates them into a single DataFrame.

    This function handles two scenarios:
    1. If metadata.csv exists, it loads data based on the metadata and adds
       'repertoire_id' and 'label_positive' columns.
    2. If metadata.csv does not exist, it loads all .tsv files and adds
       a 'filename' column as an identifier.

    Args:
        data_dir (str): The path to the data directory.

    Returns:
        pd.DataFrame: A single, concatenated DataFrame containing all the data.
    """
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    df_list = []
    data_loader = load_data_generator(data_dir=data_dir)

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        total_files = len(metadata_df)
        for rep_id, data_df, label in tqdm(data_loader, total=total_files, desc="Loading files"):
            data_df['ID'] = rep_id
            data_df['label_positive'] = label
            df_list.append(data_df)
    else:
        search_pattern = os.path.join(data_dir, '*.tsv')
        total_files = len(glob.glob(search_pattern))
        for filename, data_df in tqdm(data_loader, total=total_files, desc="Loading files"):
            data_df['ID'] = os.path.basename(filename).replace(".tsv", "")
            df_list.append(data_df)

    if not df_list:
        print("Warning: No data files were loaded.")
        return pd.DataFrame()

    full_dataset_df = pd.concat(df_list, ignore_index=True)
    return full_dataset_df


def load_and_encode_kmers(data_dir: str, k: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loading and k-mer encoding of repertoire data.

    Args:
        data_dir: Path to data directory
        k: K-mer length

    Returns:
        Tuple of (encoded_features_df, metadata_df)
        metadata_df always contains 'ID', and 'label_positive' if available
    """
    from collections import Counter

    metadata_path = os.path.join(data_dir, 'metadata.csv')
    data_loader = load_data_generator(data_dir=data_dir)

    repertoire_features = []
    metadata_records = []

    search_pattern = os.path.join(data_dir, '*.tsv')
    total_files = len(glob.glob(search_pattern))

    for item in tqdm(data_loader, total=total_files, desc=f"Encoding {k}-mers"):
        if os.path.exists(metadata_path):
            rep_id, data_df, label = item
        else:
            filename, data_df = item
            rep_id = os.path.basename(filename).replace(".tsv", "")
            label = None

        kmer_counts = Counter()
        for seq in data_df['junction_aa'].dropna():
            for i in range(len(seq) - k + 1):
                kmer_counts[seq[i:i + k]] += 1

        repertoire_features.append({
            'ID': rep_id,
            **kmer_counts
        })

        metadata_record = {'ID': rep_id}
        if label is not None:
            metadata_record['label_positive'] = label
        metadata_records.append(metadata_record)

        del data_df, kmer_counts

    features_df = pd.DataFrame(repertoire_features).fillna(0).set_index('ID')
    features_df.fillna(0)
    metadata_df = pd.DataFrame(metadata_records)

    return features_df, metadata_df


def save_tsv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep='\t', index=False)


def get_repertoire_ids(data_dir: str) -> list:
    """
    Retrieves repertoire IDs from the metadata file or filenames in the directory.

    Args:
        data_dir (str): The path to the data directory.

    Returns:
        list: A list of repertoire IDs.
    """
    metadata_path = os.path.join(data_dir, 'metadata.csv')

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        repertoire_ids = metadata_df['repertoire_id'].tolist()
    else:
        search_pattern = os.path.join(data_dir, '*.tsv')
        tsv_files = glob.glob(search_pattern)
        repertoire_ids = [os.path.basename(f).replace('.tsv', '') for f in sorted(tsv_files)]

    return repertoire_ids


def generate_random_top_sequences_df(n_seq: int = 50000) -> pd.DataFrame:
    """
    Generates a random DataFrame simulating top important sequences.

    Args:
        n_seq (int): Number of sequences to generate.

    Returns:
        pd.DataFrame: A DataFrame with columns 'ID', 'dataset', 'junction_aa', 'v_call', 'j_call'.
    """
    seqs = set()
    while len(seqs) < n_seq:
        seq = ''.join(np.random.choice(list('ACDEFGHIKLMNPQRSTVWY'), size=15))
        seqs.add(seq)
    data = {
        'junction_aa': list(seqs),
        'v_call': ['TRBV20-1'] * n_seq,
        'j_call': ['TRBJ2-7'] * n_seq,
        'importance_score': np.random.rand(n_seq)
    }
    return pd.DataFrame(data)


def validate_dirs_and_files(train_dir: str, test_dirs: List[str], out_dir: str) -> None:
    assert os.path.isdir(train_dir), f"Train directory `{train_dir}` does not exist."
    train_tsvs = glob.glob(os.path.join(train_dir, "*.tsv"))
    assert train_tsvs, f"No .tsv files found in train directory `{train_dir}`."
    metadata_path = os.path.join(train_dir, "metadata.csv")
    assert os.path.isfile(metadata_path), f"`metadata.csv` not found in train directory `{train_dir}`."

    for test_dir in test_dirs:
        assert os.path.isdir(test_dir), f"Test directory `{test_dir}` does not exist."
        test_tsvs = glob.glob(os.path.join(test_dir, "*.tsv"))
        assert test_tsvs, f"No .tsv files found in test directory `{test_dir}`."

    try:
        os.makedirs(out_dir, exist_ok=True)
        test_file = os.path.join(out_dir, "test_write_permission.tmp")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        print(f"Failed to create or write to output directory `{out_dir}`: {e}")
        sys.exit(1)


def concatenate_output_files(out_dir: str) -> None:
    """
    Concatenates all test predictions and important sequences TSV files from the output directory.

    This function finds all files matching the patterns:
    - *_test_predictions.tsv
    - *_important_sequences.tsv

    and concatenates them to match the expected output format of submissions.csv.

    Args:
        out_dir (str): Path to the output directory containing the TSV files.

    Returns:
        pd.DataFrame: Concatenated DataFrame with predictions followed by important sequences.
                     Columns: ['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']
    """
    predictions_pattern = os.path.join(out_dir, '*_test_predictions.tsv')
    sequences_pattern = os.path.join(out_dir, '*_important_sequences.tsv')

    predictions_files = sorted(glob.glob(predictions_pattern))
    sequences_files = sorted(glob.glob(sequences_pattern))

    df_list = []

    for pred_file in predictions_files:
        try:
            df = pd.read_csv(pred_file, sep='\t')
            df_list.append(df)
        except Exception as e:
            print(f"Warning: Could not read predictions file '{pred_file}'. Error: {e}. Skipping.")
            continue

    for seq_file in sequences_files:
        try:
            df = pd.read_csv(seq_file, sep='\t')
            df_list.append(df)
        except Exception as e:
            print(f"Warning: Could not read sequences file '{seq_file}'. Error: {e}. Skipping.")
            continue

    if not df_list:
        print("Warning: No output files were found to concatenate.")
        concatenated_df = pd.DataFrame(
            columns=['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call'])
    else:
        concatenated_df = pd.concat(df_list, ignore_index=True)
    submissions_file = os.path.join(out_dir, 'submissions.csv')
    concatenated_df.to_csv(submissions_file, index=False)
    print(f"Concatenated output written to `{submissions_file}`.")


def get_dataset_pairs(train_dir: str, test_dir: str) -> List[Tuple[str, List[str]]]:
    """Returns list of (train_path, [test_paths]) tuples for dataset pairs."""
    test_groups = defaultdict(list)
    for test_name in sorted(os.listdir(test_dir)):
        if test_name.startswith("test_dataset_"):
            base_id = test_name.replace("test_dataset_", "").split("_")[0]
            test_groups[base_id].append(os.path.join(test_dir, test_name))

    pairs = []
    for train_name in sorted(os.listdir(train_dir)):
        if train_name.startswith("train_dataset_"):
            train_id = train_name.replace("train_dataset_", "")
            train_path = os.path.join(train_dir, train_name)
            pairs.append((train_path, test_groups.get(train_id, [])))

    return pairs
