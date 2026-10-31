import os
import pandas as pd
import numpy as np

VOCAL_BIOMARKER_COLUMNS = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA',
    'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
]

def get_dataset_path():
    """
    Finds the parkinsons.csv file across various possible execution paths.
    """
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "parkinsons.csv"),
        os.path.join(os.getcwd(), "data", "parkinsons.csv"),
        os.path.join(os.getcwd(), "parkinsons.csv"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "parkinsons.csv"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "backend", "data", "parkinsons.csv")
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]

def load_dataset(dataset_path=None):
    """
    Loads and cleans the Parkinson's Disease Vocal Biomarker dataset.
    
    Returns:
        X (np.ndarray): 22 vocal biomarker features
        y (np.ndarray): Status label (0: Healthy, 1: Parkinson's)
        feature_names (list): List of feature names
    """
    if dataset_path is None:
        dataset_path = get_dataset_path()
        
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Parkinson's dataset not found at {dataset_path}")
        
    df = pd.read_csv(dataset_path, comment='#')
    
    # Drop name identifier if present
    if 'name' in df.columns:
        df = df.drop(columns=['name'])
        
    # Extract target
    if 'status' not in df.columns:
        raise ValueError("Dataset missing 'status' column")
        
    y = df['status'].values
    X_df = df.drop(columns=['status'])
    
    # Ensure correct columns
    feature_names = list(X_df.columns)
    X = X_df.values.astype(np.float32)
    
    return X, y, feature_names
