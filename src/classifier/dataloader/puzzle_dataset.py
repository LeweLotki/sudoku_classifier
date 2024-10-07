import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import Dataset

class PuzzleDataset(Dataset):
    def __init__(self, csv_path, max_features=10000):
        df = pd.read_csv(csv_path)
        
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.features = self.vectorizer.fit_transform(df['features']).toarray() 
        
        self.encoder = OneHotEncoder(sparse_output=False)
        target = df['target'].values.reshape(-1, 1)
        self.target = self.encoder.fit_transform(target)
        
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.target = torch.tensor(self.target, dtype=torch.float32)
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]
