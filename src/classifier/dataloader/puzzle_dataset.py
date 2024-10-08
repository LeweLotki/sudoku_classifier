import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import Dataset

class PuzzleDataset(Dataset):
    def __init__(self, csv_path, max_features=10000, vectorization='tf-idf', max_length=100):
        df = pd.read_csv(csv_path)

        self.vectorization = vectorization
        self.max_length = max_length

        if self.vectorization == 'tf-idf':
            self.vectorizer = TfidfVectorizer(max_features=max_features)
            self.features = self.vectorizer.fit_transform(df['features']).toarray()
            self.features = torch.tensor(self.features, dtype=torch.float32)

        elif self.vectorization == 'count':
            self.vectorizer = CountVectorizer(max_features=max_features, tokenizer=lambda x: x.split(), lowercase=False)
            self.features = self.vectorizer.fit_transform(df['features']).toarray()
            
            if self.features.shape[1] < max_length:
                padding = np.zeros((self.features.shape[0], max_length - self.features.shape[1]))
                self.features = np.concatenate((self.features, padding), axis=1)
            else:
                self.features = self.features[:, :max_length]

            self.features = torch.tensor(self.features, dtype=torch.long)

        self.encoder = OneHotEncoder(sparse_output=False)
        target = df['target'].values.reshape(-1, 1)
        self.target = self.encoder.fit_transform(target)
        self.target = torch.tensor(self.target, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]

    def get_vectorizer(self):
        return self.vectorizer

