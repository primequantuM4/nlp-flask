import pickle
import torch
import numpy as np
from hatespeech_lstm import HateSpeechLSTM


class PredictionService:
    def __init__(self):
        self.embedding_model_path = "embedding_matrix.pkl"
        self.best_model_path = "best_model.pth"
        self.load_models()

    def load_models(self):
        # Load embedding matrix
        with open("embedding_matrix.pkl", "rb") as f:
            self.embedding_matrix = pickle.load(f)


        # Initialize model and load weights
        self.embedding_dim = 100
        hidden_dim = 64
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.best_model = HateSpeechLSTM(embedding_dim=self.embedding_dim, hidden_dim=hidden_dim).to(device)
        self.best_model.load_state_dict(torch.load("best_model.pth"))



    # Function to vectorize posts
    def vectorize_post(self, tokens, embedding_matrix, max_len=10):
        vectors = [embedding_matrix[token] if token in embedding_matrix else np.zeros(self.embedding_dim) for token in tokens]
        if len(vectors) > max_len:
            vectors = vectors[:max_len]
        else:
            vectors.extend([np.zeros(self.embedding_dim)] * (max_len - len(vectors)))
        return np.array(vectors)

