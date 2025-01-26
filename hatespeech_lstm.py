import torch.nn as nn


# Define the LSTM model class
class HateSpeechLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0.5):
        super(HateSpeechLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True
        )
        self.batch_norm_lstm = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.batch_norm_fc = nn.BatchNorm1d(1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        hidden = hidden[-1]
        hidden = self.batch_norm_lstm(hidden)
        hidden = self.dropout(hidden)
        out = self.fc(hidden)
        out = self.batch_norm_fc(out)
        return self.sigmoid(out)
