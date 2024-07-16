import torch
import torch.nn as nn
import torch.nn.functional as F
#
from torchcrf import CRF


from configBAC import VOCAB_SIZE, WORD_PAD_ID, HIDDEN_SIZE, num_layers, num_heads, TARGET_SIZE, \
    batch_size, base_len,EMBEDDING_DIM


class ModelBAC(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM, padding_idx=WORD_PAD_ID)

        self.lstm1 = nn.LSTM(
            EMBEDDING_DIM,
            HIDDEN_SIZE,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout1 = nn.Dropout(0.6)

        self.attention = nn.MultiheadAttention(embed_dim=HIDDEN_SIZE * 2, num_heads=num_heads)
        self.dropout2 = nn.Dropout(0.6)
        self.linear1 = nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE * 2)
        self.dropout3 = nn.Dropout(0.7)
        self.linear2 = nn.Linear(HIDDEN_SIZE * 2, TARGET_SIZE)
        self.dropout4 = nn.Dropout(0.7)
        self.crf = CRF(TARGET_SIZE, batch_first=True)

    def _get_lstm_feature(self, input):
        out = self.embed(input)
        out, _ = self.lstm1(out)
        out = self.dropout1(out)

        # Transpose to (seq_length, batch_size, hidden_size*2) as required by nn.MultiheadAttention
        out = out.transpose(0, 1)  # (batch_size, seq_length, hidden_size*2) -> (seq_length, batch_size, hidden_size*2)
        out, _ = self.attention(out, out, out)
        out = out.transpose(0, 1)  # (seq_length, batch_size, hidden_size*2) -> (batch_size, seq_length, hidden_size*2)

        out = self.dropout2(out)
        out = F.relu(self.linear1(out))  # Optional non-linear transformation layer
        out = self.dropout3(out)
        out = self.linear2(out)
        out = self.dropout4(out)
        return out

    def forward(self, input, mask):
        out = self._get_lstm_feature(input)
        return self.crf.decode(out, mask)

    def loss_fn(self, input, target, mask):
        y_pred = self._get_lstm_feature(input)
        return -self.crf(y_pred, target, mask, reduction='mean')

if __name__ == '__main__':
    model = ModelBAC()
    input = torch.randint(0, VOCAB_SIZE, (batch_size, base_len))
