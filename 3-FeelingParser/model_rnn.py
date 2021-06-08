import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from dataset import get_all_words

ALL_WORDS_DIC = get_all_words()

EMBED_LEN = 256
EMBED_MAX_NORM = 1.0

class rnn(nn.Module):

    def __init__(self):
        super(rnn, self).__init__()
        self.embed = nn.Embedding(len(ALL_WORDS_DIC), EMBED_LEN,
                                  max_norm=EMBED_MAX_NORM)
        self.lstm = nn.LSTM(input_size=EMBED_LEN, hidden_size=256,
                            num_layers=1, batch_first=True)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, 7)
        self.softmax = nn.Softmax(1)

    def forward(self, sentences, lengths):

        embedded = self.embed(sentences)
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths,
                                                     batch_first=True,
                                                     enforce_sorted=False)
        lstm_out, _ = self.lstm(embedded)
        pad_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out,
                                                      batch_first=True)

        flat = torch.zeros((len(sentences), 256), requires_grad=True).to(sentences.device)
        for i, idx in enumerate(lengths):
            flat[i] = pad_out[i][idx - 1]

        out = F.relu(self.fc(self.dropout(flat)))
        return self.softmax(out)
        
        
        
        
