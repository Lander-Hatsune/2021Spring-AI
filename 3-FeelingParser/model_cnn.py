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

class cnn(nn.Module):

    def __init__(self):
        super(cnn, self).__init__()
        self.embed = nn.Embedding(len(ALL_WORDS_DIC), EMBED_LEN,
                                  max_norm=EMBED_MAX_NORM)
        self.conv3 = nn.Conv2d(1, 100, (3, EMBED_LEN))
        self.conv4 = nn.Conv2d(1, 100, (4, EMBED_LEN))
        self.conv5 = nn.Conv2d(1, 100, (5, EMBED_LEN))

        # maxpool3, 4, 5

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(300, 7)
        self.softmax = nn.Softmax(1)

    def forward(self, x):

        x = self.embed(x).unsqueeze(1)
        c3 = F.relu(self.conv3(x)).max(2)[0].view(-1, 100)
        c4 = F.relu(self.conv4(x)).max(2)[0].view(-1, 100)
        c5 = F.relu(self.conv5(x)).max(2)[0].view(-1, 100)

        flat = torch.cat((c3, c4, c5), 1)
        
        out = F.relu(self.fc(self.dropout(flat)))
        return self.softmax(out)


