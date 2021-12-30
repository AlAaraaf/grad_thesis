import torch
from torch import nn

class LSTM_MODEL(nn.Module):
    def __init__(self,vocab_size, embedding_dim, hidden_size, num_classes, padd_idx, device):
        super(LSTM_MODEL, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.device = device
        
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=padd_idx)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size,batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        h0 = torch.randn(2, batch_size, self.hidden_size).to(self.device)
        c0 = torch.randn(2, batch_size, self.hidden_size).to(self.device)
        x = self.embedding(x)
        out,(_,_)= self.lstm(x, (h0,c0))
        output = self.fc(out[:,-1,:]).squeeze(0) #因为有max_seq_len个时态，所以取最后一个时态即-1层
        return output
