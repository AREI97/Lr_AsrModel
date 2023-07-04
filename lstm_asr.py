import torch
from torch import nn
from  train_loop import  get_ctc_pad
from torch.nn import functional as F

from torch.nn.utils import  rnn

class Lstm_Asr(nn.Module):
    def __init__(self,
                 hide_dim,
                 d_model,
                 droup,
                 encoder_num_layers,
                 linear_dim,
                 vocab_size):
        super(Lstm_Asr,self).__init__()
        self.sub = nn.Sequential(
            nn.Conv2d(1,d_model,3,2),
            nn.ReLU(),
            nn.Conv2d(d_model,d_model,3,2),
            nn.ReLU(),
        )
        self.bilstm = nn.LSTM(9728,
                              hide_dim,
                              encoder_num_layers,
                              bidirectional=False,
                              batch_first=True,
                              dropout=droup
                              )
        self.ffd = nn.Sequential(
            nn.Linear(hide_dim,
                      vocab_size),

        )

    def forward(self, x, x_mask, pad_value, device):
        x = torch.unsqueeze(x, dim=1)
        x_mask = torch.unsqueeze(x_mask, dim=1)
        x = self.sub(x)
        b, c, t, w = x.size()
        x = x.transpose(1, 2).contiguous().view(b, t, c * w)
        #x = self.embed(x)
        x_mask = x_mask[:, :, :-2:2][:, :, :-2:2]
        x_mask = x_mask.squeeze(dim=1)
        x_pad_len = get_ctc_pad(x_mask)
        x = rnn.pack_padded_sequence(x, x_pad_len, batch_first=True, enforce_sorted=False)
        x, _ = self.bilstm(x)
        x, x_pad_len = rnn.pad_packed_sequence(x, padding_value=pad_value,
                                    total_length=x_mask.size(-1),
                                    batch_first=True)
        x = self.ffd(x)
        return x, x_pad_len


