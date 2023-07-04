import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=80, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1, fbank_size=80, batch_first=True):
        super().__init__()
        self.conv1 = nn.Conv2d(1, d_model, 3, stride=2)
        self.ReLU = nn.ReLU()
        self.conv2 = nn.Conv2d(d_model, d_model, 3, stride=2)
        self.embedding = nn.Linear(4608, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.d_model = d_model


    def forward(self, src, src_key_padding_mask=None):
        src_key_padding_mask = src_key_padding_mask.unsqueeze(dim=1)
        src = src.unsqueeze(dim=1)
        src = self.conv1(src)
        src = self.ReLU(src)
        src = self.conv2(src)
        src = self.ReLU(src)
        src = self.conv2(src)
        src = self.ReLU(src)
        b, c, t, w = src.size()
        src = src.transpose(1, 2).contiguous().view(b, t, c * w)
        src = self.embedding(src)
        src = self.pos_encoder(src)
        src_key_padding_mask = src_key_padding_mask[:, :, :-2:2][:, :, :-2:2][:, :, :-2:2]
        src_key_padding_mask = src_key_padding_mask.squeeze(1)


        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        return output,src_key_padding_mask

class TransformerDecoder(nn.Module):
    def __init__(self, d_model=80, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1, vocab_size=10,batch_first=True):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.embedding = nn.Embedding(vocab_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.d_model = d_model
        self.fc = nn.Linear(d_model, vocab_size)


    def forward(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None,tgt_mask=None):
        tgt = self.embedding(tgt)
        tgt = self.pos_encoder(tgt)
        output = self.transformer_decoder(tgt, memory, tgt_key_padding_mask=tgt_key_padding_mask,
                                           memory_key_padding_mask=memory_key_padding_mask, tgt_mask=tgt_mask)
        output = self.fc(output)
        return output

class TransformerASR(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=8, num_decoder_layers=4, dim_feedforward=2048,
                 dropout=0.1, vocab_size=4338,fbank_size=80,pad_value=-1,batch_first=True):
        super().__init__()
        self.encoder = TransformerEncoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout,fbank_size,batch_first=True)
        self.decoder = TransformerDecoder(d_model, nhead, num_decoder_layers, dim_feedforward, dropout, vocab_size,batch_first=True)
        self.ctc_linear_out = nn.Linear(d_model,vocab_size)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)


 
    def forward(self, src, tgt, device, tgt_key_padding_mask=None,  src_key_padding_mask=None):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(-1)).to(device)
        tgt_mask.bool()

        memory, src_key_padding_mask = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        ctc_out = self.ctc_linear_out(memory)
        
        return output, ctc_out, src_key_padding_mask
