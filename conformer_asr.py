import torch

import torchaudio

from torch import nn
from torch.nn.utils import rnn

from train_loop import get_ctc_pad
class ConformerAsr(nn.Module):
    def __init__(self,d_model,num_head,num_hide,num_encoder,vocab_size,drop=0.1):
        # 构造参数：
        # d_model 输入到conformer中的维度
        # num_head conformer的多头注意力数量
        # num_hide conformer的全连接层维度
        # num_encoder 编码器层数
        # vocab_size 词典大小
        # 输入值:
        # x 大小为(batch_size,t,w)的输入特征
        # x_mask x的填充矩阵 形状为(batch_size,max_len) dtype为bool
        # 返回值:
        # return 形状为(batch_size,vocb_size)
        # return_pad_len,形状为(batch_size,)的矩阵,值为每个结果的有效长度
        super(ConformerAsr,self).__init__()
        self.sub = nn.Sequential(
            nn.Conv2d(1, d_model, 3, 2),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, 3, 2),
            nn.ReLU(),

        )
        self.embed = nn.Sequential(
            nn.Linear(4864, d_model),
            nn.Dropout(p=drop),
        )

        self.Conformer = torchaudio.models.Conformer(d_model,num_head,num_hide,num_encoder,depthwise_conv_kernel_size=3,dropout=drop)

        self.Decoder = nn.LSTM(d_model, 1024, 2)
        self.outlayer = nn.Linear(1024,vocab_size)

    def change_outlayer(self,vocabsize):
        self.outlayer = nn.Linear(1024,vocabsize)

    def forward(self,x,x_mask,device):
        x = torch.unsqueeze(x, dim=1)
        x_mask = torch.unsqueeze(x_mask, dim=1)
        x = self.sub(x)
        b, c, t, w = x.size()
        x = x.transpose(1, 2).contiguous().view(b, t, c * w)
        x = self.embed(x)

        x_mask = x_mask[:, :, :-2:2][:, :, :-2:2]
        x_mask = x_mask.squeeze(dim=1)
        x_pad_len = get_ctc_pad(x_mask)

        x_pad_len = x_pad_len.to(device)

        x, x_pad_len = self.Conformer(x, x_pad_len)

        x = rnn.pack_padded_sequence(x, x_pad_len.to("cpu"), batch_first=True, enforce_sorted=False)
        x,_ = self.Decoder(x)
        x, x_pad_len = rnn.pad_packed_sequence(x, padding_value=0,
                                               total_length=x_mask.size(-1),
                                               batch_first=True)
        x = self.outlayer(x)

        return x, x_pad_len.to(device)


