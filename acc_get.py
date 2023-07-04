import torch
from torchaudio.models.decoder import CTCDecoder
from torchaudio.models.decoder import ctc_decoder
from torch.nn import functional as F

def get_acc(pred, labels, src_mask_len,tgt_mask_len, voc_dict, device):

    with torch.no_grad():
        pred = pred.to(device)
        labels = labels.to(device)
        src_mask_len = src_mask_len.to(device)
        pred = torch.nn.functional.softmax(pred, dim=-1)
        decoder = ctc_decoder(lexicon=None,
                               tokens=list(voc_dict.keys()),
                               lm=None,
                               nbest=3,
                               beam_size=10,
                               sil_token="-")
        results = decoder(pred, src_mask_len)
        acc_sum = 0
        for batch in range(len(labels)):


            now_hy = results[batch][1].tokens
            now_hy = now_hy[1:-1]


            pad_len = tgt_mask_len[batch] - len(now_hy)
            now_hy = F.pad(now_hy, (0, pad_len), "constant", voc_dict["<pad>"])

            for j in range(tgt_mask_len[batch]):

                if now_hy[j] == labels[batch][j]:
                    acc_sum += 1
        count_sum = 0
        for i in src_mask_len:
            count_sum += i

    return acc_sum / count_sum









