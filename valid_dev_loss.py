import torch
from torch import nn
from  torch.nn import  functional as F
from  train_loop import  get_ctc_pad
from acc_get import  get_acc

def dev_loss(dev_loader,model,ctc_loss,device):

    with torch.no_grad():
        model.eval()
        itr = iter(dev_loader)
        (src, tgt, src_key_padding_mask, tgt_key_padding_mask) = next(itr)

        pred, pred_pad_len = model(src.to(device),
                                   src_key_padding_mask.to(device),
                                   device,
                                    )

        ctc_pred = F.log_softmax(pred,dim=-1)

        tgt_pad_len = get_ctc_pad(tgt_key_padding_mask)


        ctc_loss = ctc_loss(ctc_pred.transpose(0, 1), tgt, pred_pad_len, tgt_pad_len)




        #acc = get_acc(pred,tgt,pred_pad_len,tgt_pad_len,voc_dict,device)
        print(f"dev_loss : {ctc_loss.item():>7f}")
