import torch
from torch import nn
from torch.utils.data import DataLoader
from acc_get import  get_acc

def get_ctc_pad(key_mask):
    input_len=[]
    l = len(key_mask)
    i = 0
    count = 0
    seq_len = key_mask.size(-1)
    while i < l:
        k = 0
        count += 1
        j = 0
        while j < seq_len:
            
            if key_mask[i][j] != True and j+1 != seq_len:
                k += 1
            elif key_mask[i][j] == True:
                input_len.append(k)
                break
            elif j+1 == seq_len and key_mask[i][j] == False:
                k += 1
                input_len.append(k)
            j += 1
        i += 1
    
    
    if len(input_len) != l:
        print("break")
    return torch.tensor(input_len)

def train_loop(dataloader, model,ctc_loss_fn,optimizer,train_id,device):
    size = len(dataloader.dataset)
    model.train()
    model.to(device)
    Gradient_accrual = 4
    for batch,(src,tgt,src_key_padding_mask,tgt_key_padding_mask) in enumerate(dataloader):

        pred, pred_pad_len = model.base_train(src.to(device).requires_grad_(True),
                                   src_key_padding_mask.to(device),train_id,
                                   device,
                                   )
        
        ctc_pred = torch.nn.functional.log_softmax(pred,dim=-1)

        label_pad_len = get_ctc_pad(tgt_key_padding_mask)


        ctc_loss = ctc_loss_fn(ctc_pred.transpose(0,1), tgt, pred_pad_len, label_pad_len)

        ctc_loss = ctc_loss/Gradient_accrual
        


        ctc_loss.backward()
        
        #nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)

        if (batch+1) % Gradient_accrual == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        if (batch/Gradient_accrual) % 100 == 0:
            loss, current = ctc_loss.item(), batch * len(src)

            print(f"loss: {ctc_loss*Gradient_accrual:>7f}  [{current:>5d}/{size:>5d}]")
    #device = "cpu"
    #acc = get_acc(pred,tgt,pred_pad_len,label_pad_len,voc_dict,device)
    #device = "cuda"
    #print(f"train_acc: {acc:>7f}")