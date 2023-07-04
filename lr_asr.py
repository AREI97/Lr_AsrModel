import torch
from conformer_asr import ConformerAsr

from torch import nn

class LrAsr(nn.Module):
    def __init__(self,d_model,num_head,num_hide,num_encoder,vocab_size,model_id,device,droup):
        super(LrAsr,self).__init__()
        self.device = device
        self.model_id = model_id
        self.d_model = d_model
        self.num_head = num_head
        self.num_hide = num_hide
        self.num_encoder = num_encoder
        self.vocab_size = vocab_size
        self.droup =droup
        self.baseModelList = nn.ModuleList([])
        self.lastLiner = nn.ModuleList()
        for i in range(len(self.model_id)):
            self.baseModelList.append(ConformerAsr(self.d_model,self.num_head,
                                                   self.num_hide,self.num_encoder,self.vocab_size[i],self.droup))
        self.model_nums = len(self.baseModelList)
        self.lastLiner.append(nn.Linear(in_features=2457600,out_features=self.model_nums))
        self.vote = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Linear(in_features=10, out_features=256),
            nn.Dropout(p=0.1),
            nn.Flatten(),
        )

    def now_baseModelinite(self,train_id,train_vocab_size):
        if train_id not in self.model_id:
            self.model_id.append(train_id)
            self.vocab_size.append(train_vocab_size)
            self.baseModelList.append(ConformerAsr(self.d_model,self.num_head,self.num_hide,self.num_encoder,self.vocab_size[train_id],self.droup))
            del self.lastLiner[0]
            self.lastLiner.append(nn.Linear(in_features=2457600,out_features=len(self.model_id)))

    def base_train(self,x,x_mask,train_id,device):
        pre,pre_pad_len = self.baseModelList[train_id](x,x_mask,device)
        return pre,pre_pad_len

    def vote_train(self,x):
        pre = self.vote(x)
        pre = self.lastLiner[0](pre)
        return pre

    def inference(self,x,x_mask,device):
        get_id = self.vote(x.unsqueeze(dim=1))
        get_id = self.lastLiner[0](get_id)
        get_id = nn.functional.softmax(get_id,dim=-1)
        get_id = torch.argmax(get_id,dim=-1)
        sum = []
        sum_pad_len = []
        count = 0
        for index in get_id:
            pre,pre_pad_len = self.baseModelList[index](x[count].unsqueeze(0),x_mask[count].unsqueeze(0),device)
            count += 1
            sum.append(pre)
            sum_pad_len.append(pre_pad_len)

        return sum, sum_pad_len,get_id

