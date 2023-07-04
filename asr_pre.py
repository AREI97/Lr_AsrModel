import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from  torch.utils.data import Dataset
from torch.utils.data import DataLoader
from python_speech_features import logfbank
import scipy.io.wavfile as wav


class AishellDataset(Dataset):
    def __init__(self,root,data_type,max_seq_len,max_tag_len,label_func,voc_dict):
        self.max_seq_len = max_seq_len
        self.max_tag_len = max_tag_len
        self.root = root
        self.voc_dict = voc_dict
        self.data_type = data_type
        self.transcript_path = os.path.join(self.root,r"transcript\aishell_transcript_v0.8.txt")
        self.wav_path = os.path.join(self.root,"wav")
        self.labels_dict = label_func(self.transcript_path,voc_dict,self.wav_path,self.data_type)
    def __len__(self):
        return len(self.labels_dict)
    
    def __getitem__(self,idx):
        if self.data_type == "train":
            path = os.path.join(self.root,r"wav/train",list(self.labels_dict.keys())[idx][6:11],list(self.labels_dict.keys())[idx]+'.wav')
        elif self.data_type == "dev":
            path = os.path.join(self.root,r"wav/dev",list(self.labels_dict.keys())[idx][6:11],list(self.labels_dict.keys())[idx]+'.wav')
        elif self.data_type == "test":
            path = os.path.join(self.root,r"wav/test",list(self.labels_dict.keys())[idx][6:11],list(self.labels_dict.keys())[idx]+'.wav')
        (rate,sig) = wav.read(path)
        fbank_feat = logfbank(sig,rate,nfilt=80)
        fbank_feat = torch.tensor(fbank_feat,dtype=torch.float32)
        #fbank_feat = F.normalize(fbank_feat,dim=1)
        add_seq = self.max_seq_len - len(fbank_feat)
        src_padding_mask = torch.zeros(self.max_seq_len)
        src_padding_mask[len(fbank_feat):] = 1
        fbank_feat = F.pad(fbank_feat,(0,0,0,add_seq),"constant",0)
        tgt = torch.tensor(list(self.labels_dict.values())[idx])

        tgt_padding_mask = torch.zeros(self.max_tag_len)
        tgt_padding_mask[len(tgt):] = 1
        add_tgt = self.max_tag_len - len(tgt)

        tgt = F.pad(tgt,(0,add_tgt),"constant",self.voc_dict["<pad>"])
        return fbank_feat, tgt, src_padding_mask.bool(), tgt_padding_mask.bool()
        

        


        

