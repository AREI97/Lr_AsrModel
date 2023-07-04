import torch
from torch.utils.data import Dataset
import os
import sentencepiece as spm
from python_speech_features import logfbank
import soundfile as sf
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import TimeMasking
from torchaudio.transforms import TimeStretch
from torchaudio.transforms import Spectrogram, GriffinLim

class LibrispeechAsr(Dataset):
    def __init__(self,txt_main_path,main_path,data_type,max_seq_len,max_tag_len):
        #txt_main_path txt文件的目录路径
        #main_path  包含各个编号文件的目录路径
        self.txt_main_path = txt_main_path
        self.txt_path = self.txt_main_path + data_type +".txt"
        with open(self.txt_path,"r") as f:
            self.indexs = f.readlines()
        self.main_path = main_path
        self.seq_path = self.txt_main_path + data_type + "_seq.txt"
        with open(self.seq_path,"r") as f:
            self.seqs = f.readlines()
        self.sp = spm.SentencePieceProcessor()
        self.sp.load("LibrispeechBepModel.model")
        self.max_seq_len = max_seq_len
        self.max_tag_len = max_tag_len


    def __len__(self):
        return len(self.indexs)-1

    def __getitem__(self,idx):
        now_path = self.indexs[idx].split("-")
        now_path = os.path.join(self.main_path,
                               now_path[0],
                               now_path[1])
        now_wav = os.path.join(now_path,self.indexs[idx][:-1]) + ".flac"
        now_label = self.sp.encode(self.seqs[idx])
        now_label = torch.tensor(now_label)
        sig,rate = sf.read(now_wav)
        fbank_feat = logfbank(sig,rate,nfilt=80)
        add_len = self.max_seq_len - len(fbank_feat)
        src_padding_mask = torch.zeros(self.max_seq_len)
        src_padding_mask[len(fbank_feat):] = 1
        fbank_feat = torch.tensor(fbank_feat,dtype=torch.float)
        fbank_feat = F.pad(fbank_feat,(0,0,0,add_len),"constant",0)
        label_padding_mask = torch.zeros(self.max_tag_len)
        label_padding_mask[len(now_label):] = 1
        add_len = self.max_tag_len - len(now_label)
        now_label = F.pad(now_label,(0,add_len),"constant",self.sp.piece_to_id("<pad>"))
        return fbank_feat,now_label,src_padding_mask.bool(),label_padding_mask.bool()




