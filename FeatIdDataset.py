from torch.utils.data import Dataset
import os
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import torch.nn.functional as F
import soundfile as sf
import torch
import sentencepiece as spm


class FeatIdDataset(Dataset):
    def __init__(self, root, data_type, max_seq_len, tag_id, bpemodelname=None, main_path=None, label_func=None,
                 voc_dict=None):
        # 该数据集用于训练分类器得到fbank特征对应语言种类的概率
        # librispeech类型的数据需要两个路径,root:存放数据集的路径，在路径集下有需要的txt文件,main_path存放flac格式文件的主目录
        # aishell数据集需要label_func和voc_dict，此外需要指定每个语言的bpe模型名字=bpemodelname
        # data_type 指明类librispeech数据库txt文件名称
        self.max_seq_len = max_seq_len
        self.root = root
        self.data_type = data_type
        self.main_path = main_path
        self.tag_id = tag_id
        if main_path is None:
            self.voc_dict = voc_dict
            self.transcript_path = os.path.join(self.root, r"transcript\aishell_transcript_v0.8.txt")
            self.wav_path = os.path.join(self.root, "wav")
            self.labels_dict = label_func(self.transcript_path, voc_dict, self.wav_path, self.data_type)
        else:
            self.seq_path = self.root + "/" + data_type + "_seq.txt"
            self.txt_path = self.root + "/" + data_type + ".txt"
            with open(self.txt_path, "r",encoding="utf-8") as f:
                self.indexs = f.readlines()
            with open(self.seq_path, "r",encoding="utf-8") as f:
                self.seqs = f.readlines()
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(bpemodelname)

    def __len__(self):
        if self.main_path is None:
            return len(self.labels_dict)
        else:
            len(self.indexs) - 1

    def __getitem__(self, idx):
        if self.main_path is None:
            if self.data_type == "train":
                path = os.path.join(self.root, r"wav/train", list(self.labels_dict.keys())[idx][6:11],
                                    list(self.labels_dict.keys())[idx] + '.wav')
            elif self.data_type == "dev":
                path = os.path.join(self.root, r"wav/dev", list(self.labels_dict.keys())[idx][6:11],
                                    list(self.labels_dict.keys())[idx] + '.wav')
            elif self.data_type == "test":
                path = os.path.join(self.root, r"wav/test", list(self.labels_dict.keys())[idx][6:11],
                                    list(self.labels_dict.keys())[idx] + '.wav')
            (rate, sig) = wav.read(path)
        elif self.tag_id == 1:
            now_path = self.indexs[idx].split("-")
            now_path = os.path.join(self.main_path,
                                    now_path[0],
                                    now_path[1])
            now_wav = os.path.join(now_path, self.indexs[idx][:-1]) + ".flac"
            sig, rate = sf.read(now_wav)
        elif self.tag_id == 2:
            now_path = self.indexs[idx].split("_")
            now_path = os.path.join(self.main_path,
                                    now_path[1],
                                    now_path[0])
            now_wav = os.path.join(now_path, self.indexs[idx][:-1]) + ".flac"
            sig, rate = sf.read(now_wav)

        fbank_feat = logfbank(sig, rate, nfilt=80)
        fbank_feat = torch.tensor(fbank_feat, dtype=torch.float32)
        add_len = self.max_seq_len - len(fbank_feat)
        fbank_feat = F.pad(fbank_feat, (0, 0, 0, add_len), "constant", 0)
        fbank_feat = torch.unsqueeze(fbank_feat, dim=0)

        return fbank_feat, self.tag_id
