import numpy as np
import torch.optim
from python_speech_features import fbank
import scipy.io.wavfile as wav
from creat_voc import creat_voc
from get_labels import get_labels
from transformer_asr import *
from asr_pre import AishellDataset
import os
from torch.utils.data import DataLoader
from torch import nn
from valid_train_loop import train_loop
from valid_dev_loss import dev_loss
from lstm_asr import Lstm_Asr
from conformer_asr import ConformerAsr
from librispeech import *
import sentencepiece as spm
from lr_asr import LrAsr
from torch.utils.data import ConcatDataset
from Korean import KoreanAsr

if __name__ == '__main__':
    path = r"data_aishell\data_aishell"
    max_seq_len = 1200
    max_tag_len = 120
    batch_size = 16
    n_head = 4
    num_encoder = 8
    num_woker = 8
    d_model = 256
    device = "cuda"
    droup = 0.1
    linear_dim = 2048
    fbank_size = 80
    learning_rate = 5e-5
    epochs = 1000
    hide_dim = 512

    vocab_size = [4336, 2500, 2000]

    #aishell设置
    # aishell_path = os.path.join(path,r"transcript\aishell_transcript_v0.8.txt")
    # voc_dict, token2char = creat_voc(aishell_path)
    # pad_value = voc_dict["<pad>"]
    # now_vocab_size = len(voc_dict)

    #librispeech设置
    # sp = spm.SentencePieceProcessor(model_file="LibrispeechBepModel.model")
    # pad_value = sp.piece_to_id("<pad>")
    # now_vocab_size = 2500
    # train_dataset_1 = LibrispeechAsr("liber/", "liber/train-clean-360", "train_360", max_seq_len, max_tag_len)
    # train_dataset_2 = LibrispeechAsr("liber/", "liber/train-clean-100/LibriSpeech/train-clean-100", "train_100",
    #                                  max_seq_len, max_tag_len)
    # dev_clean_dataset = LibrispeechAsr("liber/", "liber/dev-clean/LibriSpeech/dev-clean", "dev_clean", max_seq_len,
    #                                    max_tag_len)
    # dev_other_dataset = LibrispeechAsr("liber/", "liber/dev-other/LibriSpeech/dev-other", "dev_other", max_seq_len,
    #                                    max_tag_len)
    # train_dataset = ConcatDataset([train_dataset_1, train_dataset_2])
    # dev_dataset = ConcatDataset([dev_other_dataset, dev_clean_dataset])

    # aishell设置
    # train_dataset = AishellDataset(path,"train",max_seq_len,max_tag_len,get_labels,voc_dict)
    # dev_dataset = AishellDataset(path,"dev",max_seq_len,max_tag_len,get_labels,voc_dict)

    # korean
    sp = spm.SentencePieceProcessor(model_file="Korean.model")
    pad_value = sp.piece_to_id("<pad>")
    now_vocab_size = 2000
    train_dataset = KoreanAsr("korean/", "korean/train_data_01", "train", max_seq_len, max_tag_len)
    dev_dataset = KoreanAsr("korean/", "korean/test_data_01", "test", max_seq_len, max_tag_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_woker)
    dev_loader = DataLoader(dev_dataset, batch_size=256, shuffle=True, num_workers=num_woker)


    #加载模型
    valid_model = ConformerAsr(d_model, n_head, linear_dim, num_encoder, vocab_size[1], droup) #修改vocab_size
    valid_model_name = "lr_model_valid_librispeech.pth"
    valid_model.load_state_dict(torch.load(valid_model_name))
    valid_model.change_outlayer(vocab_size[2])





    #librispeech korean
    ctc_loss = nn.CTCLoss(blank=sp.piece_to_id("-"))

    # aishell
    # ctc_loss = nn.CTCLoss(blank=voc_dict["-"])

    optimizer = torch.optim.AdamW(valid_model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader, valid_model.to(device), ctc_loss, optimizer, device)
        dev_loss(dev_loader, valid_model.to("cpu"), ctc_loss, "cpu")
        torch.save(valid_model.state_dict(), "lr_model_valid_korean.pth")
    print("Done!")

