import torch
import os
from creat_voc import creat_voc
from conformer_asr import ConformerAsr
from lr_asr import LrAsr
from asr_pre import AishellDataset
from get_labels import get_labels
import scipy.io.wavfile as wav
from python_speech_features import logfbank
from torchaudio.models.decoder import ctc_decoder
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.data import ConcatDataset
import sentencepiece as spm
import Levenshtein  # 需要安装python-levenshtein包
from librispeech import LibrispeechAsr
from aishell_id_to_piece import aishell_id_to_piece
from lmtrain import get_lm
import jieba
from conformer_asr import ConformerAsr
from Korean import KoreanAsr

device = "cpu"
now_audio_nums = [0, 1, 2]
d_model = 256
num_head = 4
num_hide = 2048
num_encoder = 8
model_name = "lr_model_s2.pth"
vocab_size = [4336, 5000, 2000]
droup = 0.1

model = LrAsr(d_model, num_head, num_hide, num_encoder, vocab_size, now_audio_nums, device, droup)
model.load_state_dict(torch.load(model_name))
model.baseModelList[1].change_outlayer(2500)
model.baseModelList[1].load_state_dict(torch.load("lr_model_valid_librispeech.pth"))
torch.save(model.state_dict(), "lr_model_s2v1.pth")