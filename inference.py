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

def calculate_cer(reference, hypothesis):
    reference = reference.replace(' ', '')
    hypothesis = hypothesis.replace(' ', '')

    error_count = Levenshtein.distance(reference, hypothesis)
    total_count = len(reference)

    cer = error_count / total_count
    return cer


def calculate_wer(reference, hypothesis):
    """
    计算单词错误率（WER）
    :param reference: 参考文本（字符串）
    :param hypothesis: 预测文本（字符串）
    :return: WER（浮点数）
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    # 计算编辑距离
    distance = Levenshtein.distance(ref_words, hyp_words)

    # 计算WER
    wer = distance / len(ref_words)

    return wer


# def get_best_str(results,nbest,n,ngram_model,token2char=None):
#     best_score_index = 0
#     best_score = 0
#     for k in range(nbest):
#        sentence = aishell_id_to_piece(results[0][k].tokens.tolist(), token2char)
#        test_words = jieba.lcut(sentence)
#        # test_words = test_sentence.lower().split()
#
#        score = 0
#        for i in range(len(test_words) - n + 1):
#            ngram = ' '.join(test_words[i:i + n])
#            score += ngram_model[ngram]
#        if best_score < score:
#            best_score = score
#            best_score_index = k
#
#     return aishell_id_to_piece(results[0][best_score_index].tokens.tolist(), token2char)

def decode_func(model_type, model, dataloader, device, decoder, token_type, sp, soce_type, token2char):
    dataset_dict = {"aishell": 0, "librispeech": 1,"korean": 2}
    acc = 0
    count = 0
    # nbest = 20
    # n = 2
    for batch, (src, tgt, src_key_padding_mask, tgt_key_padding_mask) in enumerate(dataloader):
        if model_type == "train":
            preds, pred_pad_len, pre_id = model.inference(src.to(device),
                                                          src_key_padding_mask.to(device),
                                                          device,
                                                          )
        else:
            preds, pred_pad_len = model(src.to(device),
                                        src_key_padding_mask.to(device),
                                        device)
        now_idx = 0
        for pred in preds:
            if model_type == "train" and pre_id[now_idx] != dataset_dict[token_type]:
                acc += 1
                now_idx += 1
                continue
            elif model_type == "valid":
                pred = torch.unsqueeze(pred,dim=0)
            results = decoder(pred)
            now_tgt = tgt[now_idx]
            now_tgt = now_tgt[tgt_key_padding_mask[now_idx] == False]
            now_tgt = now_tgt.tolist()
            if token_type == "aishell":
                result = aishell_id_to_piece(results[0][0].tokens.tolist(), token2char)
                result = result[:-1]
                now_tgt = aishell_id_to_piece(now_tgt, token2char)
            else:
                result = sp.decode(results[0][0].tokens.tolist())
                result = result[:-1]
                now_tgt = sp.decode(now_tgt)
            if soce_type == "wer":
                acc += calculate_wer(now_tgt,result)
            else:
                acc += calculate_cer(now_tgt,result)
            count += 1
            now_idx += 1
        print("now batch is :", batch, " now acc is : ", acc / count)
    return acc / count


if __name__ == "__main__":
    batch_size = 64
    num_woker = 8
    max_seq_len = 1200
    max_tag_len = 120
    # 1 stage:加载test数据集
    # 1.1加载aishell数据集
    path = r"data_aishell\data_aishell"
    aishell_path = os.path.join(path, r"transcript\aishell_transcript_v0.8.txt")
    voc_dict, token2char = creat_voc(aishell_path)
    pad_value = voc_dict["<pad>"]
    now_vocab_size = len(voc_dict)
    aishell_test_dataset = AishellDataset(path, "test", max_seq_len, max_tag_len, get_labels, voc_dict)
    aishell_dev_dataset = AishellDataset(path, "dev", max_seq_len, max_tag_len, get_labels, voc_dict)
    aishell_test_dataset = ConcatDataset([aishell_test_dataset, aishell_dev_dataset])
    aishell_test_loader = DataLoader(aishell_test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_woker)

    # 1.2加载librispeech数据集
    libri_sp = spm.SentencePieceProcessor(model_file="LibrispeechBepModel.model")
    pad_value = libri_sp.piece_to_id("<pad>")
    now_vocab_size = 2500
    test_dataset_1 = LibrispeechAsr("liber/",
                                    "liber/test-clean/LibriSpeech/test-clean",
                                    "test_clean",
                                    max_seq_len,
                                    max_tag_len)
    test_dataset_2 = LibrispeechAsr("liber/",
                                    "liber/test-other/LibriSpeech/test-other",
                                    "test_other",
                                    max_seq_len,
                                    max_tag_len)
    libri_dev_clean_dataset = LibrispeechAsr("liber/",
                                             "liber/dev-clean/LibriSpeech/dev-clean",
                                             "dev_clean",
                                             max_seq_len,
                                             max_tag_len)
    libri_dev_other_dataset = LibrispeechAsr("liber/",
                                             "liber/dev-other/LibriSpeech/dev-other",
                                             "dev_other",
                                             max_seq_len,
                                             max_tag_len)
    libri_test_dataset = ConcatDataset([test_dataset_1, test_dataset_2])
    libri_dev_dataset = ConcatDataset([libri_dev_other_dataset, libri_dev_clean_dataset])
    libri_test_dataset = ConcatDataset([libri_test_dataset, libri_dev_dataset])
    libri_test_loader = DataLoader(libri_test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_woker)

    # 1.3加载korean数据集
    korean_sp = spm.SentencePieceProcessor(model_file="Korean.model")
    pad_value = korean_sp.piece_to_id("<pad>")
    now_vocab_size = 2000
    korean_dev_dataset = KoreanAsr("korean/", "korean/test_data_01", "test", max_seq_len, max_tag_len)
    korean_test_loader = DataLoader(korean_dev_dataset, batch_size=batch_size, shuffle=True, num_workers=num_woker)

    # 2 stage 加载模型

    device = "cpu"
    now_audio_nums = [0, 1, 2]
    d_model = 256
    num_head = 4
    num_hide = 2048
    num_encoder = 8
    model_name = "lr_model_s2v1.pth"
    vocab_size = [4336, 2500, 2000]
    droup = 0.1


    model = LrAsr(d_model, num_head, num_hide, num_encoder, vocab_size, now_audio_nums, device, droup)
    model.load_state_dict(torch.load(model_name))
    model.eval()
    #2.1 stage 加载对照组模型
    validVocSize = 2000

    valid_model = ConformerAsr(d_model, num_head, num_hide, num_encoder, validVocSize, droup)
    valid_model_name = "lr_model_valid_korean.pth"
    valid_model.load_state_dict(torch.load(valid_model_name))
    # valid_model.change_outlayer(vocab_size[1])
    #
    # asr_model_dict = model.baseModelList[1].state_dict()
    #
    # new_state_dict = {}
    # for name, param in asr_model_dict.items():
    #     if name.startswith('outlayer'):
    #         new_state_dict[name] = param
    # valid_model.load_state_dict(new_state_dict, strict=False)
    valid_model.eval()

    # 3 stage 加载解码器

    aishell_decoder = ctc_decoder(lexicon=None, tokens=list(voc_dict.keys()), sil_token="-", beam_size=10)
    libri_decoder = ctc_decoder(lexicon=None, tokens="LibrispeechBepModel.txt", sil_token="-", beam_size=10)
    korean_decoder = ctc_decoder(lexicon=None, tokens="korean.txt", sil_token="-", beam_size=10)
    # ngram_model = get_lm("aishell.txt")
    model_type = "valid"

    # 4 stage 解码
    result = decode_func(model_type, valid_model,korean_test_loader, device, korean_decoder, "korean", korean_sp, "cer",
                                                                            # 一次训练 aishell cer 12.4
                                                             token2char)    # 两次训练 aishell cer 13.6 librispeech cer 18.4
                                                                            # 三次训练 aishell cer 13.5 librispeech cer 19.2    korean 15.2
                                                                            # 四次训练
                                                                            # 对照组
                                                                            # 第一次训练 aishell cer 12.4
                                                                            # 第二次训练 aishell cer 99.8  librispeech cer 15.6
                                                                            # 第三次训练 aishell cer 107   librispeech cer 90.4 korean 16.7
    print(result)
