import nltk
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline
from nltk.lm import MLE
from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import word_tokenize
import jieba
import pickle
from lmtrain import get_lm
from collections import defaultdict
path = "aishell.txt"
ngram_model = get_lm(path)

n = 2
# 对新句子进行评分
test_sentence = '而对楼市成交作用'
test_words = jieba.lcut(test_sentence)
#test_words = test_sentence.lower().split()

score = 0
for i in range(len(test_words) - n + 1):
    ngram = ' '.join(test_words[i:i+n])
    score += ngram_model[ngram]

print('Score:', score)