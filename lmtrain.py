import nltk
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline
from nltk.lm import MLE,Laplace
from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import word_tokenize
import jieba
import pickle
from collections import defaultdict
# # 读取本地文本库的文件路径
#
# # 创建语料库读取器，使用PlaintextCorpusReader类
# #corpus = PlaintextCorpusReader(r"D:\learn\learn\asr_socr",'aishell.txt')
# corpus = open("aishell.txt", "r", encoding="utf-8").read()
# words = jieba.lcut(corpus,cut_all=False, HMM=False)
# cleaned_words = [word.strip() for word in words if word.strip() != "" and word.strip() != "\n"]
# # 获取语料库中的文本数据
# #text_data = corpus.raw()
#
# # 进行文本预处理，例如分词
# #tokens = word_tokenize(text_data)
#
# # 准备数据（示例）
# #text_data = "This is a sample sentence. Another sentence."
#
# # 数据预处理（示例）
# #tokens = text_data.split()  # 分词
#
# # 构建n-gram语言模型（示例）
# n = 3  # 选择n-gram的大小
# #train_data, padded_sents = padded_everygram_pipeline(n, tokens)
# data, padded_sents = padded_everygram_pipeline(n, text=cleaned_words)
# model = Laplace(n)  # 使用MLE（最大似然估计）训练模型
# model.fit(data, padded_sents)
#
# # 保存模型到文件
# with open('aishell_lm_model.pkl', 'wb') as f:
#     pickle.dump(model, f)

def get_lm(path):
    # 训练数据
    corpus = open(path, "r", encoding="utf-8").read()
    words = jieba.lcut(corpus, cut_all=False, HMM=False)
    cleaned_words = [word.strip() for word in words if word.strip() != "" and word.strip() != "\n"]

    n = 2  # n-gram的n值

    # 初始化n-gram模型
    ngram_model = defaultdict(int)

    # 训练模型
    # for sentence in sentences:
    #     words = sentence.lower().split()  # 分词并转换为小写
    for i in range(len(cleaned_words) - n + 1):
        ngram = ' '.join(cleaned_words[i:i + n])
        ngram_model[ngram] += 1

    return ngram_model
