import sentencepiece as spm

import os

sp = spm.SentencePieceProcessor()
sp.load("LibrispeechBepModel.model")
print(sp.encode("",out_type=int))
print(sp.decode(sp.encode("|")))
