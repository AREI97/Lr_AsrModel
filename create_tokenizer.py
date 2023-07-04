import sentencepiece as spm

input_file = "liber/tokenizertxt.txt"

model_prefix = "librispeechBepModel5000"
vocab_size = 5000

spm.SentencePieceTrainer.train(input=input_file,
                               model_prefix=model_prefix,
                               model_type="bpe",
                               vocab_size=vocab_size,
                               user_defined_symbols=["-", "<pad>"]
                               )

