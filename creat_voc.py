import os

def creat_voc(aishell_path):
    voc_set = set()
    char2token = dict()
    token2char = dict()
    char_num = 0
    char2token['-'] = 0
    token2char[0] = '-'
    with open(aishell_path,"r",encoding="UTF-8") as f:
        for line in f.readlines():
            subline = line[16:]
            for char in subline:
                if char != ' ' and char not in voc_set and char != '\n':
                    char_num += 1
                    voc_set.add(char)
                    char2token[char] = char_num
                    token2char[char_num] = char
    char_num += 1
    char2token["<pad>"] = char_num
    token2char[char_num] = "<pad>"
    char_num += 1
    char2token["<unk>"] = char_num
    token2char[char_num] = "<unk>"
    print("the char nums is :{}".format(len(char2token)))
    return char2token, token2char

    
                    