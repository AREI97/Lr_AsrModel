from creat_voc import creat_voc 
import numpy as np
import os


def get_labels(path,voc_dict,wav_path,data_type):
    labels_dict = dict()
    type_wav_path = os.path.join(wav_path,data_type)
    dirs = os.listdir(type_wav_path)
    type_set = set(dirs)
    with open(path,"r",encoding="UTF-8") as f:
       
        for line in f.readlines():
            voc_key = line[:16]
            if voc_key[6:11] not in type_set:
                continue
            labels_dict[voc_key] = []
            for char in line[16:]:
                if char != ' ' and char != '\n':
                    char_token = voc_dict[char]
                    labels_dict[voc_key].append(char_token)

    
    print("the labels number is :{}".format(len(labels_dict)))
    return labels_dict

            
            
