
#如果要生成新数据集的txt，需要先删除原来的txt文件
#类似librispeech数据集结构的数据集都可以用此脚本生成对应的文本txt文件
import os
def creat_txt(main_path,txt_type):


    files = os.listdir(main_path)
    for file in files:
        now_path = os.path.join(main_path,file)
        sub_files = os.listdir(now_path)
        for sub_file in sub_files:
            now_sub_path = os.path.join(now_path,sub_file)
            now_txt_name = sub_file + "_" + file + ".trans.txt"
            now_txt_path = os.path.join(now_sub_path,now_txt_name)
            token_list = []
            txt_list = []
            with open(now_txt_path,"r",encoding="utf-8") as fr:
                trans = fr.readlines()
                for token in trans:
                    for i in range(len(token)):
                        if token[i] != " ":
                            i += 1
                        else:
                            get_token = token[0:i]
                            get_txt = token[i+1:]
                            token_list.append(get_token)
                            txt_list.append(get_txt)
                            break

            write_path = "korean/" + txt_type + ".txt"
            with open(write_path,"a") as fw:
                for token in token_list:
                    fw.write(token+"\n")
            write_path = "korean/" + txt_type + "_seq.txt"
            with open(write_path,"a",encoding="utf-8") as fw:
                for txt in txt_list:
                    fw.write(txt)

train_main_path_1 = r"D:\learn\learn\asr_socr\korean\train_data_01"
# train_main_path_2 = "liber/train-clean-100/LibriSpeech/train-clean-100"
#
test_clean_main_path = r"D:\learn\learn\asr_socr\korean\test_data_01"
#
# test_other_main_path = "liber/test-other/LibriSpeech/test-other"
#
# dev_clean_main_path = "liber/dev-clean/LibriSpeech/dev-clean"
#
# dev_other_main_path = "liber/dev-other/LibriSpeech/dev-other"

creat_txt(train_main_path_1,"train")
#creat_txt(train_main_path_2,"train_100")
#creat_txt(test_other_main_path,"test_other")
creat_txt(test_clean_main_path,"test")
#creat_txt(dev_clean_main_path,"dev_clean")
#creat_txt(dev_other_main_path,"dev_other")
with open('korean/tokenizertxt.txt', 'w',encoding="utf-8") as outfile:
    for file_name in ['korean/train_seq.txt',
                      'korean/test_seq.txt',]:
        with open(file_name,"r",encoding="utf-8") as infile:
            outfile.write(infile.read())





