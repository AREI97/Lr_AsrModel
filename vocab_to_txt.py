import os

path = "korean.vocab"
txt_path = "korean.txt"
with open(path,"r",encoding="utf-8") as fr:
    str_list = fr.readlines()
    fw = open(txt_path,"w",encoding="utf-8")
    for nowstr in str_list:
        substr = nowstr.split("\t")[0]
        substr = substr+"\n"
        fw.write(substr)
    fw.close()

