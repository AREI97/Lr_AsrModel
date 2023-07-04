import os


with open("data_aishell/data_aishell/transcript/aishell_transcript_v0.8.txt","r",encoding="utf-8") as fr:
    fw = open("aishell.txt","w",encoding="utf-8")
    for substr in fr.readlines():
        idx = 0
        while substr[idx] != " ":
            idx += 1
        idx += 1
        now_substr = substr[idx:]
        fw.write(now_substr)

    fw.close()
