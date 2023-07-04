#该模块用于训练增量学习模型的classifer模块


import torch
from FeatIdDataset import FeatIdDataset
from torch import nn
from creat_voc import creat_voc
from get_labels import get_labels
import os
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset,SubsetRandomSampler
from lr_asr import LrAsr
# step 1 加载数据集
if __name__ == '__main__':
    device = "cuda"
    batch_size = 32
    num_workers = 0
    lr = 1e-2
    now_audio_nums = [0,1,2] #已经训练好的基础模型
    tag_id = 0 #当前数据集标签，为0时默认为aishell数据集
    root = ["data_aishell/data_aishell","liber","korean"] #各个模型对应的数据集路径
    main_path = [None,"liber/train-clean-360","korean/train_data_01"] #librispeech风格数据集需要
    max_seq_len = 1200
    bpemodelname = [None,"LibrispeechBepModel.model","Korean.model"] #各数据集对应的bpe模型
    datasetlist = [] #加载的数据集列表
    data_types = ["train","train_360","train"] #txt文件的名称
    aishell_path = os.path.join(root[0],r"transcript\aishell_transcript_v0.8.txt")

    for audio_nums in now_audio_nums:
        if tag_id == 0:
            vocab_dict,__ = creat_voc(aishell_path)
            label_func = get_labels
        else:
            label_func = None
            vocab_dict = None

        datasetlist.append(FeatIdDataset(root[tag_id],
                                         data_types[tag_id],
                                         max_seq_len,
                                         tag_id,
                                         bpemodelname[tag_id],
                                         main_path[tag_id],
                                         label_func,
                                         vocab_dict))
        tag_id += 1
    train_index = list(range(20000))
    dataset = ConcatDataset([Subset(x,train_index) for x in datasetlist])
    dataset_size = len(dataset)
    train_size = int(0.8*dataset_size)
    val_size = dataset_size - train_size

    indices = torch.randperm(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_dataloader = DataLoader(dataset,batch_size=batch_size,num_workers=num_workers,sampler=train_sampler)
    val_dataloader = DataLoader(dataset,batch_size=batch_size,num_workers=num_workers,sampler=val_sampler)

    #step 2 :加载模型
    d_model = 256
    num_head = 4
    num_hide = 2048
    num_encoder = 8
    model_name = "lr_model_korean.pth"
    vocab_size = [4336,5000,2000]
    droup = 0.1
    model = LrAsr(d_model,num_head,num_hide,num_encoder,vocab_size,now_audio_nums,device,droup)
    model.load_state_dict(torch.load(model_name))
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)

    loss_fun = nn.CrossEntropyLoss()

    #step 3 :开始训练
    epoch = 1000
    train_size = len(train_dataloader.dataset)
    model.train()
    model.to(device)
    val_size = len(val_dataloader.dataset)

    for current in range(epoch):
        acc = 0
        count = 0
        loss_value = 0
        model.train()
        for batch,(src,label) in enumerate(train_dataloader):
            pre = model.vote_train(src.to(device).requires_grad_(True))
            loss = loss_fun(pre,label.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                loss_value += loss.item()
                count += 1
                pre = nn.functional.softmax(pre,dim=-1)
                pre = pre.argmax(-1)
                acc += (label.to(device) == pre).sum() / len(label)

        print(f"train_loss: {loss_value/count:>10f} acc: {acc/count:>10f}  [{current+1:>5d}/{epoch:>5d}]")
        model.eval()
        acc = 0
        count = 0
        loss_value = 0
        for batch,(src,label) in enumerate(val_dataloader):
            pre = model.vote_train(src.to(device))
            loss = loss_fun(pre,label.to(device))
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            with torch.no_grad():
                loss_value += loss.item()
                count += 1
                pre = nn.functional.softmax(pre,dim=-1)
                pre = pre.argmax(-1)

                acc += (label.to(device) == pre).sum() / len(label)

        print(f"val_loss:  {loss_value/count:>10f} acc: {acc/count:>10f}  [{current+1:>5d}/{epoch:>5d}]")
        torch.save(model.state_dict(), "lr_model_s2.pth")











