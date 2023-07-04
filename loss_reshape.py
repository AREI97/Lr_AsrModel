import torch

def loss_reshape(pred,label):
    for row in range(len(label)):
        for col in label[row]:
            