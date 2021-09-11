# -*- coding: utf-8 -*-
"""
Created on Sun May 19 03:31:58 2019

@author: liang
"""
#%%
import numpy as np
import torch
from torch import optim
import torch.nn.functional as nnF
#%%
def cal_performance(confusion):
    num_classes=confusion.shape[0]
    acc = confusion.diagonal().sum()/confusion.sum()
    sens=np.zeros(num_classes)
    prec=np.zeros(num_classes)
    for m in range(0, num_classes):
        sens[m]=confusion[m,m]/(np.sum(confusion[m,:])+1e-8)
        prec[m]=confusion[m,m]/(np.sum(confusion[:,m])+1e-8)
    return acc, sens, prec
#%%
def update_confusion(confusion, Y, Yp):
    Y=Y.detach().cpu().numpy()
    Yp=Yp.detach().cpu().numpy()
    num_classes=confusion.shape[0]
    if num_classes <= Y.shape[0]:
        for i in range(0, num_classes):
            for j in range(0, num_classes):
                confusion[i,j]+=np.sum((Y==i)&(Yp==j))
    else:
        for n in range(0, Y.shape[0]):
            confusion[Y[n],Yp[n]]+=1
#%%
def test(model, device, dataloader, num_classes, run_model=None):
    model.eval()#set model to evaluation mode
    sample_count=0
    sample_idx_wrong=[]
    confusion=np.zeros((num_classes,num_classes))
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            X, Y = batch_data[0].to(device), batch_data[1].to(device)
            if run_model is None:
                Z = model(X)#forward pass
            else:
                Z = run_model(model, X)
            if len(Z.size()) <= 1:
                Yp = (Z.data>0).to(torch.int64) #binary/sigmoid
            else:
                Yp = Z.data.max(dim=1)[1] #multiclass/softmax
            update_confusion(confusion, Y, Yp)
            #------------------
            for n in range(0,X.size(0)):
                if Y[n] != Yp[n]:
                    sample_idx_wrong.append(sample_count+n)
            sample_count+=X.size(0)
    #------------------
    acc, sens, prec = cal_performance(confusion)
    result={}
    result['confusion']=confusion
    result['acc']=acc
    result['sens']=sens
    result['prec']=prec
    result['sample_idx_wrong']=sample_idx_wrong
    print('testing')
    print('acc', result['acc'])
    print('sens', result['sens'])
    print('prec', result['prec'])
    return result
#%%