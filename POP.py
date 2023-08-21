import numpy as np
from torch.autograd import Function,Variable
import torch.nn as nn
import torch
import random
import torch.optim as optim
import torch.utils.data

from DADAN_Datasets import *

from DA_DAN import *
from train import args
from utils import *
def get_eval_pop(predlist, truelist, klist):#return recall@k and mrr@k
    recall = []
    mrr = []
    for k in klist:
        recall.append(0)
        mrr.append(0)
        templist = (predlist[:,:k])#the result of argsort is in ascending 
        i = 0
        while i < truelist.shape[0]:
            for j in range(truelist.shape[1]):
                pos = np.argwhere(templist[i]==(truelist[i,j]))
                if len(pos)>0:
                    break
            if len(pos) >0:
                recall[-1] += 1
                # mrr[-1] += 1/(k-pos[0][0])
                mrr[-1] += 1/(pos[0][0]+1)
            else:
                recall[-1] += 0
                mrr[-1] += 0
            i += 1
    return recall, mrr
Elist="data/A_domain_list.txt"
train='data/A_domain_train.txt'
def get_pop_from_train(Elist,train):
    def id_dict(fname):
        itemdict = {}
        with open(fname,'r') as f:
            items =  f.readlines()
        for item in items:
            item = item.strip().split('\t')
            itemdict[item[1]] = int(item[0])+1
        return itemdict
    itemdict=id_dict(Elist)
    dict_id={}
    with open(train, 'r') as f:  
        for line in f.readlines():
            line=line.strip().split('\t')
            for item in line[1:]:
                if item[0]=='E':
                    dict_id[item]=0
    with open(train, 'r') as f:  
        for line in f.readlines():
            line=line.strip().split('\t')
            for item in line[1:]:
                if item[0]=='E':
                    dict_id[item]=dict_id[item]+1
    paixu=sorted(dict_id.items(),key=lambda  x:x[1],reverse=True)
    prelist=[]
    for item in paixu:
        prelist.append(item[0])
        if len(prelist)>=100:
            break
    prelist_id=[]
    for item in prelist:
        prelist_id.append(itemdict[item])
    return prelist_id
prelist_id=np.array(get_pop_from_train(Elist,train)).reshape(1,100)
print(prelist_id)
r5_a = 0
m5_a = 0
r10_a = 0
m10_a = 0
r20_a = 0
m20_a = 0



data_val=TVdatasets_val('data/A_domain_list.txt','data/B_domain_list.txt',
                        'data/test.txt',args,'E','V',1)

data_loader_test=DataLoader(data_val,batch_size=128,shuffle=True)
for idx,(b,target_a) in enumerate(data_loader_test):
    target_a=target_a.numpy()
    temp=np.broadcast_to(prelist_id,(target_a.shape[0],prelist_id.shape[1]))
    recall, mrr=get_eval_pop(temp,target_a,[5,10,20])
    r5_a += recall[0]
    m5_a += mrr[0]
    r10_a += recall[1]
    m10_a += mrr[1]
    r20_a += recall[2]
    m20_a += mrr[2]
print('Recall5_a: {:.5f}; Mrr5: {:.5f}'.format(r5_a/len(data_val),m5_a/len(data_val)))
print('Recall10_a: {:.5f}; Mrr10: {:.5f}'.format(r10_a/len(data_val),m10_a/len(data_val)))
print('Recall20_a: {:.5f}; Mrr20: {:.5f}'.format(r20_a/len(data_val),m20_a/len(data_val)))
