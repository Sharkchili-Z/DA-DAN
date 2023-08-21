from utils import *
from DADAN_Datasets import *
import os
import time
import torch
from torch import nn
import pandas as pd

import torch.optim as optim
from itertools import cycle
from utils import *
import copy
import datetime
import argparse


parser = argparse.ArgumentParser(description='DA-DAN with PyTorch')
parser.add_argument('--d_model', type=int, help='Item embedding dimension',default=100)
parser.add_argument('--d_v_e', type=int, default=100)
parser.add_argument('--d_v_v',  type=int, default=100)
parser.add_argument('--d_k_e', type=int, default=100)                                    
parser.add_argument('--d_k_v',  type=int, default=100)
parser.add_argument('--src_vocab_size_e', type=int ,help='Total number of products in the source domain', default=3389)  
parser.add_argument('--n_heads_e', type=int, help='Number of attention headers in source domain', default=1)      # It is required that n_heads_e times d_k_e equals d_model
parser.add_argument('--n_layers_e',  type=int, help='Attention level of source domain', default=1)
parser.add_argument('--d_ff_e', type=int, help="Source domain multilayer feedforward network parameters", default=2048)
parser.add_argument('--d_ff_v', type=int, help="Target domain multilayer feedforward network parameters", default=2048)
parser.add_argument('--src_vocab_size_v',  type=int ,help='Total number of products in the target domain', default=16432)
parser.add_argument('--n_heads_v', type=int, help='Number of attention headers in target domain', default=1)
parser.add_argument('--n_layers_v',  type=int, help='Attention level of target domain', default=1)
parser.add_argument('--maxlen', type=int, help="Maximum sequence length", default=50)
parser.add_argument('--en_feature_in', type=int, help="Encoder input dimension", default=100)
parser.add_argument('--de_feature_out', type=int, help="Decoder output dimension", default=100)
parser.add_argument('--hidden_dim', type=int,  default=4000)
parser.add_argument('--common_dim', type=int, default=4500)
parser.add_argument('--dynamic_tuning', type=bool, default=True)

parser.add_argument('--alpha', type=float, required=True)
parser.add_argument('--Beta', type=float,required=True)
parser.add_argument('--Gamma', type=float,required=True)

parser.add_argument('--cla', type=int, help="Recommended item set, which is equal to the number of products in the source domain by default", default=3389)
parser.add_argument('--qianyi', type=bool, help="embedding size of autoencoder", default=False)
parser.add_argument('--device', help="GPU device", default=torch.device('cuda:0'))
args = parser.parse_args()

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        
        return mse
class SIMSE(nn.Module):


    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)

        return simse

class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss

def calc_gan_loss(A_ture, A_fake,B_fake,B_ture ):
    bs=A_ture.shape[0]
    img_md = torch.ones(bs, dtype=torch.long).to(args.device)
    txt_md = torch.zeros(bs, dtype=torch.long).to(args.device)
    return criterion_md(A_ture, img_md) + criterion_md(A_fake, txt_md) + \
           criterion_md(B_fake, img_md) + criterion_md(B_ture, txt_md)

criterion_md = nn.CrossEntropyLoss().to(args.device)
cal_diff=DiffLoss().to(args.device)
rec_loss=MSE().to(args.device)



def train_model(model,seq_dataloader_a,seq_dataloader_b,seq_dataloader_val,optimizer,num_epochs,len_,args):
    best_cirtion=0.0
    
    
    dual_epoch = np.floor(20000 / len(seq_dataloader_a) * 1.0)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 20)
        i = 0
        for phase in ['train','val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            if phase == 'train':
                for idx,((seq_a,target_a),(seq_b)) in enumerate(zip(seq_dataloader_a,seq_dataloader_b)):
                    # with torch.set_grad_enabled(phase == 'train'):
                    if torch.cuda.is_available():
                        seq_a=seq_a.to(args.device)
                        target_a=target_a.to(args.device)
                        seq_b=seq_b.to(args.device)
                    optimizer.zero_grad()    
                    seq_A,seq_B,hinder_vertor_A,hinder_vertor_B,recon_A,recon_B,common_A,common_B,A_ture,A_fake,B_fake,B_ture,classifer_=model(seq_a,seq_b)
                    gan_loss=calc_gan_loss(A_ture, A_fake,B_fake,B_ture )

                    if args.dynamic_tuning == True:
                        alpha = float(i + (epoch - dual_epoch)  / (num_epochs - dual_epoch) )
                        alpha = 2. / (2 + np.exp(-1 * alpha)) - 1
                        args.alpha = alpha
                    diff_loss_A=cal_diff(common_A,hinder_vertor_A)
                    diff_loss_B=cal_diff(common_B,hinder_vertor_B)
                    diff_all=diff_loss_A+diff_loss_B
                    rec_loss_A=rec_loss(seq_A,recon_A)
                    rec_loss_B=rec_loss(seq_B,recon_B)
                    rec_loss_all=rec_loss_A+rec_loss_B
                    task_loss=criterion_md(classifer_,(target_a[:,0]-1))  
                    loss_all=task_loss+(args.alpha)*gan_loss+(args.Beta)*diff_all+(args.Gamma)*rec_loss_all 

                    if phase == 'train':
                        loss_all.backward()
                        optimizer.step()  
            if phase == 'val':
                with torch.no_grad():
                    r5_b = 0
                    m5_b = 0
                    r10_b = 0
                    m10_b = 0
                    r20_b = 0
                    m20_b = 0
                    for idx,(seq_b,target_a) in enumerate(seq_dataloader_val):
                        if torch.cuda.is_available():
                            seq_b=seq_b.to(args.device)
                            target_a=target_a.to(args.device)     
                        logits_B=model.predict(seq_b)
               
                        recall,mrr = get_eval(logits_B, target_a, [5,10,20])
                        r5_b += recall[0]
                        m5_b += mrr[0]
                        r10_b += recall[1]
                        m10_b += mrr[1]
                        r20_b += recall[2]
                        m20_b += mrr[2]
                    if (r20_b+m20_b)/2 > best_cirtion:
                        best_cirtion=(r20_b+m20_b)/2
                   
                        torch.save(model.state_dict(),'model/DA_DAN.pth')
          
                    print('Recall5_b: {:.5f}; Mrr5: {:.5f}'.format(r5_b/len_,m5_b/len_))
                    print('Recall10_b: {:.5f}; Mrr10: {:.5f}'.format(r10_b/len_,m10_b/len_))
                    print('Recall20_b: {:.5f}; Mrr20: {:.5f}'.format(r20_b/len_,m20_b/len_))
        i += 1

if __name__=='__main__':
    seed = 608
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    data_train_A=TVdatasets_A('data/A_domain_list.txt','data/B_domain_list.txt','data/A_domain_train.txt',args,'E',5)
    data_train_B=TVdatasets_B('data/A_domain_list.txt','data/B_domain_list.txt','data/B_domain_train.txt',args,'V',5)
    data_val=TVdatasets_val('data/A_domain_list.txt','data/B_domain_list.txt','data/val.txt',args,'E','V',1)
    data_test=TVdatasets_val('data/A_domain_list.txt','data/B_domain_list.txt','data/test.txt',args,'E','V',1)
    len_val=len(data_val)
    len_test=len(data_test)
    data_loader_train_A=DataLoader(data_train_A,batch_size=128,shuffle=True)
    data_loader_train_B=DataLoader(data_train_B,batch_size=128,shuffle=True)
    data_loader_val=DataLoader(data_val,batch_size=128,shuffle=True)
    data_loader_test=DataLoader(data_test,batch_size=128,shuffle=True)
    from DA_DAN import DADAN
    model=DADAN(args.en_feature_in,args.common_dim,args.de_feature_out,args.cla).to(args.device)
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_model(model,data_loader_train_A,data_loader_train_B,data_loader_val,optimizer,50,len_val,args)
    
    print('Loading model--------------')
 
    state_dict=torch.load('model/DA_DAN.pth')
    model.load_state_dict(state_dict)
    print('Loading model succeeded, testing.......................')
    with torch.no_grad():
        r5_b = 0
        m5_b = 0
        r10_b = 0
        m10_b = 0
        r20_b = 0
        m20_b = 0
        for idx,(seq_b,target_a) in enumerate(data_loader_test):
            if torch.cuda.is_available():
             
                seq_b=seq_b.to(args.device)
                target_a=target_a.to(args.device)     
            logits_B=model.predict(seq_b)
            recall,mrr = get_eval(logits_B, target_a, [5,10,20])
            r5_b += recall[0]
            m5_b += mrr[0]
            r10_b += recall[1]
            m10_b += mrr[1]
            r20_b += recall[2]
            m20_b += mrr[2]
        print('Recall5_b: {:.5f}; Mrr5: {:.5f}'.format(r5_b/len_test,m5_b/len_test))
        print('Recall10_b: {:.5f}; Mrr10: {:.5f}'.format(r10_b/len_test,m10_b/len_test))
        print('Recall20_b: {:.5f}; Mrr20: {:.5f}'.format(r20_b/len_test,m20_b/len_test))































    # train_model(model,data_loader_train_A,data_loader_train_B,data_loader_val,optimizer,20,len_val,args)
    
    # state_dict=torch.load('model/{},{},{}model_best.pth'.format(args.alpha,args.Beta,args.Gamma))
    # # state_dict=torch.load('model/0.01,0.01,0.01model_best.pth')
    # model.load_state_dict(state_dict)
    # print('加载模型成功，正在测试.......................')
    # with torch.no_grad():
    #     r5_a = 0
    #     m5_a = 0
    #     r10_a = 0
    #     m10_a = 0
    #     r20_a = 0
    #     m20_a = 0
    #     r5_b = 0
    #     m5_b = 0
    #     r10_b = 0
    #     m10_b = 0
    #     r20_b = 0
    #     m20_b = 0
    #     for idx,(seq_a,seq_b,target_a) in enumerate(data_loader_test):
    #         if torch.cuda.is_available():
    #             seq_a=seq_a.to(args.device)
    #             seq_b=seq_b.to(args.device)
    #             target_a=target_a.to(args.device)     
    #         logits_A,logits_B=model.predict(seq_a,seq_b)
    #         recall, mrr=get_eval(logits_A,target_a,[5,10,20])
    #         r5_a += recall[0]
    #         m5_a += mrr[0]
    #         r10_a += recall[1]
    #         m10_a += mrr[1]
    #         r20_a += recall[2]
    #         m20_a += mrr[2]
    #         recall,mrr = get_eval(logits_B, target_a, [5,10,20])   
    #         r5_b += recall[0]
    #         m5_b += mrr[0]
    #         r10_b += recall[1]
    #         m10_b += mrr[1]
    #         r20_b += recall[2]
    #         m20_b += mrr[2]
    #     print('Recall5_a: {:.5f}; Mrr5: {:.5f}'.format(r5_a/len_test,m5_a/len_test))
    #     print('Recall10_a: {:.5f}; Mrr10: {:.5f}'.format(r10_a/len_test,m10_a/len_test))
    #     print('Recall20_a: {:.5f}; Mrr20: {:.5f}'.format(r20_a/len_test,m20_a/len_test))
    #     print('Recall5_b: {:.5f}; Mrr5: {:.5f}'.format(r5_b/len_test,m5_b/len_test))
    #     print('Recall10_b: {:.5f}; Mrr10: {:.5f}'.format(r10_b/len_test,m10_b/len_test))
    #     print('Recall20_b: {:.5f}; Mrr20: {:.5f}'.format(r20_b/len_test,m20_b/len_test))