
import torch
from torch import nn
from utils import *
import torch.optim as optim
from collections import defaultdict
from torch.utils.data import DataLoader,Dataset
import math
import argparse
from DADAN_Datasets import *  

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
def get_attn_subsequence_mask(seq):

    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask
def get_attn_pad_mask(seq_q, seq_k):

    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]
class ScaledDotProductAttention(nn.Module):
    def __init__(self,dk):
        super(ScaledDotProductAttention, self).__init__()
        self.dk=dk
    def forward(self, Q, K, V, attn_mask):

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.dk) # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,d_k,n_heads,d_v):
        super(MultiHeadAttention, self).__init__()
        self.d_model=d_model
        self.dk=d_k
        self.n_heads=n_heads
        self.dv=d_v
        self.W_Q = nn.Linear(self.d_model, self.dk * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.d_model, self.dk * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.d_model, self.dv * self.n_heads, bias=False)
        self.fc = nn.Linear(self.n_heads * self.dv, self.d_model, bias=False)
    def forward(self, input_Q, input_K, input_V, attn_mask):

        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.dk).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.dk).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.dv).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention(self.dk)(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1,(self.n_heads) * (self.dv)) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        
        return nn.LayerNorm(self.d_model).to(args.device)(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,d_model,d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model=d_model
        self.d_ff=d_ff
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d_model, bias=False)
        )
    def forward(self, inputs):

        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).to(args.device)(output + residual) # [batch_size, seq_len, d_model]

class EncoderLayer(nn.Module):
    def __init__(self,d_model,d_k,n_heads,d_v,d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model,d_k,n_heads,d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model,d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):

        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn
class classifer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(classifer, self).__init__()
        self.denseL1 = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        out = self.denseL1(x)
        return out
class Encoder_E(nn.Module):
    def __init__(self):
        super(Encoder_E, self).__init__()
        self.src_emb = nn.Embedding(args.src_vocab_size_e+1, args.d_model)
        self.pos_emb = PositionalEncoding(args.d_model)
        self.layers = nn.ModuleList([EncoderLayer(args.d_model,args.d_k_e,args.n_heads_e,args.d_v_e,args.d_ff_e) for _ in range(args.n_layers_e)])
        
    def forward(self, enc_inputs):
 
        enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]  
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # [batch_size, src_len, src_len]
        enc_attn_mask=get_attn_subsequence_mask(enc_inputs).to(args.device)
        all_mask=torch.gt((enc_self_attn_mask + enc_attn_mask), 0).to(args.device)
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, all_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs[:,-1,:]

def train_src(model,Classifier,seq_dataloader_a,seq_dataloader_val,optimizer,num_epochs,len_,args,criterion_md):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 20)
        for phase in ['train','val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            if phase == 'train':
                for idx,(seq_a,target_a) in enumerate(seq_dataloader_a):
                    # with torch.set_grad_enabled(phase == 'train'):
                    if torch.cuda.is_available():
                        seq_a=seq_a.to(args.device)
                        target_a=target_a.to(args.device)
                    optimizer.zero_grad()    
                    feat=model(seq_a)
                    classifer_=Classifier(feat)    
                    task_loss=criterion_md(classifer_,(target_a[:,0]-1))
                    if phase == 'train':
                        task_loss.backward()
                        optimizer.step()       
            if phase == 'val':
                with torch.no_grad():
                    r5_a = 0
                    m5_a = 0
                    r10_a = 0
                    m10_a = 0
                    r20_a = 0
                    m20_a = 0
                    for idx,(seq_a,target_a) in enumerate(seq_dataloader_val):
                        if torch.cuda.is_available():
                            seq_a=seq_a.to(args.device)
                            target_a=target_a.to(args.device)     
                        feat_a=model(seq_a)
                        classifer_=Classifier(feat_a)
                        recall, mrr=get_eval(classifer_,target_a,[5,10,20])
                        r5_a += recall[0]
                        m5_a += mrr[0]
                        r10_a += recall[1]
                        m10_a += mrr[1]
                        r20_a += recall[2]
                        m20_a += mrr[2]
                    print('Recall5_a: {:.5f}; Mrr5: {:.5f}'.format(r5_a/len_,m5_a/len_))
                    print('Recall10_a: {:.5f}; Mrr10: {:.5f}'.format(r10_a/len_,m10_a/len_))
                    print('Recall20_a: {:.5f}; Mrr20: {:.5f}'.format(r20_a/len_,m20_a/len_))
    return model,Classifier

  
if __name__  ==  '__main__':
    parser = argparse.ArgumentParser(description='DA-DAN with PyTorch')
    parser.add_argument('--d_model', type=int, help='Item embedding dimension',default=100)
    parser.add_argument('--d_v_e', type=int, default=100)
    parser.add_argument('--d_k_e', type=int, default=100)
    parser.add_argument('--src_vocab_size_e', type=int ,help='Total number of products in the source domain', default=3389)    #Select the number with the largest number of products in the two fields
    parser.add_argument('--n_heads_e', type=int, help='Number of attention headers in source domain', default=1)
    parser.add_argument('--n_layers_e',  type=int, help='Attention level of source domain', default=1)
    parser.add_argument('--d_ff_e', type=int, help="Source domain multilayer feedforward network parameters", default=2048)
    parser.add_argument('--maxlen', type=int, help="Maximum sequence length", default=50)
    parser.add_argument('--cla', type=int, help="Recommended item set, which is equal to the number of products in the source domain by default", default=3389)
    parser.add_argument('--device', help="GPU device", default=torch.device('cuda:1'))
    args = parser.parse_args()
    model=Encoder_E().to(args.device)
    Classifier=classifer(args.d_model,args.cla).to(args.device)
    optimizer = optim.Adam(list(model.parameters()) + list(Classifier.parameters()),lr=1e-3)
    seed = 608
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    criterion_md = nn.CrossEntropyLoss().to(args.device)
    data_train_A=TVdatasets_A('data/+A_domain_id.txt','data/B_domain_id.txt','data/+train.txt',args,'flag',5)
    data_val=TVdatasets_val('data/A_domain_id.txt','data/B_domain_id.txt','data/test.txt',args,'A_flag','B_flag',1)
    len_val=len(data_val)
    data_loader_train_A=DataLoader(data_train_A,batch_size=128,shuffle=True)
    data_loader_val=DataLoader(data_val,batch_size=128,shuffle=True)
    model_b, src_classifier = train_src(model,Classifier,data_loader_train_A,data_loader_val,optimizer,50,len_val,args,criterion_md)
    print('Loading model succeeded, testing......')
    with torch.no_grad():
        r5_a = 0
        m5_a = 0
        r10_a = 0
        m10_a = 0
        r20_a = 0
        m20_a = 0
        r5_b = 0
        m5_b = 0
        r10_b = 0
        m10_b = 0
        r20_b = 0
        m20_b = 0
        for idx,(seq_b,target_a) in enumerate(data_loader_val):
            if torch.cuda.is_available():
                seq_b=seq_b.to(args.device)
                target_a=target_a.to(args.device)     
            feat_B=model_b(seq_b)
            logits_B=src_classifier(feat_B)
            recall,mrr = get_eval(logits_B, target_a, [5,10,20])   
            r5_b += recall[0]
            m5_b += mrr[0]
            r10_b += recall[1]
            m10_b += mrr[1]
            r20_b += recall[2]
            m20_b += mrr[2]
        print('Recall5_b: {:.5f}; Mrr5: {:.5f}'.format(r5_b/len_val,m5_b/len_val))
        print('Recall10_b: {:.5f}; Mrr10: {:.5f}'.format(r10_b/len_val,m10_b/len_val))
        print('Recall20_b: {:.5f}; Mrr20: {:.5f}'.format(r20_b/len_val,m20_b/len_val))
  
