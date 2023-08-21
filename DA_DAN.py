import math
from re import I
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from train import args
import torch.nn.functional as F
from torch.autograd import Function
from typing import Any, Optional, Tuple
class GradientReverseFunction(Function):
        """
        Override custom gradient calculation method
        """
        @staticmethod
        def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.0) -> torch.Tensor:
            ctx.coeff = coeff
          
            output = input * 1.0
    
            return output
        @staticmethod
        def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
           
            return grad_output.neg() * ctx.coeff, None
class GRL_Layer(nn.Module):
    def __init__(self):
        super(GRL_Layer, self).__init__()
    def forward(self, *input):
        return GradientReverseFunction.apply(*input)
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
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask
def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    '''
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
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
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
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
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
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).to(args.device)(output + residual) # [batch_size, seq_len, d_model]

class EncoderLayer(nn.Module):
    def __init__(self,d_model,d_k,n_heads,d_v,d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model,d_k,n_heads,d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model,d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
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
        
    def forward(self, enc_inputs,for_e):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        if args.qianyi==True:
            enc_outputs=enc_outputs+for_e.unsqueeze(1)   
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # [batch_size, src_len, src_len]
        enc_attn_mask=get_attn_subsequence_mask(enc_inputs).to(args.device)
        all_mask=torch.gt((enc_self_attn_mask + enc_attn_mask), 0).to(args.device)
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, all_mask)
            # enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        
        
        return enc_outputs[:,-1,:]
        # return enc_outputs.sum(1)/args.maxlen
class Encoder_V(nn.Module):
    def __init__(self):
        super(Encoder_V, self).__init__()
        self.src_emb = nn.Embedding(args.src_vocab_size_v+1, args.d_model)
        self.pos_emb = PositionalEncoding(args.d_model)
        self.layers = nn.ModuleList([EncoderLayer(args.d_model,args.d_k_v,args.n_heads_v,args.d_v_v,args.d_ff_v) for _ in range(args.n_layers_v)])
        
    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # [batch_size, src_len, src_len]
        enc_attn_mask=get_attn_subsequence_mask(enc_inputs).to(args.device)
        all_mask=torch.gt((enc_self_attn_mask + enc_attn_mask), 0).to(args.device)
        
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, all_mask)
            # enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        # enc_outputs=self.class_(enc_outputs[:,-1,:])
        
        
        return enc_outputs[:,-1,:]
        # return enc_outputs.sum(1)/args.maxlen


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * -ctx.alpha

        return output, None
class DADAN(nn.Module):
    def __init__(self,en_feature_in,common_dim,de_feature_out,num_class):
        super(DADAN,self).__init__()
        self.seuence_extract_A=Encoder_E()
        self.seuence_extract_B=Encoder_V()

        self.encoder_common=nn.Sequential(
           
            nn.Linear(in_features=en_feature_in, out_features=common_dim),
            nn.ReLU(True)
            )
        self.decoder_A=nn.Sequential(
            nn.Linear(in_features=common_dim, out_features=de_feature_out),
            nn.ReLU(True)
        )
        self.encoder_B=nn.Sequential(
            nn.Linear(in_features=en_feature_in, out_features=common_dim),
            nn.ReLU(True)
            )
        self.decoder_B=nn.Sequential(
            nn.Linear(in_features=common_dim, out_features=de_feature_out),
            nn.ReLU(True)
            )
        self.project_common=nn.Sequential(
            # nn.BatchNorm1d(args.common_dim),
            nn.Linear(in_features=common_dim, out_features=common_dim),
            nn.ReLU(True)
            )
        self.genecator_A=nn.Sequential(
            # nn.BatchNorm1d(args.common_dim),
            nn.Linear(in_features=common_dim, out_features=common_dim),
            nn.ReLU(True)
            )
        self.genecator_B=nn.Sequential(
            # nn.BatchNorm1d(args.common_dim),
            nn.Linear(in_features=common_dim, out_features=common_dim),
            nn.ReLU(True)
            )
        self.domainClassifier_A=nn.Sequential(
            GRL_Layer(),
            nn.Linear(in_features=common_dim, out_features=100), 
            nn.ReLU(True),
            nn.Linear(in_features=100, out_features=2)
            )
        self.domainClassifier_B=nn.Sequential(
            GRL_Layer(),
            nn.Linear(in_features=common_dim, out_features=100), 
            nn.ReLU(True),
            nn.Linear(in_features=100, out_features=2)
            )
        # self.taskClassifier=nn.Sequential(
        #     nn.Linear(in_features=common_dim, out_features=num_class),   ##################################
        # )
        self.taskClassifier=nn.Sequential(
            nn.Linear(in_features=common_dim, out_features=5000), 
            # nn.BatchNorm1d(5000), ##################################
            nn.Linear(in_features=5000, out_features=num_class),
        )
    def forward(self,seq_A,seq_B):
        seq_B=self.seuence_extract_B(seq_B)
        seq_A=self.seuence_extract_A(seq_A,seq_B)
        
        hinder_vertor_A=self.encoder_common(seq_A)
        hinder_vertor_B=self.encoder_common(seq_B)
        recon_A=self.decoder_A(hinder_vertor_A)
        recon_B=self.decoder_B(hinder_vertor_B)
        common_A=self.project_common(hinder_vertor_A)
        common_B=self.project_common(hinder_vertor_B)
        fake_data_from_A=self.genecator_A(common_A)
        fake_data_from_B=self.genecator_B(common_B)
        A_ture=self.domainClassifier_A(hinder_vertor_A)
        A_fake=self.domainClassifier_A(fake_data_from_B)
        B_fake=self.domainClassifier_B(fake_data_from_A)
        B_ture=self.domainClassifier_B(hinder_vertor_B)
        classifer_=self.taskClassifier(common_A)
        return seq_A,seq_B,hinder_vertor_A,hinder_vertor_B,recon_A,recon_B,common_A,common_B,\
        A_ture,A_fake,B_fake,B_ture,classifer_
    def predict(self,seq_B):
        '''
        When predicting, the sequence level representation is obtained from the attention network of the target domain. 
        Through the encoder, the MLP layer obtains the representation of the final public space, 
        which is sent to the scorer of the source domain training 
        to recommend the goods of the source domain to the users of the target domain
        '''
        seq_B=self.seuence_extract_B(seq_B)
        
        hinder_vertor_B=self.encoder_common(seq_B)
        common_B=self.project_common(hinder_vertor_B)
        classifer_B=self.taskClassifier(common_B)
        return classifer_B