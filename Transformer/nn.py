import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class selfAttention(nn.Module):
    def __init__(self,dim_in,dim_k,dim_v):
        super().__init__()
        self.dim_in,self.dim_k,self.dim_v=dim_in,dim_k,dim_v
        self.W_q=nn.Linear(dim_in,dim_k)
        self.W_k=nn.Linear(dim_in,dim_k)
        self.W_v=nn.Linear(dim_in,dim_v)
        self.norm=1/sqrt(dim_k)
        self.output=nn.Linear(dim_v,dim_in)

    def forward(self,X,mask=None):
        # X: B*L*in
        Q,K,V=self.W_q(X),self.W_k(X),self.W_v(X) # B*L*K/V
        score=torch.matmul(Q,K.transpose(-2,-1)) *self.norm # B*L*L

        if mask is not None:
            score=score.masked_fill(mask==0,-1e9)

        weight=F.softmax(score,dim=-1) # B*L*L
        attention=torch.matmul(weight,V) # B*L*V
        output=self.output(attention) # B*L*in

        return weight , output
    
class MultiHeadAttention(nn.Module):
    def __init__(self,dim_in,dim_k,dim_v,n_head):
        super().__init__()
        self.dim_in,self.dim_k,self.dim_v=dim_in,dim_k,dim_v
        self.n_head=n_head
        self.W_q=nn.Linear(dim_in,dim_k)
        self.W_k=nn.Linear(dim_in,dim_k)
        self.W_v=nn.Linear(dim_in,dim_v)
        self.norm=1/sqrt(dim_k/n_head)
        self.output=nn.Linear(dim_v,dim_in)

    def forward(self,X,mask=None):
        # X: B*L*in
        batch,seq_len,embed=X.shape
        Q,K,V=self.W_q(X),self.W_k(X),self.W_v(X) # B*L*nK/nV
        # 分头：B*n*L*K/V
        Q=Q.view(batch,seq_len,self.n_head,-1).transpose(1,2)
        K=K.view(batch,seq_len,self.n_head,-1).transpose(1,2)
        V=V.view(batch,seq_len,self.n_head,-1).transpose(1,2)
        score=torch.matmul(Q,K.transpose(-2,-1)) *self.norm # B*n*L*L

        if mask is not None:
            if mask.dim() == 2:  # L*L -> B*n*L*L
                mask = mask.unsqueeze(0).unsqueeze(0).expand(batch, self.n_head, -1, -1)
            elif mask.dim() == 3:  # B*L*L -> B*n*L*L
                mask = mask.unsqueeze(1).expand(-1, self.n_head, -1, -1)
            score = score.masked_fill(mask == 0, -1e9)


        weight=F.softmax(score,dim=-1) # B*n*L*L
        attention=torch.matmul(weight,V) # B*n*L*V
        output=self.output(attention.transpose(1,2).contiguous().view(batch,seq_len,-1)) # B*L*in

        return weight , output


if __name__ == "__main__":
    dim_in,dim_k,dim_v=16,8,8
    n_head=4
    batch_size=2
    seq_len=5

    mask=torch.randn(batch_size,seq_len,seq_len)
    X=torch.randn(batch_size,seq_len,dim_in)
    Atten=selfAttention(dim_in,dim_k,dim_v)
    MultiAtten=MultiHeadAttention(dim_in,dim_k,dim_v,n_head)
    weight,output=Atten(X,mask)
    print("Input shape:",X.shape)
    print("Attention weight shape:",weight.shape)
    print("Output shape",output.shape)



