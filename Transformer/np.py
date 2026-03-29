import numpy as np 
from math import sqrt


class selfAttention:
    def __init__(self,dim_in,dim_k,dim_v):
        self.dim_in,self.dim_k,self.dim_v=dim_in,dim_k,dim_v

        self.W_q=np.random.randn(dim_in,dim_k) * 0.01
        self.b_q=np.zeros(dim_k)

        self.W_k=np.random.randn(dim_in,dim_k) * 0.01
        self.b_k=np.zeros(dim_k)

        self.W_v=np.random.randn(dim_in,dim_v) * 0.01
        self.b_v=np.zeros(dim_v)

        self.W_o=np.random.randn(dim_v,dim_in) * 0.01
        self.b_o=np.zeros(dim_in)

        self.norm=1/sqrt(dim_k)

    def forward(self,X,mask=None):
        # X : batch * seq_len * embed
        Q=X @ self.W_q +self.b_q # B*L*K
        K=X @ self.W_k +self.b_k # B*L*K
        V=X @ self.W_v +self.b_v # B*L*V

        score= Q @ K.transpose(0,2,1) *self.norm # B*L*L

        if mask is not None:
            score=np.where(mask==0,-1e9,score)

        # 为了数值稳定，先减去每行最大值
        score_max=np.max(score,axis=-1,keepdims=True)
        exp_score=np.exp(score-score_max)
        weight = exp_score/np.sum(exp_score,axis=-1,keepdims=True)

        atten=weight @ V # B*L*V
        output=atten @ self.W_o +self.b_o
        return weight,output

class MultiHeadAttention:
    def __init__(self,dim_in,dim_k,dim_v,n_head):
        self.dim_in,self.dim_k,self.dim_v=dim_in,dim_k,dim_v
        self.n_head=n_head

        self.W_q=np.random.randn(dim_in,dim_k)*0.01
        self.b_q=np.zeros(dim_k)

        self.W_k=np.random.randn(dim_in,dim_k)*0.01
        self.b_k=np.zeros(dim_k)

        self.W_v=np.random.randn(dim_in,dim_v)*0.01
        self.b_v=np.zeros(dim_v)

        self.W_o=np.random.randn(dim_v,dim_in)*0.01
        self.b_o=np.zeros(dim_in)

        self.norm=(1/sqrt(dim_k/n_head))

    def forward(self,X,mask=None):
        batch_size,seq_len,embed=X.shape
        Q = X @ self.W_q + self.b_q # B*L*(nk)
        Q = Q.reshape(batch_size,seq_len,self.n_head,-1).transpose(0,2,1,3) # B*n*L*k

        K = X @ self.W_k + self.b_k # B*L*(nk)
        K = K.reshape(batch_size,seq_len,self.n_head,-1).transpose(0,2,1,3) # B*n*L*k

        V = X @ self.W_v + self.b_v # B*L*(nv)
        V = V.reshape(batch_size,seq_len,self.n_head,-1).transpose(0,2,1,3) # B*n*L*v

        score = Q @ K.transpose(0,1,3,2) * self.norm # B*n*L*L

        if mask is not None:
            if mask.ndim==2: # l*L -> B*n*L*L
                mask = mask[None, None, :, :]
                mask = np.broadcast_to(mask, (batch_size, n_head, mask.shape[-2], mask.shape[-1]))
            elif mask.ndim==3:
                mask = mask[:,None, :, :]
                mask = np.broadcast_to(mask, (mask.shape[0], n_head, mask.shape[-2], mask.shape[-1]))
            score=np.where(mask==0,-1e9,score)

        score_max=np.max(score,axis=-1,keepdims=True)
        score_exp=np.exp(score-score_max)
        weight=score_exp/np.sum(score_exp,axis=-1,keepdims=True)
        attention=weight @ V # B*n*L*v
        attention=attention.transpose(0,2,1,3).reshape(batch_size,seq_len,-1) # B*L*V
        output= attention @ self.W_o +self.b_o

        return weight,output



        





if __name__ == "__main__":
    dim_in,dim_k,dim_v=16,8,8
    batch_size=3
    seq_len=5
    n_head=4

    X=np.random.randn(batch_size,seq_len,dim_in)
    mask=np.random.randn(batch_size,seq_len,seq_len)

    Atten=selfAttention(dim_in,dim_k,dim_v)
    MultiAtten=MultiHeadAttention(dim_in,dim_k,dim_v,n_head)
    weight,output = MultiAtten.forward(X,mask)
    print("Input shape:",X.shape)
    print("Attention weight shape:",weight.shape)
    print("Output shape",output.shape)

        
        