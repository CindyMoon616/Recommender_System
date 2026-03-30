import torch
import numpy as np

def softmax(X):
    X_max=torch.max(X,dim=-1,keepdim=True)[0]
    X-=X_max
    X_exp=torch.exp(X)
    return X_exp/(torch.sum(X_exp,dim=-1,keepdim=True))

def softmax_np(X):
    X_max=np.max(X,axis=-1,keepdims=True)
    X-=X_max
    X_exp=np.exp(X)
    return X_exp/(np.sum(X_exp,axis=-1,keepdims=True))

if __name__=="__main__":
    batch_size,seq_len,embed=4,3,2
    X=torch.randn(batch_size,seq_len,embed)
    X_np=np.random.randn(batch_size,seq_len,embed)
    print("原始数据:",X_np)
    print("Softmax后:",softmax_np(X_np))
