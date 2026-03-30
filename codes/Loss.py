import numpy as np


def mse_loss(y_true,y_pred):
    diff=y_true-y_pred
    sq_diff = np.square(diff)
    loss=np.mean(sq_diff)

    return loss

def cross_entropy_loss(y_true,y_pred):
    epsilon=1e-15
    # F.binary_cross_entropy_with_logits
    # nn.BCEWithLogitsLoss
    y_pred=np.clip(y_pred,epsilon,1-epsilon)
    loss=-np.mean(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))
    return loss

def multi_cross_entropy_loss(y_true,y_pred):
    epsilon=1e-15
    # nn.CrossEntropyLoss
    y_pred=np.clip(y_pred,epsilon,1-epsilon)
    loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))
    return loss

if __name__ == "__main__":
    batch_size=32
    length=10
    y_true=np.random.randn(batch_size,length)
    y_pred=y_true+np.random.randn(batch_size,length)*0.01
    loss=mse_loss(y_true,y_pred)
    loss2=cross_entropy_loss(y_true,y_pred)
    loss3=multi_cross_entropy_loss(y_true,y_pred)
    print("MSE loss:",loss)
    print("Cross Entropy Loss:",loss2)
    print("Muliti Class Cross Entropy Loss:",loss3)