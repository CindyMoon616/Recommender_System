import numpy as np


def mse_loss(y_true,y_pred):
    ''' loss = (y-y')^2 '''
    diff=y_true-y_pred
    sq_diff = np.square(diff)
    loss=np.mean(sq_diff)

    return loss

def cross_entropy_loss(y_true,y_pred):
    ''' loss = -(y*log(p) + (1-y)log(1-p)) '''
    epsilon=1e-15
    # F.binary_cross_entropy_with_logits
    # nn.BCEWithLogitsLoss
    y_pred=np.clip(y_pred,epsilon,1-epsilon)
    loss=-np.mean(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))
    return loss

def multi_cross_entropy_loss(y_true,y_pred):
    ''' loss = -sum(yi*log(pi)) '''
    epsilon=1e-15
    # nn.CrossEntropyLoss
    y_pred=np.clip(y_pred,epsilon,1-epsilon)
    loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))
    return loss

def focal_loss(y_true,y_pred,alpha=0.25,gamma=2):
    ''' loss = -(alpha * (1-pt)^gamma * log(pt)) '''
    epsilon=1e-15
    y_pred=np.clip(y_pred,epsilon,1-epsilon)
    pt=np.sum(y_pred*y_true,axis=1) # 这里的y_true是one-hot
    loss=-alpha*(1-pt)**gamma*np.log(pt)
    return loss
    
    

if __name__ == "__main__":
    batch_size=32
    classes=10
    y_true=np.random.randn(batch_size,1)
    y_pred=y_true+np.random.randn(batch_size,1)*0.01
    y_true_multi=np.random.randn(batch_size,classes)
    y_pred_multi=y_true+np.random.randn(batch_size,classes)*0.01
    loss=mse_loss(y_true,y_pred)
    loss2=cross_entropy_loss(y_true,y_pred)
    loss3=multi_cross_entropy_loss(y_true_multi,y_pred_multi)
    loss4=focal_loss(y_true_multi,y_pred_multi)
    print("MSE loss:",loss)
    print("Cross Entropy Loss:",loss2)
    print("Muliti Class Cross Entropy Loss:",loss3)
    print("Focal Loss:",loss4)