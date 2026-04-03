from sklearn.model_selection import train_test_split
import numpy as np 

X=np.random.randn(10240,8)
Y=np.random.randn(10240)

# 数据预处理
ones=np.ones(shape=(X.shape[0],1))
X=np.hstack([X,ones])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

# batch函数
def get_batch(batch_size,X,Y):
    batchnum=X.shape[0]//batch_size
    X_new=X.reshape((batchnum,batch_size,X.shape[1]))
    Y_new=Y.reshape(batchnum,batch_size,)
    for i in range(batchnum):
        yield X_new[i,:,:],Y_new[i,:]

# 损失函数
def mse(X,Y,W):
    return 0.5*np.mean(np.square(X@W-Y))

def diff_mse(X,Y,W):
    return X.T@(X@W-Y)/X.shape[0]

# SGD
lr=0.001
batch_size=64
num_epochs=100
evaluate_time=10
def train(num_epochs,batch_size,W,X_train, X_test, Y_train, Y_test):
    loss_train=[]
    loss_evaluate=[]
    for epoch in range(num_epochs):
        loss_train_epoch=0
        for x_batch,y_batch in get_batch(batch_size,X_train,Y_train):
            loss_batch=mse(x_batch,y_batch,W)
            loss_train_epoch += loss_batch*x_batch.shape[0]/X_train.shape[0]
            grad=diff_mse(x_batch,y_batch,W)
            W=W-lr*grad
            
        loss_train.append(loss_train_epoch)
        if 0==epoch % evaluate_time:
            loss_evaluate_epoch=mse(X_test,Y_test,W)
            loss_evaluate.append(loss_evaluate_epoch)
            print(f'Epoch: {epoch}, train loss: {loss_train_epoch}, val loss: {loss_evaluate_epoch}')


if __name__=="__main__":
    W=np.random.randn(X.shape[1],)
    train(num_epochs,batch_size,W,X_train, X_test, Y_train, Y_test)
            


