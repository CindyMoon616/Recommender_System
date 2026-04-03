import torch
from torch.utils.data import DataLoader, TensorDataset


X = torch.randn(10240, 8, dtype=torch.float32)


true_w = torch.randn(8,1)
true_b = torch.randn(1,)
Y = X @ true_w + true_b + torch.randn(10240, 1) * 0.01


#初始化模型参数
w = torch.randn((8, 1), requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

# 定义损失函数和优化器
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD([w, b], lr=0.01)

# 4. 定义超参数
batch_size = 32  # 对于 10240 条数据，建议增加 batch_size 提高计算效率
epochs = 50

# 创建 DataLoader
dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 5. 随机梯度下降循环
for epoch in range(epochs):
    epoch_loss = 0
    for x_batch, y_batch in loader:
        # 前向传播：使用矩阵乘法 @ (形状: [batch_size, 8] @ [8, 1] -> [batch_size, 1])
        y_pred = x_batch @ w + b
        
        # 计算损失
        loss = loss_fn(y_pred, y_batch)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        epoch_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(loader):.4f}")

# 输出结果
print("\n训练完成！")
print(f"学习到的 w: \n{w.detach().numpy()}")
print(f"学习到的 b: {b.item():.4f}")