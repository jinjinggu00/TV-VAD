import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子
torch.manual_seed(0)

# 定义独热向量矩阵 A (7x7)
A = torch.eye(7)  # 创建一个 7x7 的单位矩阵

# 定义一个可学习的随机矩阵 B (7x7)
B = torch.rand((7, 7), requires_grad=True)  # 随机初始化 B

# 设置学习率和优化器
learning_rate = 0.01
optimizer = optim.Adam([B], lr=learning_rate)

# 设置训练次数
num_epochs = 100  # 可以手动设置训练次数

# 训练过程
for epoch in range(num_epochs):
    optimizer.zero_grad()  # 清空梯度

    # 计算 KL 散度
    kl_divergence = nn.KLDivLoss(reduction='batchmean')(torch.log_softmax(B, dim=1), torch.softmax(A, dim=1))

    # 反向传播
    kl_divergence.backward()

    # 更新矩阵 B
    optimizer.step()

    # 每 100 次打印一次损失
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], KL Divergence: {kl_divergence.item():.4f}')


# 可视化矩阵 A 和 B
def plot_matrix(matrix, title, filename):
    plt.imshow(matrix.detach().numpy(), cmap='Blues', vmin=0, vmax=1)
    plt.colorbar()
    plt.title(title)
    plt.xticks(np.arange(7), np.arange(1, 8))
    plt.yticks(np.arange(7), np.arange(1, 8))
    plt.savefig(filename)
    plt.show()


# 保存和显示矩阵 A 和 B
plot_matrix(A, 'Matrix A (One-hot Vector)', 'matrix_A.png')
plot_matrix(B, 'Matrix B (Learned Distribution)', 'matrix_B.png')


