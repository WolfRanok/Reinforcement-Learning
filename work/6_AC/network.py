"""
Deep q-learning network
辅助神经网络
输入[s_x,s_y,a]
输出：[q^hat]
"""
from torch.nn.init import kaiming_uniform_
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F

# 学习率
learning_rate = 0.001


## 用于actor(policy update)的神经网络模型
class AC_actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(AC_actor, self).__init__()

        # 输入层
        self.fc1 = nn.Linear(input_size, 256)  # 生成适合卷积层的特征数
        kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        self.act1 = nn.ReLU()

        # 第二个全连接层
        self.fc2 = nn.Linear(256, 64)  # 假设卷积输出大小为 4x4
        kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        self.act2 = nn.ReLU()

        # 输出层
        self.fc3 = nn.Linear(64, output_size)
        self.act3 = nn.Softmax(dim=-1)

    def forward(self, x):
        # 输入层
        x = self.fc1(x)
        x = self.act1(x)

        # 全连接层
        x = self.fc2(x)
        x = self.act2(x)

        # 输出层
        x = self.fc3(x)
        x = self.act3(x)
        return x

# 用于Critic(value update)的神经网络模型
class QAC_critic(nn.Module):
    def __init__(self, input_size, output_size):
        super(QAC_critic, self).__init__()
        # 输入层
        self.fc1 = nn.Linear(input_size, 64)
        kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        self.act1 = nn.ReLU()

        # 全连接层
        self.fc2 = nn.Linear(64, 64)
        kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        self.act2 = nn.ReLU()

        # 输出层
        self.fc3 = nn.Linear(64, output_size)
        self.act3 = nn.Softplus()

    def forward(self, x):
        # 输入层
        x = self.fc1(x)
        x = self.act1(x)

        # 全连接层
        x = self.fc2(x)
        x = self.act2(x)

        # 输出层
        x = self.fc3(x)
        x = self.act3(x)
        return x


class A2C_critic(nn.Module):
    def __init__(self, input_size, output_size):
        super(A2C_critic, self).__init__()
        # 输入层 (全连接层)
        self.fc1 = nn.Linear(input_size, 256)  # 输出更大的特征维度，以适应卷积操作
        kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        self.act1 = nn.ReLU()

        # 第二个全连接层
        self.fc2 = nn.Linear(256, 64)  # 假设卷积输出大小为 4x4
        kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        self.act2 = nn.ReLU()

        # 输出层
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        # 输入层
        x = self.fc1(x)
        x = self.act1(x)

        # 全连接层
        x = self.fc2(x)
        x = self.act2(x)

        # 输出层
        x = self.fc3(x)
        return x


# 确定性 Actor 网络
class DeterministicActor(nn.Module):
    def __init__(self, state_dim, action_dim, embedding_dim=32):
        """
        :param state_dim: 状态的值域大小
        :param action_dim: 行动的值域大小
        :param embedding_dim: 嵌入层的规模
        """
        super(DeterministicActor, self).__init__()
        # 状态的每个维度使用独立的嵌入层
        self.x_embedding = nn.Embedding(state_dim, embedding_dim)
        self.y_embedding = nn.Embedding(state_dim, embedding_dim)

        # 全连接层
        self.fc1 = nn.Linear(embedding_dim * 2, 64)
        self.fc2 = nn.Linear(64, action_dim)

    def forward(self, state):
        """
        前向传播，输入一个动作，输出一个确定的行动
        :param state: 状态
        :return: 行动
        """
        # 将状态的两个离散维度 (X, Y) 分别嵌入
        x = self.x_embedding(state[:, 0])  # 提取 X 维度
        y = self.y_embedding(state[:, 1])  # 提取 Y 维度

        # 将 X 和 Y 嵌入向量拼接
        state_embedded = torch.cat([x, y], dim=-1)

        # 全连接层计算动作的分值
        x = F.relu(self.fc1(state_embedded))
        action_values = self.fc2(x)


        action_probs = F.softmax(action_values, dim=-1)  # 使用 softmax 计算概率分布

        action = torch.multinomial(action_probs, 1)  # 从中采样
        return action[0]


# Critic 网络
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeterministicCritic(nn.Module):
    def __init__(self, state_dim, action_dim, embedding_dim=32, hidden_dim=64):
        """
        :param state_dim: 状态的值域大小
        :param action_dim: 行动的值域大小
        :param embedding_dim: 嵌入层的规模
        :param hidden_dim: LSTM 隐藏层大小
        """
        super(DeterministicCritic, self).__init__()
        # 状态的两个嵌入层和动作的嵌入层
        self.x_embedding = nn.Embedding(state_dim, embedding_dim)
        self.y_embedding = nn.Embedding(state_dim, embedding_dim)
        self.action_embedding = nn.Embedding(action_dim, embedding_dim)

        # LSTM 层，输入大小为 embedding_dim * 3，输出大小为 hidden_dim
        self.lstm = nn.LSTM(input_size=embedding_dim * 3, hidden_size=hidden_dim, batch_first=True)

        # 全连接层
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, state, action):
        """
        输入一个确定的状态和行动，返回一个评价值
        :param state: 状态
        :param action: 行动
        :return: q_value
        """
        # 嵌入状态的 X 和 Y 维度
        x = self.x_embedding(state[:, 0])  # 提取 X 维度
        y = self.y_embedding(state[:, 1])  # 提取 Y 维度
        action_embedded = self.action_embedding(action)  # 嵌入动作

        # 拼接状态和动作的嵌入向量
        combined = torch.cat([x, y, action_embedded], dim=-1).unsqueeze(1)  # 添加时间维度，形状为 [batch_size, 1, embedding_dim * 3]

        # LSTM 层计算
        lstm_out, _ = self.lstm(combined)  # 输出形状为 [batch_size, 1, hidden_dim]
        lstm_out = lstm_out.squeeze(1)  # 移除时间维度，形状为 [batch_size, hidden_dim]

        # 全连接层计算 Q 值
        x = F.relu(self.fc1(lstm_out))
        q_value = self.fc2(x)
        return q_value


def ac_loss(probs, q_value):
    """
    自定义损失函数
    :param probs:模型输出
    :param q_value: 预测值
    :return: None
    """
    return -torch.sum(torch.log(probs) * q_value)


def train_network(model, inputs, outputs, loss_function=nn.MSELoss(), epochs=100, learning_rate=learning_rate):
    """
    为给定的模型进行训练
    :param model: 模型
    :param inputs: 输入数据
    :param outputs: 输出数据
    :param loss_function: 损失函数，默认为均方误差
    :param epochs: 一组数据块的大小
    :param learning_rate: 学习率
    :return: None
    """
    # 将数据转换为Tensor
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).transpose(0,
                                                                                     1)  # 转换为 (n_samples, n_features)
    outputs_tensor = torch.tensor(outputs, dtype=torch.float32).unsqueeze(0).transpose(0, 1)

    # 创建数据加载器
    dataset = TensorDataset(inputs_tensor, outputs_tensor)
    dataloader = DataLoader(dataset, batch_size=len(inputs), shuffle=True)

    # 定义损失函数和优化器
    criterion = loss_function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):

        for inputs_batch, outputs_batch in dataloader:
            # 前向传播
            outputs_pred = model(inputs_batch)
            loss = criterion(outputs_pred, outputs_batch)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# 对一行数据进行预测
# 对一行数据进行预测
def predict(row, model):
    # 转换原数据并添加批次维度
    row = torch.tensor([row], dtype=torch.float32)
    # 预测结果
    yhat = model(row)

    # 转为numpy数组并去掉批次维度
    yhat = yhat.detach().numpy()[0]

    return yhat


if __name__ == '__main__':
    actor = DeterministicActor(5, 5)
    critic = DeterministicCritic(5, 5)

    # 示例输入
    state = torch.tensor([[2, 3]], dtype=torch.long)  # (X=2, Y=3) 的状态
    action = torch.tensor([1], dtype=torch.long)  # 动作为 1

    print(actor(state))
    print(critic(state, action)[0])
