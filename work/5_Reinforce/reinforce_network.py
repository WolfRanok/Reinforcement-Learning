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

# 学习率
learning_rate = 0.001


class ReinForceNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(ReinForceNetwork, self).__init__()

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


def reinforce_loss(probs, q_value):
    """
    自定义损失函数
    :param probs:模型输出
    :param q_value: 预测值
    :return: None
    """
    return -torch.sum(torch.log(probs) * q_value)


def train_network(model, inputs, outputs, epochs=100, learning_rate=learning_rate):
    # 将数据转换为Tensor
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).transpose(0,1)  # 转换为 (n_samples, n_features)
    outputs_tensor = torch.tensor(outputs, dtype=torch.float32).unsqueeze(0).transpose(0, 1)

    # 创建数据加载器
    dataset = TensorDataset(inputs_tensor, outputs_tensor)
    dataloader = DataLoader(dataset, batch_size=len(inputs), shuffle=True)

    # 定义损失函数和优化器
    criterion = reinforce_loss
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
def predict(row, model):
    # 转换原数据并添加批次维度
    row = torch.tensor([row], dtype=torch.float32)

    # 预测结果
    yhat = model(row)

    # 去掉批次维度并转换为numpy数组
    yhat = yhat.detach().numpy()[0]
    return yhat


def copy_model(model1, model2):
    """
    将model2模型的参数复制给model1模型
    :param model1: 模型1
    :param model2: 模型2
    :return: None
    """
    model1.load_state_dict(model2.state_dict())


if __name__ == '__main__':
    model = ReinForceNetwork(2,5)

    date_input = [[0, 0]]
    date_output = [[0,0.1,0.2,0.3,0.4]]

    train_network(model, date_input, date_output)


    print(predict([0, 0], model))
