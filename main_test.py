"""
这个公式实现的是一个基于策略梯度的方法，用于更新策略参数 \(\theta\)。我们可以通过编写一个神经网络来实现策略的近似，然后使用反向传播来优化策略。实现这个公式的关键步骤如下：
    1. **定义策略网络**：使用一个神经网络表示策略 \(\pi(a|s, \theta)\)，即给定状态 \(s\)，输出动作 \(a\) 的概率分布。
    2. **计算策略梯度**：根据策略梯度定理，计算 \(\nabla_\theta J(\theta)\)，其中 \(J(\theta)\) 是策略的目标函数。
    3. **优化参数**：使用策略梯度公式更新参数 \(\theta\)。
具体实现可以参考如下代码：
"""

import torch
import torch.nn as nn
import torch.optim as optim

# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

# 初始化策略网络和优化器
state_dim = 4  # 状态的维度
action_dim = 2  # 动作的数量
policy_net = PolicyNetwork(state_dim, action_dim)
optimizer = optim.Adam(policy_net.parameters(), lr=0.01)

# 定义策略梯度的损失函数
def policy_gradient_loss(log_probs, rewards):
    return -torch.sum(log_probs * rewards)

# 假设已经获取到 (s, a, q(s, a)) 的数据
# s: 状态，a: 动作，q_val: 奖励/回报
s = torch.tensor([0.1, 0.2, 0.3, 0.4])  # 示例状态
q_val = 1.0  # q(s, a) 的值

# 前向传播获取动作分布并计算 log π(a|s, θ)
action_probs = policy_net(s)
log_probs = torch.log(action_probs)
action = torch.multinomial(action_probs, 1).item()  # 选取一个动作
log_prob = log_probs[action]  # 获取该动作的 log 概率

# 使用策略梯度公式计算损失并反向传播
loss = policy_gradient_loss(log_prob, q_val)
optimizer.zero_grad()
loss.backward()
optimizer.step()
"""
### 代码解释
1. `PolicyNetwork` 是策略网络，输入状态 \( s \)，输出每个动作的概率。
2. 在 `policy_gradient_loss` 函数中，使用 \(\nabla_\theta J(\theta)\) 作为目标，即 \(-q(s, a) \cdot \nabla_\theta \ln \pi(a|s, \theta)\)。
3. 通过反向传播 (`loss.backward()`) 计算梯度，然后使用优化器更新参数
"""