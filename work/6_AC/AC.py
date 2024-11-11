"""
QAC 算法实现
该算法是AC算法的一种经典算法，是使用了Sarsa算法来做实现
在Critic和Actor方面使用了两个神经网络来实现
"""
from environment import *
from network import *
import torch

class AC(Board):
    ## 属性
    # 最大迭代次数
    iteration_count = 1000000
    # 迭代终止条件
    minimize_error = 1e-4

    # 神经网络
    critic = None
    actor = None

    # 重要性采样参数
    beita = None        # 用于采样的权重矩阵
    sum_beita = None    # 所有权重的和
    beita_q = None      # 权重矩阵对应的参数矩阵
    def get_action_by_probability(self, x, y):
        """
        基于模型的方法，根据模型给出的概率选择行动
        :param x: x
        :param y: y
        :return: a
        """
        probabilities = predict([x, y], self.actor)
        action_list = list(range(self.sum_action))

        action = random.choices(action_list, probabilities, k=1)[0]
        return action

    def get_sample(self):
        """
        获取一个样本
        :return: (s_t, a_t, r, s_{t+1}, a_{t+1})
        """
        # 随机选择一个起始点及其行动和回报
        x = random.randint(0, self.n - 1)
        y = random.randint(0, self.m - 1)

        a = self.get_action_by_probability(x, y)
        r = self.get_reward_by_action(x, y, a)

        # 下一个状态值
        next_x, next_y = self.next_point(x, y, a)
        # 越界
        if self.is_crossing_boundaries(next_x, next_y):
            next_x, next_y = x, y
        next_a = self.get_action_by_probability(next_x, next_y)
        return (x, y), a, r, (next_x, next_y), next_a

    def get_values(self, x, y):
        """
        根据模型返回一个状态下所有的行动值
        :param x: x
        :param y: y
        :return: [[a0,a1,a2...]]
        """
        input_date = [[x, y, a] for a in range(self.sum_action)]
        values = predict(input_date, self.critic)
        values = np.array(values).flatten().tolist()
        return values

    def update_all_action_value(self):
        """
        更新所有状态的行动值用于debug
        """
        for x in range(0, self.n):
            for y in range(0, self.m):
                self.state_action_values[x][y] = self.get_values(x, y)
                probabilities_value = predict([x, y], self.actor)
                probabilities_value = np.array(probabilities_value).flatten().tolist()

                # 更新
                self.state_action_probabilities[x][y] = probabilities_value

    def run_QAC(self):
        """
        QAC算法实现
        :return: None
        """
        # 初始化神经网络
        self.critic = QAC_critic(3, 1)  # 用于生成行动值
        self.actor = AC_actor(2, self.sum_action)  # 用于给出一个状态的不同行动的概率

        for each in range(self.iteration_count):
            # 采样
            (x, y), a, r, (next_x, next_y), next_a = self.get_sample()

            ## Critic(value update)
            critic_input_date = [[x, y, a]]  # 输入
            q_hat = r + self._lambda * predict([next_x, next_y, next_a], self.critic)[0]

            critic_output_date = [[q_hat]]  # 输出
            # 训练
            train_network(self.critic, critic_input_date, critic_output_date)

            ## Actor（policy update）
            actor_input_date = [[x, y]]
            actor_output_date = [self.get_values(x, y)]

            # 训练
            train_network(self.actor, actor_input_date, actor_output_date, loss_function=ac_loss)
            # print(predict([[3, 2, 0], [3, 2, 1], [3, 2, 2], [3, 2, 3], [3, 2, 4]], self.critic, False))
            # print(predict([3, 2], self.actor))
            if each % 20 == 0:
                self.update_all_action_value()
                print(f"已完成{each + 1}次迭代")
                self.debug(self.state_action_values)
                # self.debug(self.state_action_probabilities)

        self.show("QAC 算法实现")

    def update_all_action_value_by_A2C(self):
        """
        更新所有状态的行动值用于debug
        return: 误差情况
        """
        error_count = 0

        for x in range(0, self.n):
            for y in range(0, self.m):
                self.state_action_values[x][y] = predict([x, y], self.critic)
                probabilities_value = predict([x, y], self.actor)
                probabilities_value = np.array(probabilities_value).flatten().tolist()

                # 计算误差
                error_count += sum(abs(a - b) for a, b in zip(probabilities_value, self.state_action_probabilities[x][y]))

                # 更新
                self.state_action_probabilities[x][y] = probabilities_value

        return error_count

    def run_A2C(self):
        """
        A2C 算法实现
        A2C 相较于QAC引入了一个偏置，该偏置在本算法中表现为v，其余与QAC相同
        :return: None
        """
        # 初始化神经网络
        self.critic = A2C_critic(2, self.sum_action)
        self.actor = AC_actor(2, self.sum_action)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0005)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

        for each in range(self.iteration_count):
            # 获取样本
            (x, y), a, r, (next_x, next_y), next_a = self.get_sample()

            state = torch.FloatTensor([x, y]).unsqueeze(0)
            next_state = torch.FloatTensor([next_x, next_y]).unsqueeze(0)
            reward = torch.FloatTensor([r])

            ## 计算TD误差
            v_s = self.critic(state)
            v_s_next = self.critic(next_state)
            td_error = reward + self._lambda * v_s_next - v_s  # δ_t = r + γV(s') - V(s)

            ## Critic（value update） 训练
            critic_loss = td_error.pow(2).mean()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            ## Actor（policy update）
            action_probs = self.actor(state)
            action_log_prob = torch.log(action_probs[0, a])
            actor_loss = -action_log_prob * td_error.detach().mean()

            # 训练
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if each % 1000 == 999:
                print(f'第{each + 1}轮迭代结束')
                error_count = self.update_all_action_value_by_A2C()
                print(f"当前误差为{error_count}")

                if error_count < self.minimize_error:
                    print("迭代终止")
                    break

                self.debug(self.state_action_probabilities)

        self.show('A2C 算法实现')

    def init_beita(self):
        """
        初始化参数权重矩阵
        权重的计算方式按照n+m+1-该点到终点的路径长度
        :return: None
        """
        # 初始化终点位置
        x0, y0 = -1, -1
        # 初始化权重矩阵和参数矩阵
        self.beita = [[0 for _ in range(self.m)] for _ in range(self.n)]
        self.beita_q = [[0 for _ in range(self.m)] for _ in range(self.n)]
        self.sum_beita = 0

        # 寻找终点位置
        for i in range(self.n):
            for j in range(self.m):
                if self.chessboard[i][j] == 'X':
                    x0, y0 = i, j
                    break
            if x0 != -1:
                break

        # 从终点出发，计算每一个点应有的权重
        q = [[x0, y0]]
        self.beita[x0][y0] = 1
        minn = 0
        while q:
            x, y = q.pop(0)
            # 遍历每一个点
            for i in range(1, 5):
                x1 = x + self.xx[i]
                y1 = y + self.yy[i]
                if not self.is_crossing_boundaries(x1, y1) and self.beita[x1][y1] == 0:
                    self.beita[x1][y1] = self.beita[x][y] + 1
                    minn = max(minn, self.beita[x1][y1])
                    q.append([x1, y1])
        # 权重矩阵计算
        minn = (self.n + self.m - minn + 1) ** 2 - 1
        for i in range(self.n):
            for j in range(self.m):
                self.beita[i][j] = (self.n + self.m + 1 - self.beita[i][j]) ** 2 - minn
                self.sum_beita += self.beita[i][j]

        # 参数矩阵计算
        for i in range(self.n):
            for j in range(self.m):
                self.beita_q[i][j] = self.sum_beita / self.beita[i][j] / self.n / self.m

        # 权重矩阵一维化
        print("权重矩阵")
        self.debug(self.beita)

        print("参数矩阵")
        self.debug(self.beita_q)

        self.beita = np.array(self.beita).flatten().tolist()


    def get_importance_sample(self):
        """
        重要性采样算法实现
        :return: None
        """
        # 随机选择一个起始点及其行动和回报
        s_list = range(self.n * self.m)
        s = random.choices(s_list, self.beita, k=1)[0]
        x = s // self.m
        y = s - x * self.m

        a = self.get_action_by_probability(x, y)
        r = self.get_reward_by_action(x, y, a)

        # 下一个状态值
        next_x, next_y = self.next_point(x, y, a)
        # 越界
        if self.is_crossing_boundaries(next_x, next_y):
            next_x, next_y = x, y
        next_a = self.get_action_by_probability(next_x, next_y)
        return (x, y), a, r, (next_x, next_y), next_a


    def run_off_policy_A2C(self):
        """
        off policy 版的A2C 算法实现
        off policy A2C 相较于A2C引入了重要性采样的概念。即每一个样本被采样的概率会有所不同
        同样的，也要处理由于采样的不同而导致的偏重不同
        :return: None
        """
        # 初始化权重参数
        self.init_beita()

        # 初始化神经网络
        self.critic = A2C_critic(2, self.sum_action)
        self.actor = AC_actor(2, self.sum_action)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0005)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

        for each in range(self.iteration_count):
            # 获取样本
            (x, y), a, r, (next_x, next_y), next_a = self.get_importance_sample()

            state = torch.FloatTensor([x, y]).unsqueeze(0)
            next_state = torch.FloatTensor([next_x, next_y]).unsqueeze(0)
            reward = torch.FloatTensor([r])

            ## 计算TD误差
            v_s = self.critic(state)
            v_s_next = self.critic(next_state)
            td_error = reward + self._lambda * v_s_next - v_s  # δ_t = r + γV(s') - V(s)

            ## Critic（value update） 训练
            critic_loss = self.beita_q[x][y] * td_error.pow(2).mean()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            ## Actor（policy update）
            action_probs = self.actor(state)
            action_log_prob = torch.log(action_probs[0, a])
            actor_loss = self.beita_q[x][y] * (-action_log_prob * td_error.detach().mean())

            # 训练
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if each % 1000 == 999:
                print(f'第{each + 1}轮迭代结束')
                error_count = self.update_all_action_value_by_A2C()
                print(f"当前误差为{error_count}")

                if error_count < self.minimize_error:
                    print("迭代终止")
                    break

                self.debug(self.state_action_probabilities)

        self.show('A2C 算法实现')

    def get_sample_by_DPG(self, s=None):
        """
        专用于DPG算法的样本生成
        注意以下对象都是tensor.LongTensor类型的
        :return: s_t, a, r, s_{t+1}, a_{t+1}
        """
        # 随机生成一个状态
        if s is None:
            x = random.randint(0, self.n - 1)
            y = random.randint(0, self.m - 1)
            s = torch.LongTensor([x, y]).unsqueeze(0)
        else:
            x, y = s[0].tolist()
        # 生成一个行动，计算回报，和下一个状态，以及下一个状态的行动
        a = self.actor(s)
        r = self.get_reward_by_action(x, y, a)

        next_x, next_y = self.next_point(x, y, a)
        # 越界
        if self.is_crossing_boundaries(next_x, next_y):
            next_x, next_y = x, y
        # 生成下一个点的状态和行动
        next_s = torch.LongTensor([next_x, next_y]).unsqueeze(0)

        next_a = self.actor(next_s)

        return s, a, r, next_s, next_a

    def update_policy_by_DPG(self):
        """
        更新策略
        :return:None
        """
        for x in range(self.n):
            for y in range(self.m):
                s = torch.LongTensor([x, y]).unsqueeze(0)
                self.policy[x][y] = self.actor(s).tolist()[0]

                self.state_action_probabilities[x][y] = [0 for _ in range(self.m)]
                self.state_action_probabilities[x][y][self.policy[x][y]] = 1
    def run_DPG(self):
        """
        DPG算法实现
        DPG的决策模型actor不再是给出一个概率而是直接给出一个确定的行动
        :return: None
        """
        # 模型初始化，这里输入的是状态和行动的值域大小(所以默认这里是一个方阵)，用于构造嵌入层
        self.actor = DeterministicActor(self.n, self.sum_action)    # 可以通过给入一个状态返回一个行动
        self.critic = DeterministicCritic(self.n, self.sum_action)  # 可以通过给一个状态和行动返回q_value行动价值

        # 定义优化器
        actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0001)
        critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.0001)

        # 遍历
        # next_s = None
        for each in range(self.iteration_count):
            # 采样（示例：(tensor([[2, 1]]), tensor([2]), -5, tensor([[3, 1]]), tensor([0]))）
            s, a, r, next_s, next_a = self.get_sample_by_DPG()

            # TD error(计算误差)
            q_value = self.critic(s, a)
            next_q_value = self.critic(next_s, next_a)
            td_target = r + self._lambda * next_q_value

            # critic (update value)
            critic_loss = nn.MSELoss()(q_value, td_target)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # actor (update policy)
            actor_loss = -self.critic(s, a)
            actor_optimizer.zero_grad()
            actor_loss.backward()

            actor_optimizer.step()

            if each % 1000 == 0:
                print(f"Iteration {each}: Actor Loss = {actor_loss}, Critic Loss = {critic_loss}")

                self.update_policy_by_DPG()
                self.show()


    def __init__(self):
        super().__init__()
        # 初始化神经网络




if __name__ == '__main__':
    ac = AC()
    ac.run_DPG()
