"""
ReinforceAgent 算法实现
这是一种基于蒙特卡洛是算法，但是策略使用了模型（函数）的方法，而不是表格，这种方法就需要使用神经网络来实现。
"""
import torch
from environment import *
from reinforce_network import *
import numpy as np

class ReinforceAgent(Board):
    ## 属性，最大迭代次数
    iteration_number = 200
    # 路径长度
    track_length = 1000
    # 后k个不参与计算的状态
    k = 50
    # 迭代终止条件，最小误差
    minimum_error = 1e-9

    # 神经网络
    net = None

    def softmax(self, t):
        """
        softmax 公式实现
        :param t: t
        :return: list
        """
        return torch.softmax(torch.tensor(t, dtype=torch.float32), dim=-1).tolist()

    def get_action_by_net(self, x, y):
        """
        根据神经网络的返回值来随机给出一个行动
        :param x: x
        :param y: y
        :return: a
        """
        # date_input = [[x, y, a] for a in range(self.sum_action)]
        # value_list = predict(date_input, self.net)
        # value_list = np.array(value_list).flatten()  # 这里将二维矩阵一维化为向量
        #
        # probability_list = self.softmax(value_list)

        action_list = range(self.sum_action)
        probability_list = predict([x, y], self.net)
        a = random.choices(action_list, probability_list, k=1)[0]
        return a

    def get_track(self):
        """
        生成一个路径
        :return: [(s_t,a_t,r_{t+1}),...]
        """
        # 初始化路径
        T = []

        # 创造起始点
        x = random.randint(0, self.n - 1)
        y = random.randint(0, self.m - 1)

        for _ in range(self.track_length):
            a = self.get_action_by_net(x, y)
            r = self.get_reward_by_action(x, y, a)

            next_x, next_y = self.next_point(x, y, a)
            # 越界
            if self.is_crossing_boundaries(next_x, next_y):
                next_x, next_y = x, y

            # 更新
            T.append([(x, y), a, r])
            x, y = next_x, next_y

        return T

    def run_Reinforce(self):
        """
        5_Reinforce 算法实现
        :return: None
        """
        for each in range(self.iteration_number):
            # 随机获取一条路径用于更新
            T = self.get_track()
            # 初始化误差
            error_count = 0

            ## 更新value
            input_date = []
            g = 0
            # 列表翻转
            T.reverse()
            for i, t in enumerate(T):
                (x, y), a, r = t
                g = r + self._lambda * g
                if i >= self.k:
                    # 更新输入数据
                    input_date.append([x, y])

                    # 累计误差
                    error_count += abs(self.state_action_values[x][y][a] - g)

                    # 更新期望状态
                    self.state_action_values[x][y][a] = g
            if error_count <= self.minimum_error:
                print(f"模型在第{each}次之后收敛")
                break

            ## 更新policy（即训练）
            # 整理输入输出数据
            output_date = []
            for x, y in input_date:
                y_hat = self.softmax(self.state_action_values[x][y])  # 这里需要转化为概率，所以使用了归一化函数
                output_date.append(y_hat)

            # 训练

            train_network(self.net, input_date, output_date)

            if each % 1 == 0:
                print(f"第{each + 1}轮迭代已完成")
                self.debug(self.state_action_values)
                self.update_probabilities()
                self.debug(self.state_action_probabilities)

        print("5_Reinforce 算法实现")
        self.show()

    def update_probabilities(self):
        """
        保存所有状态的模型输出
        :return: None
        """
        for x in range(self.n):
            for y in range(self.m):
                # input_date = [[x, y, a] for a in range(self.sum_action)]
                # value_list = predict(input_date, self.net)
                # value_list = np.array(value_list).flatten()
                # self.state_action_probabilities[x][y] = cp(value_list)
                self.state_action_probabilities[x][y] = predict([x, y], self.net)

    def update_policy(self):
        """
        根据最大概率更新策略
        :return: None
        """
        for x in range(self.n):
            for y in range(self.m):
                max_index = np.argmax(predict([x, y], self.net))  # 返回的最大下标（最优行动值）
                self.policy[x][y] = max_index

    def __init__(self):
        super(ReinforceAgent, self).__init__()

        # 初始化神经网络
        self.net = ReinForceNetwork(2, self.sum_action)


if __name__ == '__main__':
    ra = ReinforceAgent()
    ra.run_Reinforce()
